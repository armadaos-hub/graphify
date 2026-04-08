#!/usr/bin/env python3
"""
ArmadaOS Institutional Knowledge Graph Compiler

Processes the entire ArmadaOS repository (17,640+ Markdown files) through
Graphify's extraction pipeline using gpt-4.1-mini for semantic extraction.

Usage:
    python armadaos_extract.py <repo_path> <output_dir>

Example:
    python armadaos_extract.py ../ArmadaOS ../ArmadaOS/org/knowledge/graph

Environment Variables:
    OPENAI_API_KEY          - Required. OpenAI API key.
    GRAPHIFY_CACHE_DIR      - Optional. Path to persistent cache directory.
    GRAPHIFY_MAX_CONCURRENCY - Optional. Max concurrent LLM calls (default: 50).
    GRAPHIFY_MODEL          - Optional. Model to use (default: gpt-4.1-mini).
    GRAPHIFY_LABEL_MODEL    - Optional. Model for community labeling (default: gpt-4.1-nano).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

# Graphify imports
from graphify.detect import detect
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions, graph_diff
from graphify.report import generate as generate_report
from graphify.export import to_json
from graphify.cache import file_hash, load_cached, save_cached, save_semantic_cache, cache_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENCY = int(os.environ.get("GRAPHIFY_MAX_CONCURRENCY", "50"))
MODEL = os.environ.get("GRAPHIFY_MODEL", "gpt-4.1-mini")
LABEL_MODEL = os.environ.get("GRAPHIFY_LABEL_MODEL", "gpt-4.1-nano")
CHUNK_SIZE = 15
MAX_FILE_CHARS = 8000
MAX_FILE_BYTES = 100 * 1024  # 100 KB — skip files larger than this
FAILURE_THRESHOLD = 0.50  # Abort if > 50% of chunks fail
MAX_RETRIES = 3

EXCLUDE_DIRS = {".git", "node_modules", ".graphify_cache", "__pycache__"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("armadaos_extract")

# Dedicated error logger — writes permanently failed chunks to graph_errors.log
_error_log: logging.Logger | None = None


def _init_error_log(output_dir: Path) -> None:
    """Initialize a dedicated file handler for graph_errors.log."""
    global _error_log
    _error_log = logging.getLogger("armadaos_extract.errors")
    _error_log.setLevel(logging.ERROR)
    fh = logging.FileHandler(output_dir / "graph_errors.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _error_log.addHandler(fh)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a knowledge graph extraction engine for ArmadaOS, an AI operating system. Read the following documents and extract entities (concepts, decisions, protocols, agents, systems, strategies, specifications) and relationships between them.

For each document, extract:
- Nodes: key concepts, entities, decisions, protocols, agents, systems, specifications mentioned
- Edges: relationships between nodes

Rules:
- Node IDs must be lowercase snake_case, globally consistent (e.g., always "armadaos_engine", never "the_engine" or "aos_engine")
- Use consistent IDs for well-known ArmadaOS concepts: armadaos_engine, vfs, gateway, genesis_db, knowledge_sync, cos_agent, shadow_agent, ame, chairman
- confidence_score: EXTRACTED=1.0 (directly stated), INFERRED=0.6-0.9 (reasonably implied)
- Focus on CROSS-DOCUMENT relationships — these are the most valuable
- Extract the "why" behind decisions (rationale_for edges), not just the "what"
- For protocols/standards, extract what they govern and what implements them

Output exactly this JSON (no other text):
{"nodes":[{"id":"snake_case_id","label":"Human Readable Name","file_type":"document","source_file":"relative/path"}],"edges":[{"source":"node_id","target":"node_id","relation":"references|implements|rationale_for|conceptually_related_to|shares_data_with|governs|supersedes|depends_on","confidence":"EXTRACTED|INFERRED","confidence_score":0.8,"source_file":"relative/path","weight":1.0}]}"""

# ---------------------------------------------------------------------------
# Async extraction
# ---------------------------------------------------------------------------

semaphore: asyncio.Semaphore


async def extract_chunk(
    client: AsyncOpenAI,
    files: list[Path],
    repo_path: Path,
    chunk_idx: int,
    total_chunks: int,
) -> dict:
    """Send a chunk of files to the LLM for semantic extraction with retries."""
    content_parts: list[str] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_FILE_CHARS:
                text = text[:MAX_FILE_CHARS] + "\n\n[... truncated ...]"
            rel = str(f.relative_to(repo_path))
            content_parts.append(f"=== FILE: {rel} ===\n{text}\n")
        except Exception as exc:
            log.warning("Could not read %s: %s", f, exc)

    combined = "\n".join(content_parts)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": EXTRACTION_PROMPT},
                        {"role": "user", "content": combined},
                    ],
                    temperature=0.1,
                    max_tokens=16000,
                    response_format={"type": "json_object"},
                )

            result = json.loads(response.choices[0].message.content)
            inp = response.usage.prompt_tokens if response.usage else 0
            out = response.usage.completion_tokens if response.usage else 0
            result["input_tokens"] = inp
            result["output_tokens"] = out

            log.info(
                "Chunk %d/%d: %d nodes, %d edges (%d in / %d out tokens)",
                chunk_idx,
                total_chunks,
                len(result.get("nodes", [])),
                len(result.get("edges", [])),
                inp,
                out,
            )
            return result

        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                "Chunk %d attempt %d failed: %s — retrying in %ds",
                chunk_idx,
                attempt,
                exc,
                wait,
            )
            await asyncio.sleep(wait)

    file_list = ", ".join(str(f.name) for f in files)
    msg = f"Chunk {chunk_idx} permanently failed after {MAX_RETRIES} attempts. Files: {file_list}"
    log.error(msg)
    if _error_log:
        _error_log.error(msg)
    return {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0, "failed": True, "failed_files": [str(f) for f in files]}


async def label_community(
    client: AsyncOpenAI,
    cid: int,
    node_labels: list[str],
) -> tuple[int, str]:
    """Ask a cheap model to produce a 2-5 word label for a community."""
    prompt = (
        "Given these 5 concepts from an AI operating system knowledge graph, "
        "provide a 2-5 word label for the community they form. "
        "Output ONLY the label, nothing else.\n\n"
        + "\n".join(f"- {lbl}" for lbl in node_labels)
    )
    try:
        async with semaphore:
            resp = await client.chat.completions.create(
                model=LABEL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=20,
            )
        label = resp.choices[0].message.content.strip().strip('"').strip("'")
        return cid, label
    except Exception as exc:
        log.warning("Failed to label community %d: %s", cid, exc)
        return cid, f"Community {cid}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(repo_path: Path, output_dir: Path) -> None:
    global semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    client = AsyncOpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)
    _init_error_log(output_dir)

    # ------------------------------------------------------------------
    # Step 1: Detect files
    # ------------------------------------------------------------------
    log.info("Step 1/7: Detecting files in %s", repo_path)
    detection = detect(repo_path)
    doc_files_raw: list[str] = detection.get("files", {}).get("document", [])

    # Filter out excluded dirs and oversized files
    doc_files: list[Path] = []
    skipped_large = 0
    skipped_dir = 0
    for fp in doc_files_raw:
        p = Path(fp)
        parts = set(p.relative_to(repo_path).parts)
        if parts & EXCLUDE_DIRS:
            skipped_dir += 1
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            skipped_large += 1
            continue
        doc_files.append(p)

    log.info(
        "Found %d document files (%d skipped: %d excluded dirs, %d oversized)",
        len(doc_files),
        skipped_dir + skipped_large,
        skipped_dir,
        skipped_large,
    )

    # ------------------------------------------------------------------
    # Step 2: Check cache
    # ------------------------------------------------------------------
    cache_root = Path(os.environ.get("GRAPHIFY_CACHE_DIR", str(output_dir / ".cache")))
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["GRAPHIFY_CACHE_ROOT"] = str(cache_root)

    cached_nodes: list[dict] = []
    cached_edges: list[dict] = []
    uncached_files: list[Path] = []

    for f in doc_files:
        cached = load_cached(f, root=cache_root)
        if cached:
            cached_nodes.extend(cached.get("nodes", []))
            cached_edges.extend(cached.get("edges", []))
        else:
            uncached_files.append(f)

    log.info(
        "Cache: %d files hit, %d files need extraction",
        len(doc_files) - len(uncached_files),
        len(uncached_files),
    )

    # ------------------------------------------------------------------
    # Step 3: Extract uncached files
    # ------------------------------------------------------------------
    total_input_tokens = 0
    total_output_tokens = 0
    new_nodes: list[dict] = []
    new_edges: list[dict] = []
    failed_chunks = 0

    if uncached_files:
        chunks = [
            uncached_files[i : i + CHUNK_SIZE]
            for i in range(0, len(uncached_files), CHUNK_SIZE)
        ]
        total_chunks = len(chunks)
        log.info(
            "Step 2/7: Extracting %d files in %d chunks (concurrency=%d)",
            len(uncached_files),
            total_chunks,
            MAX_CONCURRENCY,
        )

        start = time.monotonic()
        tasks = [
            extract_chunk(client, chunk, repo_path, i + 1, total_chunks)
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

        all_failed_files: list[str] = []
        for i, result in enumerate(results):
            if result.get("failed"):
                failed_chunks += 1
                all_failed_files.extend(result.get("failed_files", []))
                continue
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)
            chunk_nodes = result.get("nodes", [])
            chunk_edges = result.get("edges", [])
            new_nodes.extend(chunk_nodes)
            new_edges.extend(chunk_edges)

            # True per-file caching: group nodes/edges by source_file
            # so each file's cache entry contains only ITS OWN extractions
            chunk_files = chunks[i]
            saved = save_semantic_cache(
                chunk_nodes, chunk_edges, None, root=cache_root
            )
            # For files that produced no nodes/edges, save an empty entry
            # so they are not re-extracted on the next run
            cached_source_files = {n.get("source_file", "") for n in chunk_nodes}
            cached_source_files |= {e.get("source_file", "") for e in chunk_edges}
            for f in chunk_files:
                rel = str(f.relative_to(repo_path))
                if rel not in cached_source_files:
                    save_cached(f, {"nodes": [], "edges": []}, root=cache_root)

        failure_rate = failed_chunks / total_chunks if total_chunks > 0 else 0
        log.info(
            "Extraction complete: %.1fs, %d/%d chunks succeeded (%.0f%% failure rate)",
            elapsed,
            total_chunks - failed_chunks,
            total_chunks,
            failure_rate * 100,
        )

        if failure_rate > FAILURE_THRESHOLD:
            log.error(
                "ABORT: %.0f%% failure rate exceeds %.0f%% threshold",
                failure_rate * 100,
                FAILURE_THRESHOLD * 100,
            )
            sys.exit(1)
    else:
        log.info("Step 2/7: All files cached — skipping extraction")

    # ------------------------------------------------------------------
    # Step 4: Merge and deduplicate
    # ------------------------------------------------------------------
    log.info("Step 3/7: Merging extractions")
    all_nodes = cached_nodes + new_nodes
    all_edges = cached_edges + new_edges

    seen_ids: set[str] = set()
    deduped_nodes: list[dict] = []
    for node in all_nodes:
        nid = node.get("id", "")
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            deduped_nodes.append(node)

    merged = {"nodes": deduped_nodes, "edges": all_edges}
    log.info("Merged: %d unique nodes, %d edges", len(deduped_nodes), len(all_edges))

    # ------------------------------------------------------------------
    # Step 5: Build graph, cluster, analyze
    # ------------------------------------------------------------------
    log.info("Step 4/7: Building graph")
    G = build_from_json(merged)
    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    log.info("Step 5/7: Clustering")
    communities = cluster(G)
    cohesion = score_all(G, communities)
    log.info("Communities: %d", len(communities))

    # ------------------------------------------------------------------
    # Step 6: Auto-label communities
    # ------------------------------------------------------------------
    log.info("Step 6/7: Auto-labeling communities")
    label_tasks = []
    for cid, members in communities.items():
        # Get top-5 most connected nodes in this community
        sub = G.subgraph(members)
        top_nodes = sorted(sub.nodes(), key=lambda n: sub.degree(n), reverse=True)[:5]
        top_labels = [G.nodes[n].get("label", n) for n in top_nodes]
        label_tasks.append(label_community(client, cid, top_labels))

    label_results = await asyncio.gather(*label_tasks)
    community_labels: dict[int, str] = dict(label_results)
    log.info("Labeled %d communities", len(community_labels))

    # ------------------------------------------------------------------
    # Step 7: Analyze and generate report
    # ------------------------------------------------------------------
    log.info("Step 7/7: Analyzing and generating report")
    gods = god_nodes(G, top_n=20)
    surprises = surprising_connections(G, communities, top_n=15)

    try:
        questions = suggest_questions(G, communities, community_labels)
    except Exception:
        questions = None

    token_cost = {
        "input": total_input_tokens,
        "output": total_output_tokens,
    }

    try:
        report = generate_report(
            G,
            communities,
            cohesion,
            community_labels,
            gods,
            surprises,
            detection,
            token_cost,
            str(repo_path),
            suggested_questions=questions,
        )
    except Exception as exc:
        log.warning("Graphify report generator failed (%s), building fallback", exc)
        report = _fallback_report(G, communities, community_labels, gods, cohesion)

    # ------------------------------------------------------------------
    # Export artifacts
    # ------------------------------------------------------------------
    report_path = output_dir / "GRAPH_REPORT.md"
    report_path.write_text(report)
    log.info("Report: %s", report_path)

    graph_json_path = str(output_dir / "graph.json")
    try:
        to_json(G, communities, graph_json_path)
    except Exception:
        import networkx as nx
        data = nx.node_link_data(G)
        (output_dir / "graph.json").write_text(json.dumps(data, indent=2))
    log.info("Graph: %s", graph_json_path)

    # Graph diff (compare with previous if exists)
    diff_summary = None
    prev_graph_path = output_dir / "graph_previous.json"
    if prev_graph_path.exists():
        try:
            import networkx as nx
            prev_data = json.loads(prev_graph_path.read_text())
            G_old = nx.node_link_graph(prev_data)
            diff_summary = graph_diff(G_old, G)
            log.info(
                "Diff: +%d nodes, -%d nodes, +%d edges, -%d edges",
                len(diff_summary.get("added_nodes", [])),
                len(diff_summary.get("removed_nodes", [])),
                len(diff_summary.get("added_edges", [])),
                len(diff_summary.get("removed_edges", [])),
            )
        except Exception as exc:
            log.warning("Graph diff failed: %s", exc)

    # Stats file
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_files_detected": detection.get("total_files", 0),
        "total_words": detection.get("total_words", 0),
        "files_processed": len(doc_files),
        "files_cached": len(doc_files) - len(uncached_files),
        "files_extracted": len(uncached_files),
        "chunks_failed": failed_chunks,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "communities": len(communities),
        "god_nodes_count": len(gods),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "model": MODEL,
        "label_model": LABEL_MODEL,
    }
    if diff_summary:
        stats["diff"] = {
            "added_nodes": len(diff_summary.get("added_nodes", [])),
            "removed_nodes": len(diff_summary.get("removed_nodes", [])),
            "added_edges": len(diff_summary.get("added_edges", [])),
            "removed_edges": len(diff_summary.get("removed_edges", [])),
        }

    (output_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Stats: %s", output_dir / "graph_stats.json")

    # Rotate current graph.json to graph_previous.json for next diff
    import shutil
    shutil.copy2(output_dir / "graph.json", prev_graph_path)

    log.info("=" * 60)
    log.info("COMPLETE: %d nodes, %d edges, %d communities", G.number_of_nodes(), G.number_of_edges(), len(communities))
    log.info("=" * 60)


def _fallback_report(G, communities, labels, gods, cohesion) -> str:
    """Generate a minimal report if the Graphify report generator fails."""
    lines = [
        "# ArmadaOS Institutional Knowledge Graph Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Graph Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Nodes | {G.number_of_nodes()} |",
        f"| Edges | {G.number_of_edges()} |",
        f"| Communities | {len(communities)} |",
        "",
        "## God Nodes (Most Connected Concepts)",
        "",
        "| Rank | Concept | Connections |",
        "|------|---------|-------------|",
    ]
    for i, g in enumerate(gods, 1):
        lines.append(f"| {i} | {g['label']} | {g['edges']} |")

    lines.extend(["", "## Communities", ""])
    lines.append("| ID | Label | Size | Cohesion |")
    lines.append("|----|-------|------|----------|")
    for cid in sorted(communities.keys()):
        lbl = labels.get(cid, f"Community {cid}")
        size = len(communities[cid])
        coh = cohesion.get(cid, 0.0)
        lines.append(f"| {cid} | {lbl} | {size} | {coh:.2f} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <repo_path> <output_dir>", file=sys.stderr)
        sys.exit(1)

    repo_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    if not repo_path.is_dir():
        print(f"Error: {repo_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(repo_path, output_dir))


if __name__ == "__main__":
    main()
