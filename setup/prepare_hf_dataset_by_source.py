#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Prepare dolma3/dolmino local data into per-source shuffled chunks.

This script is intended for local preprocessing when the downloaded dataset has
many ingredient folders under:

    <local_dir>/data/<ingredient-folder>/*.jsonl.zst

It groups folders by a source prefix map (prefix -> ratio), then prepares one
shuffled dataset per source so training can mix sources with explicit weights
via `data.sources` in config YAML.

Output layout (compatible with `lingua/data.py`):

    <out_dir>/
      <source_a>/
        <source_a>.chunk.00.jsonl
        ...
        <source_a>.val.jsonl
      <source_b>/
        ...
      source_ratios.yaml

Notes:
* Reading inside each source is still deterministic in `lingua/data.py`; this
  script improves source-level mixing by exposing source-specific directories.
* To avoid a source disappearing on a node, sources with too few files fall
  back to "process all files on every node" mode for that source.
"""

import argparse
import glob as globmod
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml


DEFAULT_PREFIX_TO_RATIO: Dict[str, float] = {
    "common_crawl-high-quality": 22.5,
    "olmocr_science_pdfs-high_quality": 5.0,
    "stack_edu_fim": 10.0,
    "stem-heavy-crawl": 5.0,
    "cranecode": 10.0,
    "cranemath": 5.63,
    "megamatt": 1.73,
    "dolmino-math": 10.7,
    "omr-rewrite-fullthoughts": 0.85,
    "tinymath-mind": 0.9,
    "tinymath-pot": 0.24,
    "reddit_to_flashcards": 5.9,
    "wiki_to_rcqa": 3.0,
    "nemotron-synth-qa": 5.0,
    "tulu-3-sft": 1.1,
    "dolmino_1-flan": 5.0,
    "qwq-reasoning-traces": 1.87,
    "gemini-reasoning-traces": 0.25,
    "llama_nemotron-reasoning-traces": 1.25,
    "openthoughts2-reasoning-traces": 1.25,
    "program_verifiable": 0.16,
    "math-meta-reasoning": 0.38,
    "code-meta-reasoning": 0.46,
    "general_reasoning_mix": 1.87,
}


@dataclass
class SourceAssignment:
    source: str
    ratio: float
    folders: List[str]
    files: List[str]
    files_for_this_node: List[str]
    fallback_all_files: bool


@dataclass
class SourceGroup:
    source: str
    ratio: float
    prefixes: List[str]


def run_command(command: str):
    print(f"[CMD] {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def get_node_rank_from_hostname() -> int:
    hostname = os.environ.get("HOSTNAME", "")
    try:
        return int(hostname.rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        print(
            f"WARNING: Could not infer node rank from HOSTNAME='{hostname}', defaulting to 0.",
            file=sys.stderr,
        )
        return 0


def stream_lines_from_local_zst(path: str):
    cmd = f"zstdcat '{path}'"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            stripped = line.rstrip("\n\r")
            if stripped:
                yield stripped
    finally:
        proc.stdout.close()
        proc.wait()
        if proc.returncode != 0:
            print(f"WARNING: decompression of {path} exited with code {proc.returncode}", file=sys.stderr)


def prepare_data(
    file_labels: List[str],
    out_dir: str,
    dataset: str,
    nchunks: int,
    seed: int,
    k_validation: int,
):
    os.makedirs(out_dir, exist_ok=True)

    chunk_paths = []
    chunk_fps = []
    for i in range(nchunks):
        p = os.path.join(out_dir, f"{dataset}.chunk.{i:02d}.jsonl")
        chunk_paths.append(p)
        chunk_fps.append(open(p, "w"))

    rng = random.Random(seed)
    total_lines = 0
    for idx, path in enumerate(file_labels):
        print(f"[{idx + 1}/{len(file_labels)}] Processing {path}", flush=True)
        for line in stream_lines_from_local_zst(path):
            chunk_fps[rng.randint(0, nchunks - 1)].write(line + "\n")
            total_lines += 1
            if total_lines % 500_000 == 0:
                print(f"  ... {total_lines:,} lines so far", flush=True)

    for fp in chunk_fps:
        fp.close()

    val_path = os.path.join(out_dir, f"{dataset}.val.jsonl")
    print(f"Carving validation set ({k_validation} lines/chunk) -> {val_path}", flush=True)
    with open(val_path, "w") as vf:
        for cp in chunk_paths:
            run_command(f"head -n {k_validation} '{cp}' >> '{val_path}'")
    for cp in chunk_paths:
        run_command(f"sed -i '1,{k_validation}d' '{cp}'")

    print(
        f"Prepared source '{dataset}': {total_lines:,} lines, {nchunks} chunks, val={val_path}",
        flush=True,
    )


def load_prefix_to_ratio(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("--ratio_json must contain a JSON object: {prefix: ratio, ...}")
    out: Dict[str, float] = {}
    for k, v in payload.items():
        out[str(k)] = float(v)
    return out


def default_source_groups() -> List[SourceGroup]:
    return [
        SourceGroup(source=source, ratio=ratio, prefixes=[source])
        for source, ratio in DEFAULT_PREFIX_TO_RATIO.items()
    ]


def load_source_groups_yaml(path: str) -> List[SourceGroup]:
    with open(path, "r") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict) or "groups" not in payload:
        raise ValueError(
            "--group_yaml must be a YAML object containing a top-level 'groups' list"
        )

    groups_payload = payload["groups"]
    if not isinstance(groups_payload, list):
        raise ValueError("'groups' must be a list")

    out: List[SourceGroup] = []
    seen_sources = set()
    seen_prefixes = set()
    for i, g in enumerate(groups_payload):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{i}] must be an object")
        source = str(g.get("source", "")).strip()
        ratio = g.get("ratio", None)
        prefixes = g.get("prefixes", None)
        if not source:
            raise ValueError(f"groups[{i}] missing non-empty 'source'")
        if source in seen_sources:
            raise ValueError(f"Duplicate source in group_yaml: '{source}'")
        if ratio is None:
            raise ValueError(f"groups[{i}] missing 'ratio'")
        if not isinstance(prefixes, list) or len(prefixes) == 0:
            raise ValueError(f"groups[{i}] must contain non-empty 'prefixes' list")
        cleaned_prefixes: List[str] = []
        for p in prefixes:
            prefix = str(p).strip()
            if not prefix:
                continue
            if prefix in seen_prefixes:
                raise ValueError(
                    f"Prefix '{prefix}' is mapped by more than one group in {path}"
                )
            seen_prefixes.add(prefix)
            cleaned_prefixes.append(prefix)
        if len(cleaned_prefixes) == 0:
            raise ValueError(f"groups[{i}] has no valid prefixes")
        out.append(
            SourceGroup(
                source=source,
                ratio=float(ratio),
                prefixes=cleaned_prefixes,
            )
        )
        seen_sources.add(source)

    return out


def canonicalize_folder_name(folder_name: str) -> str:
    # Typical naming: ingredient1-<source-prefix>_<suffix>
    name = re.sub(r"^ingredient\d+-", "", folder_name)
    return name


def find_matching_prefix(folder_name: str, prefixes: List[str]) -> str:
    canonical = canonicalize_folder_name(folder_name)
    matches = [p for p in prefixes if canonical.startswith(p)]
    if not matches:
        matches = [p for p in prefixes if p in canonical]
    if not matches:
        return ""
    # Prefer longest prefix when multiple match.
    matches.sort(key=len, reverse=True)
    return matches[0]


def find_matching_group(folder_name: str, groups: List[SourceGroup]) -> str:
    canonical = canonicalize_folder_name(folder_name)

    prefix_to_source = {}
    for g in groups:
        for prefix in g.prefixes:
            prefix_to_source[prefix] = g.source

    matches = [p for p in prefix_to_source if canonical.startswith(p)]
    if not matches:
        matches = [p for p in prefix_to_source if p in canonical]
    if not matches:
        return ""
    matches.sort(key=len, reverse=True)
    return prefix_to_source[matches[0]]


def list_immediate_subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )


def list_jsonl_zst_files(folder_path: str) -> List[str]:
    return sorted(globmod.glob(os.path.join(folder_path, "**/*.jsonl.zst"), recursive=True))


def assign_sources(
    data_dir: str,
    source_groups: List[SourceGroup],
    node_rank: int,
    num_nodes: int,
) -> Tuple[List[SourceAssignment], List[str]]:
    source_to_folders: Dict[str, List[str]] = {
        g.source: [] for g in source_groups
    }
    unmatched_folders: List[str] = []

    all_folders = list_immediate_subdirs(data_dir)
    for folder in all_folders:
        source = find_matching_group(folder, source_groups)
        if source:
            source_to_folders[source].append(os.path.join(data_dir, folder))
        else:
            unmatched_folders.append(folder)

    assignments: List[SourceAssignment] = []
    for group in source_groups:
        source, ratio = group.source, group.ratio
        folders = sorted(source_to_folders.get(source, []))
        files: List[str] = []
        for folder_path in folders:
            files.extend(list_jsonl_zst_files(folder_path))
        files = sorted(files)
        if not files:
            continue

        files_for_this_node = [f for i, f in enumerate(files) if i % num_nodes == node_rank]
        fallback_all_files = False
        if len(files_for_this_node) == 0:
            # For tiny sources, ensure each node still materializes this source.
            files_for_this_node = files
            fallback_all_files = True

        assignments.append(
            SourceAssignment(
                source=source,
                ratio=float(ratio),
                folders=folders,
                files=files,
                files_for_this_node=files_for_this_node,
                fallback_all_files=fallback_all_files,
            )
        )

    return assignments, unmatched_folders


def write_sources_yaml(out_dir: str, root_dir_for_yaml: str, assignments: List[SourceAssignment]):
    yaml_path = os.path.join(out_dir, "source_ratios.yaml")
    with open(yaml_path, "w") as f:
        f.write("data:\n")
        f.write(f"  root_dir: {root_dir_for_yaml}\n")
        f.write("  sources:\n")
        for a in assignments:
            f.write(f"    {a.source}: {a.ratio}\n")
    print(f"Wrote YAML source map: {yaml_path}", flush=True)


def write_summary_json(out_dir: str, assignments: List[SourceAssignment], unmatched_folders: List[str]):
    payload = {
        "num_sources": len(assignments),
        "sources": [
            {
                "source": a.source,
                "ratio": a.ratio,
                "num_folders": len(a.folders),
                "num_files_total": len(a.files),
                "num_files_this_node": len(a.files_for_this_node),
                "fallback_all_files": a.fallback_all_files,
                "folders": a.folders,
            }
            for a in assignments
        ],
        "unmatched_folders": unmatched_folders,
    }
    summary_path = os.path.join(out_dir, "source_prepare_summary.json")
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote preparation summary: {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare local dolma/dolmino data into source-specific shuffled datasets."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Path to downloaded dataset root (contains data/) OR directly to data/ directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output root directory; one subdirectory per source is created.",
    )
    parser.add_argument(
        "--ratio_json",
        type=str,
        default=None,
        help="Optional JSON file with {prefix: ratio}. Defaults to built-in map.",
    )
    parser.add_argument(
        "--group_yaml",
        type=str,
        default=None,
        help=(
            "Optional YAML file describing grouped sources with shape: "
            "{groups: [{source, ratio, prefixes: [...]}, ...]}."
        ),
    )
    parser.add_argument("--num_nodes", type=int, default=None, help="Total nodes.")
    parser.add_argument("--node_rank", type=int, default=None, help="Current node rank.")
    parser.add_argument("--nchunks", type=int, default=8, help="Chunks per source (per node).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--k_validation", type=int, default=10000, help="Validation lines per chunk.")
    parser.add_argument(
        "--yaml_root_dir",
        type=str,
        default=None,
        help="Root dir written in source_ratios.yaml. Defaults to --out_dir.",
    )
    args = parser.parse_args()

    if args.node_rank is not None:
        node_rank = args.node_rank
    else:
        node_rank = get_node_rank_from_hostname()

    if args.num_nodes is not None:
        num_nodes = args.num_nodes
    else:
        ws = os.environ.get("WORLD_SIZE")
        num_nodes = max(1, int(ws) // 8) if ws is not None else 1

    if args.group_yaml:
        source_groups = load_source_groups_yaml(args.group_yaml)
    else:
        prefix_to_ratio = (
            load_prefix_to_ratio(args.ratio_json)
            if args.ratio_json
            else DEFAULT_PREFIX_TO_RATIO
        )
        source_groups = [
            SourceGroup(source=k, ratio=v, prefixes=[k])
            for k, v in prefix_to_ratio.items()
        ]

    local_dir = args.local_dir.rstrip("/")
    if os.path.basename(local_dir) == "data":
        data_dir = local_dir
    else:
        data_dir = os.path.join(local_dir, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Could not find data directory: {data_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Node rank: {node_rank} / {num_nodes}", flush=True)
    print(f"Data dir : {data_dir}", flush=True)
    print(f"Out dir  : {args.out_dir}", flush=True)

    assignments, unmatched_folders = assign_sources(
        data_dir=data_dir,
        source_groups=source_groups,
        node_rank=node_rank,
        num_nodes=num_nodes,
    )
    if not assignments:
        raise RuntimeError(
            "No source folders matched prefix map. "
            "Check folder names or pass --ratio_json with updated prefixes."
        )

    print(f"Matched {len(assignments)} sources.", flush=True)
    if unmatched_folders:
        print(f"WARNING: {len(unmatched_folders)} folders were unmatched.", flush=True)
        for folder in unmatched_folders[:20]:
            print(f"  - {folder}", flush=True)
        if len(unmatched_folders) > 20:
            print("  ... (truncated)", flush=True)

    # Prepare each source independently.
    for idx, a in enumerate(assignments):
        source_out = os.path.join(args.out_dir, a.source)
        source_seed = args.seed + node_rank * 10_000 + idx
        print(
            f"[{idx + 1}/{len(assignments)}] Source '{a.source}': "
            f"ratio={a.ratio}, folders={len(a.folders)}, files(node)={len(a.files_for_this_node)}"
            + (" [fallback_all_files]" if a.fallback_all_files else ""),
            flush=True,
        )
        prepare_data(
            file_labels=a.files_for_this_node,
            out_dir=source_out,
            dataset=a.source,
            nchunks=args.nchunks,
            seed=source_seed,
            k_validation=args.k_validation,
        )

    yaml_root_dir = args.yaml_root_dir if args.yaml_root_dir else args.out_dir
    write_sources_yaml(args.out_dir, yaml_root_dir, assignments)
    write_summary_json(args.out_dir, assignments, unmatched_folders)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
