#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Space-efficient, multi-node data preparation from S3 (or local copy).

Problem
-------
The dolmino-mix dataset is ~340 GB compressed.  Downloading the entire blob and
then creating a globally-shuffled copy requires >500 GB, exceeding node-local
``/dev/shm`` capacity.

Solution
--------
* Each node processes only its ``1 / num_nodes`` share of the source files.
* Data is either **streamed from S3** (zero local compressed storage) or
  processed from an **already-synced local directory** (compressed files are
  deleted one-by-one after decompression to reclaim space).
* Lines are distributed *randomly* across output chunk files, providing a
  good-enough shuffle without needing 2× storage.
* A small validation split is carved out of each chunk.

Output structure (matches the ``lingua/data.py`` expectations)::

    {out_dir}/
        {dataset}.chunk.00.jsonl
        {dataset}.chunk.01.jsonl
        …
        {dataset}.val.jsonl

Usage examples
--------------
Stream from S3 (most space-efficient — nothing stored until the final chunks)::

    python setup/aws_prepare_hf_dataset.py \\
        --s3_uri s3://agi-mm-training-shared-us-east-2/beidchen/data/stem/dolma3_dolmino_mix-100B-1125/ \\
        --region us-east-2 \\
        --out_dir /dev/shm/dolmino-mix_shuffled \\
        --dataset dolmino-mix \\
        --num_nodes 4 \\
        --nchunks 8

Process already-downloaded local ``.jsonl.zst`` files (one-at-a-time decompression)::

    python setup/aws_prepare_hf_dataset.py \\
        --local_dir /dev/shm/dolma3_dolmino_mix-100B-1125 \\
        --out_dir /dev/shm/dolmino-mix_shuffled \\
        --dataset dolmino-mix \\
        --num_nodes 4 \\
        --nchunks 8

Notes
-----
* ``nchunks`` should divide the *global* world-size evenly.  The default of 8
  (= GPUs per node on p5en.48xlarge) means every local rank gets its own chunk.
  Other local ranks on *remote* nodes will read *their* node-local chunks,
  so there is no wasted storage.
* Node rank is inferred from ``$HOSTNAME`` (PyTorchJob pod names end with
  ``-worker-<N>``).  Override with ``--node_rank`` if needed.
"""

import argparse
import os
import random
import subprocess
import sys
import glob as globmod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(command):
    print(f"[CMD] {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def get_node_rank_from_hostname():
    """Infer rank from the Kubernetes pod hostname (e.g. ``job-worker-3``)."""
    hostname = os.environ.get("HOSTNAME", "")
    try:
        return int(hostname.rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        print(
            f"WARNING: Could not infer node rank from HOSTNAME='{hostname}', "
            "defaulting to 0.  Pass --node_rank explicitly if needed.",
            file=sys.stderr,
        )
        return 0


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def list_s3_files(s3_uri, region, pattern=".jsonl.zst"):
    """Return sorted list of S3 keys matching *pattern* under *s3_uri*."""
    cmd = f"aws s3 ls '{s3_uri}' --recursive --region {region}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    keys = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(None, 3)  # date, time, size, key
        if len(parts) >= 4 and pattern in parts[3]:
            keys.append(parts[3])
    keys.sort()
    return keys


def stream_lines_from_s3(s3_bucket, s3_key, region):
    """Stream-download *s3_key*, decompress on-the-fly, and yield lines."""
    cmd = f"aws s3 cp 's3://{s3_bucket}/{s3_key}' - --region {region} | zstd -d"
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
            print(f"WARNING: stream of {s3_key} exited with code {proc.returncode}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def list_local_files(local_dir, pattern="**/*.jsonl.zst"):
    """Return sorted list of .jsonl.zst files under *local_dir*."""
    files = sorted(globmod.glob(os.path.join(local_dir, pattern), recursive=True))
    return files


def stream_lines_from_local_zst(path):
    """Decompress a local ``.jsonl.zst`` file and yield lines."""
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


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def prepare_data(
    line_iterator_fn,
    file_labels,
    out_dir,
    dataset,
    nchunks,
    seed,
    k_validation,
    delete_after=None,
):
    """
    Read lines from *line_iterator_fn(label)* for each label in *file_labels*,
    distribute them randomly across *nchunks* chunk files, then carve out a
    validation split.

    Parameters
    ----------
    line_iterator_fn : callable(label) -> Iterator[str]
        Given a file label (S3 key or local path), yields decompressed lines.
    file_labels : list[str]
        File identifiers assigned to this node.
    out_dir : str
        Directory for output chunk + validation files.
    dataset : str
        Dataset name prefix (e.g. ``dolmino-mix``).
    nchunks : int
        Number of output chunk files.
    seed : int
        Random seed for reproducible line assignment.
    k_validation : int
        Number of lines to reserve per chunk for the validation set.
    delete_after : list[str] or None
        If provided, local paths to delete after processing each file
        (used for local mode to free space incrementally).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Open output chunk files
    chunk_paths = []
    chunk_fps = []
    for i in range(nchunks):
        p = os.path.join(out_dir, f"{dataset}.chunk.{i:02d}.jsonl")
        chunk_paths.append(p)
        chunk_fps.append(open(p, "w"))

    rng = random.Random(seed)
    total_lines = 0

    for idx, label in enumerate(file_labels):
        print(f"[{idx + 1}/{len(file_labels)}] Processing {label} ...", flush=True)
        for line in line_iterator_fn(label):
            chunk_fps[rng.randint(0, nchunks - 1)].write(line + "\n")
            total_lines += 1
            if total_lines % 500_000 == 0:
                print(f"  … {total_lines:,} lines so far", flush=True)

        # Optionally delete the source file to reclaim space
        if delete_after is not None:
            src = delete_after[idx]
            if os.path.isfile(src):
                os.remove(src)
                print(f"  Deleted {src} to free space.", flush=True)

    for fp in chunk_fps:
        fp.close()

    print(f"Wrote {total_lines:,} lines across {nchunks} chunks.", flush=True)

    # ---- validation split ----
    val_path = os.path.join(out_dir, f"{dataset}.val.jsonl")
    print(f"Carving validation set ({k_validation} lines/chunk) → {val_path}", flush=True)
    with open(val_path, "w") as vf:
        for cp in chunk_paths:
            run_command(f"head -n {k_validation} '{cp}' >> '{val_path}'")
    for cp in chunk_paths:
        run_command(f"sed -i '1,{k_validation}d' '{cp}'")

    print("✓ Data preparation complete!", flush=True)
    print(f"  Output dir : {out_dir}", flush=True)
    print(f"  Chunks     : {nchunks}", flush=True)
    print(f"  Validation : {val_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Space-efficient multi-node data preparation from S3 or local copy."
    )

    # Data source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--s3_uri",
        type=str,
        default=None,
        help="S3 URI containing .jsonl.zst files, e.g. s3://bucket/prefix/",
    )
    src.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local directory (already synced from S3) with .jsonl.zst files.",
    )

    parser.add_argument("--region", type=str, default="us-east-2", help="AWS region")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for shuffled chunk files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dolmino-mix",
        help="Dataset name prefix for output files.",
    )
    parser.add_argument(
        "--filter_pattern",
        type=str,
        default=None,
        help="Only include source files whose path contains this substring "
        "(e.g. 'ingredient1-common_crawl' for dolmino-mix-subset).",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=None,
        help="Total number of nodes. Inferred from $NUM_NODES / $WORLD_SIZE if unset.",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=None,
        help="This node's rank (0-indexed). Inferred from $HOSTNAME if unset.",
    )
    parser.add_argument(
        "--nchunks",
        type=int,
        default=8,
        help="Number of output chunks per node (default 8 = GPUs per node).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--k_validation",
        type=int,
        default=10000,
        help="Lines per chunk reserved for validation.",
    )
    parser.add_argument(
        "--delete_local_after",
        action="store_true",
        help="(local_dir mode) Delete each .jsonl.zst file after processing to free space.",
    )

    args = parser.parse_args()

    # ---- resolve node rank / num_nodes ----
    if args.node_rank is not None:
        node_rank = args.node_rank
    else:
        node_rank = get_node_rank_from_hostname()

    if args.num_nodes is not None:
        num_nodes = args.num_nodes
    else:
        # Try common env vars set by PyTorchJob / torchrun
        for var in ("NUM_NODES", "NNODES"):
            val = os.environ.get(var)
            if val is not None:
                num_nodes = int(val)
                break
        else:
            # Fallback: WORLD_SIZE / gpus_per_node
            ws = os.environ.get("WORLD_SIZE")
            if ws is not None:
                num_nodes = max(1, int(ws) // 8)
            else:
                num_nodes = 1

    print(f"Node rank : {node_rank} / {num_nodes}", flush=True)

    # ---- discover source files ----
    if args.s3_uri is not None:
        s3_uri = args.s3_uri.rstrip("/") + "/"
        parts = s3_uri.replace("s3://", "").split("/", 1)
        s3_bucket = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ""

        all_keys = list_s3_files(s3_uri, args.region)
        if args.filter_pattern:
            all_keys = [k for k in all_keys if args.filter_pattern in k]
        print(f"Found {len(all_keys)} .jsonl.zst files in S3.", flush=True)

        # Assign subset to this node (round-robin)
        my_keys = [k for i, k in enumerate(all_keys) if i % num_nodes == node_rank]
        print(f"This node will stream {len(my_keys)} files.", flush=True)

        line_iter_fn = lambda key: stream_lines_from_s3(s3_bucket, key, args.region)
        prepare_data(
            line_iterator_fn=line_iter_fn,
            file_labels=my_keys,
            out_dir=args.out_dir,
            dataset=args.dataset,
            nchunks=args.nchunks,
            seed=args.seed + node_rank,  # different seed per node for diversity
            k_validation=args.k_validation,
        )

    else:
        # ---- local mode ----
        all_files = list_local_files(args.local_dir)
        if args.filter_pattern:
            all_files = [f for f in all_files if args.filter_pattern in f]
        print(f"Found {len(all_files)} .jsonl.zst files locally.", flush=True)

        my_files = [f for i, f in enumerate(all_files) if i % num_nodes == node_rank]
        print(f"This node will process {len(my_files)} files.", flush=True)

        line_iter_fn = lambda path: stream_lines_from_local_zst(path)
        delete_list = my_files if args.delete_local_after else None
        prepare_data(
            line_iterator_fn=line_iter_fn,
            file_labels=my_files,
            out_dir=args.out_dir,
            dataset=args.dataset,
            nchunks=args.nchunks,
            seed=args.seed + node_rank,
            k_validation=args.k_validation,
            delete_after=delete_list,
        )


if __name__ == "__main__":
    main()

