import argparse
import sys
import time
import statistics
import torch

from setup import stem_gather_cpu


def python_gather_reference(
    token_ids_cpu: torch.Tensor,
    cpu_tbl: torch.Tensor,       # [vocab, L, D] bf16
    stage: torch.Tensor,         # [L, Umax, D] bf16 pinned
    inv_stage: torch.Tensor,     # [Tmax] int32 pinned
):
    """
    Reference implementation matching your old Python path:
      unique + inverse
      tmp = index_select
      stage[:, :U, :] = tmp.permute(1,0,2)
      inv_stage[:T] = inv
    Returns (U, T).
    """
    token_ids_1d = token_ids_cpu.reshape(-1).contiguous()
    unique_ids, inv = torch.unique(token_ids_1d, sorted=False, return_inverse=True)
    U = int(unique_ids.numel())
    T = int(inv.numel())

    tmp = cpu_tbl.index_select(0, unique_ids)                # [U, L, D]
    stage[:, :U, :].copy_(tmp.permute(1, 0, 2))              # [L, U, D]
    inv_stage[:T].copy_(inv.to(torch.int32))                 # [T]
    return U, T


def reconstruct_from_stage(
    token_ids_cpu_1d: torch.Tensor,  # [T] int32/int64
    cpu_tbl: torch.Tensor,           # [vocab, L, D] bf16
    stage: torch.Tensor,             # [L, U, D] bf16
    inv: torch.Tensor,               # [T] int32
    U: int,
):
    """
    Semantic check: for each token position t:
      stage[:, inv[t], :] must equal cpu_tbl[token_id[t], :, :]
    """
    T = token_ids_cpu_1d.numel()
    assert inv.numel() == T
    assert inv.dtype == torch.int32
    assert int(inv.max().item()) < U
    assert int(inv.min().item()) >= 0

    # Compare by gathering stage for each token position (L,D) and comparing to cpu_tbl row.
    # We do it in a vectorized-ish way:
    # stage is [L, U, D]. We want per t -> stage[:, inv[t], :] => [L, T, D]
    # then permute to [T, L, D] and compare to cpu_tbl[token_ids] => [T, L, D].
    gathered = stage[:, inv.to(torch.long), :]               # [L, T, D]
    gathered = gathered.permute(1, 0, 2).contiguous()         # [T, L, D]
    target = cpu_tbl.index_select(0, token_ids_cpu_1d.to(torch.long))  # [T, L, D]

    # bf16 exact equality is expected if everything is pure copies.
    ok = torch.equal(gathered, target)
    if not ok:
        # find first mismatch
        diff = (gathered != target)
        idx = diff.nonzero(as_tuple=False)[0].tolist()
        t, l, d = idx
        return False, (t, l, d, gathered[t, l, d].item(), target[t, l, d].item())
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--T", type=int, default=4096, help="tokens per test case (flattened)")
    ap.add_argument("--Umax", type=int, default=4096, help="stage unique capacity")
    ap.add_argument("--Tmax", type=int, default=None, help="inv capacity (defaults to T)")
    ap.add_argument("--cases", type=str, default="mixed",
                    help="comma-separated: mixed,all_unique,all_same,half_unique,max_unique")
    ap.add_argument("--benchmark", action="store_true", help="Run performance benchmark (skip correctness checks)")
    ap.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations for benchmarking")
    ap.add_argument("--skip_correctness", action="store_true", help="Skip correctness checks (faster)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    vocab = args.vocab
    L = args.L
    D = args.D
    T = args.T
    Umax = args.Umax
    Tmax = args.T if args.Tmax is None else args.Tmax

    assert Tmax >= T, "Tmax must be >= T"

    # CPU table: [vocab, L, D]
    cpu_tbl = torch.empty((vocab, L, D), device="cpu", dtype=torch.bfloat16, pin_memory=True)
    # Fill with deterministic-ish random values
    cpu_tbl.copy_(torch.randn((vocab, L, D), dtype=torch.float32).to(torch.bfloat16))

    # Buffers for python ref
    stage_py = torch.empty((L, Umax, D), device="cpu", dtype=torch.bfloat16, pin_memory=True)
    inv_py = torch.empty((Tmax,), device="cpu", dtype=torch.int32, pin_memory=True)

    # Buffers for C++ op
    stage_cpp = torch.empty((L, Umax, D), device="cpu", dtype=torch.bfloat16, pin_memory=True)
    inv_cpp = torch.empty((Tmax,), device="cpu", dtype=torch.int32, pin_memory=True)

    # C++ op required scratch
    seen = torch.zeros((vocab,), device="cpu", dtype=torch.int32)
    slot = torch.empty_like(seen)
    unique_out = torch.empty((Umax,), device="cpu", dtype=torch.int32)

    # Epoch tagging
    epoch = 1

    # Case generators
    enabled = [c.strip() for c in args.cases.split(",") if c.strip()]

    def gen_tokens(case_name: str) -> torch.Tensor:
        if case_name == "all_unique":
            # Ensure unique as much as possible (cap by vocab and T)
            n = min(T, vocab)
            base = torch.randperm(vocab)[:n]
            if n < T:
                # wrap if T > vocab
                extra = torch.randint(0, vocab, (T - n,))
                ids = torch.cat([base, extra], dim=0)
            else:
                ids = base
            return ids.to(torch.int64)

        if case_name == "all_same":
            val = torch.randint(0, vocab, (1,), dtype=torch.int64)
            return val.repeat(T)

        if case_name == "half_unique":
            # T/2 unique, rest repeats
            n = min(T // 2, vocab)
            uniq = torch.randperm(vocab)[:n].to(torch.int64)
            reps = uniq[torch.randint(0, n, (T - n,))]
            ids = torch.cat([uniq, reps], dim=0)
            return ids[torch.randperm(T)]

        if case_name == "max_unique":
            # Try to hit U close to Umax (but <= vocab and <= T)
            n = min(Umax, T, vocab)
            uniq = torch.randperm(vocab)[:n].to(torch.int64)
            if n < T:
                reps = uniq[torch.randint(0, n, (T - n,))]
                ids = torch.cat([uniq, reps], dim=0)
            else:
                ids = uniq
            return ids[torch.randperm(T)]

        # default: mixed random with repeats
        return torch.randint(0, vocab, (T,), dtype=torch.int64)

    # Timing statistics
    py_times = []
    cpp_times = []
    
    print(f"Running {args.trials} trials. cases={enabled}")
    if args.benchmark:
        print(f"Benchmark mode: {args.warmup} warmup iterations, then {args.trials} timed trials")
    start = time.time()

    # Warmup iterations for benchmarking
    if args.benchmark and args.warmup > 0:
        print(f"Warming up ({args.warmup} iterations)...")
        for _ in range(args.warmup):
            case = enabled[0]
            token_ids = gen_tokens(case)
            stage_py.zero_()
            inv_py.zero_()
            stage_cpp.zero_()
            inv_cpp.zero_()
            python_gather_reference(token_ids, cpu_tbl, stage_py, inv_py)
            stem_gather_cpu(
                token_ids, cpu_tbl, stage_cpp, inv_cpp,
                unique_out, seen, slot, int(epoch)
            )
            epoch += 1
            if epoch >= 2**31 - 2:
                seen.zero_()
                epoch = 1
        print("Warmup complete.")

    for trial in range(args.trials):
        case = enabled[trial % len(enabled)]
        token_ids = gen_tokens(case)

        # Clear buffers to catch stale writes (optional)
        stage_py.zero_()
        inv_py.zero_()
        stage_cpp.zero_()
        inv_cpp.zero_()

        # Python reference with timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        py_start = time.perf_counter()
        U_py, T_py = python_gather_reference(token_ids, cpu_tbl, stage_py, inv_py)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        py_end = time.perf_counter()
        py_time = (py_end - py_start) * 1000  # Convert to milliseconds
        py_times.append(py_time)

        # C++ fused with timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cpp_start = time.perf_counter()
        # NOTE: stem_gather_cpu returns (U, T)
        U_cpp, T_cpp = stem_gather_cpu(
            token_ids,
            cpu_tbl,
            stage_cpp,
            inv_cpp,
            unique_out,
            seen,
            slot,
            int(epoch),
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cpp_end = time.perf_counter()
        cpp_time = (cpp_end - cpp_start) * 1000  # Convert to milliseconds
        cpp_times.append(cpp_time)
        
        epoch += 1
        if epoch >= 2**31 - 2:
            seen.zero_()
            epoch = 1

        U_cpp = int(U_cpp)
        T_cpp = int(T_cpp)

        if not args.benchmark:
            # Correctness checks (skip in benchmark mode)
            if T_cpp != T_py or T_py != T:
                raise RuntimeError(f"T mismatch: py={T_py}, cpp={T_cpp}, expected={T}")

            if U_cpp > Umax or U_py > Umax:
                raise RuntimeError(f"U overflow: py={U_py}, cpp={U_cpp}, Umax={Umax}")

            if not args.skip_correctness:
                # Semantic correctness for Python stage/inv
                ok_py, info_py = reconstruct_from_stage(token_ids, cpu_tbl, stage_py, inv_py[:T], U_py)
                if not ok_py:
                    t, l, d, got, exp = info_py
                    raise RuntimeError(
                        f"[Trial {trial} case={case}] Python ref failed at t={t} l={l} d={d}: got={got} exp={exp}"
                    )

                # Semantic correctness for C++ stage/inv
                ok_cpp, info_cpp = reconstruct_from_stage(token_ids, cpu_tbl, stage_cpp, inv_cpp[:T], U_cpp)
                if not ok_cpp:
                    t, l, d, got, exp = info_cpp
                    raise RuntimeError(
                        f"[Trial {trial} case={case}] C++ op failed at t={t} l={l} d={d}: got={got} exp={exp}"
                    )

                # Cross-check: reconstructed embeddings from both paths match per token
                # (This is stronger than needed, but should hold because both reconstruct cpu_tbl[token].)
                gathered_py = stage_py[:, inv_py[:T].to(torch.long), :].permute(1, 0, 2).contiguous()
                gathered_cpp = stage_cpp[:, inv_cpp[:T].to(torch.long), :].permute(1, 0, 2).contiguous()
                if not torch.equal(gathered_py, gathered_cpp):
                    raise RuntimeError(f"[Trial {trial} case={case}] Reconstruction mismatch between py and cpp")

        # Progress output with timing
        if trial % max(1, args.trials // 10) == 0:
            speedup = py_time / cpp_time if cpp_time > 0 else 0
            print(f"  trial {trial:4d}/{args.trials} case={case:10s}  U_py={U_py:5d} U_cpp={U_cpp:5d}  "
                  f"py={py_time:6.3f}ms  cpp={cpp_time:6.3f}ms  speedup={speedup:.2f}x")

    elapsed = time.time() - start
    
    # Print summary statistics
    if py_times and cpp_times:
        py_mean = statistics.mean(py_times)
        py_median = statistics.median(py_times)
        py_min = min(py_times)
        py_max = max(py_times)
        
        cpp_mean = statistics.mean(cpp_times)
        cpp_median = statistics.median(cpp_times)
        cpp_min = min(cpp_times)
        cpp_max = max(cpp_times)
        
        speedup_mean = py_mean / cpp_mean if cpp_mean > 0 else 0
        speedup_median = py_median / cpp_median if cpp_median > 0 else 0
        
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print(f"{'Metric':<20} {'Python (ms)':<15} {'C++ (ms)':<15} {'Speedup':<15}")
        print("-"*70)
        print(f"{'Mean':<20} {py_mean:>12.3f}    {cpp_mean:>12.3f}    {speedup_mean:>12.2f}x")
        print(f"{'Median':<20} {py_median:>12.3f}    {cpp_median:>12.3f}    {speedup_median:>12.2f}x")
        print(f"{'Min':<20} {py_min:>12.3f}    {cpp_min:>12.3f}    {py_min/cpp_min if cpp_min > 0 else 0:>12.2f}x")
        print(f"{'Max':<20} {py_max:>12.3f}    {cpp_max:>12.3f}    {py_max/cpp_max if cpp_max > 0 else 0:>12.2f}x")
        print("="*70)
        
        # Calculate throughput
        total_tokens = args.trials * T
        py_throughput = total_tokens / (sum(py_times) / 1000)  # tokens per second
        cpp_throughput = total_tokens / (sum(cpp_times) / 1000)  # tokens per second
        print(f"\nThroughput:")
        print(f"  Python: {py_throughput:,.0f} tokens/sec")
        print(f"  C++:    {cpp_throughput:,.0f} tokens/sec")
        print(f"  Speedup: {cpp_throughput/py_throughput if py_throughput > 0 else 0:.2f}x")
        print("="*70)
    
    if not args.benchmark:
        print(f"\nOK: all trials passed in {elapsed:.3f}s")
    else:
        print(f"\nBenchmark completed in {elapsed:.3f}s")


if __name__ == "__main__":
    main()