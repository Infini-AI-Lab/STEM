import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from lingua.checkpoint import load_from_checkpoint
from lingua.tokenizer import build_tokenizer
from lingua.transformer import cross_entropy
from apps.main import stem_train
from apps.main import train as base_train
from apps.main.stem_distill_train import DistillStemTrainArgs, build_distill_model_cls


def _parse_stem_layers(raw: str) -> List[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("stem_layers cannot be empty")
    layers = [int(x) for x in vals]
    if len(set(layers)) != len(layers):
        raise ValueError(f"Duplicate stem layer IDs found: {layers}")
    return layers


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    w = rank - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def _summary(vals: List[float]) -> Dict[str, float]:
    s = sorted(vals)
    return {
        "mean_ms": mean(vals),
        "median_ms": median(vals),
        "p95_ms": _percentile(s, 0.95),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def _record_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline distillation latency benchmark for STEM models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="apps/main/configs/stem_llama3_1B_prefine_distill.yaml",
        help="Config file used to construct model args.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama",
        choices=["llama", "qwen3", "olmo3"],
    )
    parser.add_argument(
        "--stem-layers",
        type=str,
        default="2,6,10,14",
        help="Comma-separated layer IDs for STEM FFNs.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "fmha", "flex_attention"],
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Autocast dtype for student/teacher forward paths.",
    )
    parser.add_argument(
        "--student-ckpt-path",
        type=str,
        default="",
        help="Overrides checkpoint.init_ckpt_path from config.",
    )
    parser.add_argument(
        "--teacher-ckpt-path",
        type=str,
        default="",
        help="Overrides teacher_ckpt_path from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional output path for JSON benchmark report.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    file_cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.structured(DistillStemTrainArgs())
    cfg = OmegaConf.merge(base_cfg, file_cfg)
    cfg = OmegaConf.to_object(cfg)

    cfg.model_type = args.model_type
    cfg.model.stem_layers = _parse_stem_layers(args.stem_layers)
    cfg.data.batch_size = args.batch_size
    cfg.data.seq_len = args.seq_len
    if args.student_ckpt_path:
        cfg.checkpoint.init_ckpt_path = args.student_ckpt_path
    if args.teacher_ckpt_path:
        cfg.teacher_ckpt_path = args.teacher_ckpt_path

    if not cfg.checkpoint.init_ckpt_path:
        raise ValueError(
            "Student checkpoint required. Set --student-ckpt-path or checkpoint.init_ckpt_path in config."
        )
    if not cfg.teacher_ckpt_path:
        raise ValueError(
            "Teacher checkpoint required. Set --teacher-ckpt-path or teacher_ckpt_path in config."
        )

    tokenizer = build_tokenizer(cfg.data.tokenizer.name, cfg.data.tokenizer.path)
    if cfg.model.vocab_size <= 0:
        cfg.model.vocab_size = tokenizer.n_words

    if cfg.model_type not in stem_train.STEM_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{cfg.model_type}', expected one of {list(stem_train.STEM_MODEL_REGISTRY.keys())}"
        )
    if cfg.model_type not in base_train.MODEL_REGISTRY:
        raise ValueError(
            f"Teacher model_type '{cfg.model_type}' not found in base MODEL_REGISTRY"
        )

    student_model_cls = stem_train.STEM_MODEL_REGISTRY[cfg.model_type][0]
    teacher_model_cls = base_train.MODEL_REGISTRY[cfg.model_type][0]
    distill_model_cls = build_distill_model_cls(
        student_model_cls,
        teacher_model_cls=teacher_model_cls,
        ce_loss_weight=float(cfg.ce_loss_weight),
        distill_loss_weight=float(cfg.distill_loss_weight),
        distill_temperature=float(cfg.distill_temperature),
        teacher_ckpt_path=cfg.teacher_ckpt_path,
    )

    device = torch.device("cuda")
    model = distill_model_cls(cfg.model).to(device)
    model.train()
    # load_from_checkpoint(cfg.checkpoint.init_ckpt_path, model, model_key="model")

    amp_ctx = nullcontext()
    if args.amp_dtype == "bf16":
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif args.amp_dtype == "fp16":
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)

    batch_size = args.batch_size
    seq_len = args.seq_len
    vocab_size = cfg.model.vocab_size

    teacher_forward_ms: List[float] = []
    student_forward_only_ms: List[float] = []
    student_forward_grad_ms: List[float] = []
    teacher_forward_in_kl_ms: List[float] = []
    kl_loss_compute_ms: List[float] = []
    student_forward_with_kl_ms: List[float] = []
    student_backward_with_kl_ms: List[float] = []
    student_forward_no_kl_ms: List[float] = []
    student_backward_no_kl_ms: List[float] = []

    total_iters = args.warmup_iters + args.iters
    for it in range(total_iters):
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        labels = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )

        # 1) Teacher forward (no grad).
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            with amp_ctx:
                t0.record()
                _ = model._get_teacher_model()(
                    token_values=input_ids,
                    target=None,
                    tok_idx=None,
                    mask=None,
                    attn_impl=args.attn_impl,
                )
                t1.record()
        teacher_ms = _record_ms(t0, t1)

        # 2) Student forward only (no teacher/KL), no grad.
        s0 = torch.cuda.Event(enable_timing=True)
        s1 = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            with amp_ctx:
                s0.record()
                _ = student_model_cls.forward(
                    model,
                    token_values=input_ids,
                    target=None,
                    tok_idx=None,
                    mask=None,
                    attn_impl=args.attn_impl,
                )
                s1.record()
        student_only_ms = _record_ms(s0, s1)

        # 3) Student forward+backward with KL.
        model.zero_grad(set_to_none=True)
        fk0 = torch.cuda.Event(enable_timing=True)
        fk1 = torch.cuda.Event(enable_timing=True)
        sfk0 = torch.cuda.Event(enable_timing=True)
        sfk1 = torch.cuda.Event(enable_timing=True)
        tfk0 = torch.cuda.Event(enable_timing=True)
        tfk1 = torch.cuda.Event(enable_timing=True)
        kl0 = torch.cuda.Event(enable_timing=True)
        kl1 = torch.cuda.Event(enable_timing=True)
        bk0 = torch.cuda.Event(enable_timing=True)
        bk1 = torch.cuda.Event(enable_timing=True)
        with amp_ctx:
            fk0.record()
            sfk0.record()
            student_logits = student_model_cls.forward(
                model,
                token_values=input_ids,
                target=None,
                tok_idx=None,
                mask=None,
                attn_impl=args.attn_impl,
            )
            sfk1.record()

            tfk0.record()
            with torch.no_grad():
                teacher_logits = model._get_teacher_model()(
                    token_values=input_ids,
                    target=None,
                    tok_idx=None,
                    mask=None,
                    attn_impl=args.attn_impl,
                )
            tfk1.record()

            kl0.record()
            ce_loss = cross_entropy(student_logits, labels)
            temp = model._distill_temperature
            student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits / temp, dim=-1)
            distill_loss = (
                F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="batchmean",
                    log_target=True,
                ) / student_log_probs.size(1)
                * (temp ** 2)
            )
            loss_with_kl = model._ce_loss_weight * ce_loss + model._distill_loss_weight * distill_loss
            kl1.record()
            fk1.record()
        bk0.record()
        loss_with_kl.backward()
        bk1.record()
        fwd_with_kl_ms = _record_ms(fk0, fk1)
        student_fwd_grad_iter_ms = _record_ms(sfk0, sfk1)
        teacher_fwd_in_kl_iter_ms = _record_ms(tfk0, tfk1)
        kl_loss_iter_ms = _record_ms(kl0, kl1)
        bwd_with_kl_ms = _record_ms(bk0, bk1)

        # 4) Student forward+backward without KL (CE-only path).
        model.zero_grad(set_to_none=True)
        fn0 = torch.cuda.Event(enable_timing=True)
        fn1 = torch.cuda.Event(enable_timing=True)
        bn0 = torch.cuda.Event(enable_timing=True)
        bn1 = torch.cuda.Event(enable_timing=True)
        with amp_ctx:
            fn0.record()
            loss_no_kl = student_model_cls.forward(
                model,
                token_values=input_ids,
                target=labels,
                tok_idx=None,
                mask=None,
                attn_impl=args.attn_impl,
            )
            fn1.record()
        bn0.record()
        loss_no_kl.backward()
        bn1.record()
        fwd_no_kl_ms = _record_ms(fn0, fn1)
        bwd_no_kl_ms = _record_ms(bn0, bn1)
        model.zero_grad(set_to_none=True)

        if it >= args.warmup_iters:
            teacher_forward_ms.append(teacher_ms)
            student_forward_only_ms.append(student_only_ms)
            student_forward_grad_ms.append(student_fwd_grad_iter_ms)
            teacher_forward_in_kl_ms.append(teacher_fwd_in_kl_iter_ms)
            kl_loss_compute_ms.append(kl_loss_iter_ms)
            student_forward_with_kl_ms.append(fwd_with_kl_ms)
            student_backward_with_kl_ms.append(bwd_with_kl_ms)
            student_forward_no_kl_ms.append(fwd_no_kl_ms)
            student_backward_no_kl_ms.append(bwd_no_kl_ms)

            print(
                f"[iter {it - args.warmup_iters + 1:02d}/{args.iters:02d}] "
                f"teacher_fwd={teacher_ms:.2f}ms "
                f"student_fwd_only={student_only_ms:.2f}ms "
                f"student_fwd_grad={student_fwd_grad_iter_ms:.2f}ms "
                f"teacher_fwd_in_kl={teacher_fwd_in_kl_iter_ms:.2f}ms "
                f"kl_loss_compute={kl_loss_iter_ms:.2f}ms "
                f"fwd_with_kl={fwd_with_kl_ms:.2f}ms "
                f"bwd_with_kl={bwd_with_kl_ms:.2f}ms "
                f"fwd_no_kl={fwd_no_kl_ms:.2f}ms "
                f"bwd_no_kl={bwd_no_kl_ms:.2f}ms "
                f"bwd_delta={bwd_with_kl_ms - bwd_no_kl_ms:.2f}ms"
            )

    report = {
        "settings": {
            "config": args.config,
            "model_type": cfg.model_type,
            "student_ckpt_path": cfg.checkpoint.init_ckpt_path,
            "teacher_ckpt_path": cfg.teacher_ckpt_path,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "iters": args.iters,
            "warmup_iters": args.warmup_iters,
            "attn_impl": args.attn_impl,
            "amp_dtype": args.amp_dtype,
            "model_stem_layers": list(cfg.model.stem_layers),
            "model_dim": int(cfg.model.dim),
            "model_layers": int(cfg.model.n_layers),
            "model_vocab_size": int(cfg.model.vocab_size),
        },
        "timings_ms": {
            "teacher_forward": _summary(teacher_forward_ms),
            "student_forward_only": _summary(student_forward_only_ms),
            "student_forward_grad": _summary(student_forward_grad_ms),
            "teacher_forward_in_kl": _summary(teacher_forward_in_kl_ms),
            "kl_loss_compute": _summary(kl_loss_compute_ms),
            "student_forward_with_kl": _summary(student_forward_with_kl_ms),
            "student_backward_with_kl": _summary(student_backward_with_kl_ms),
            "student_forward_no_kl": _summary(student_forward_no_kl_ms),
            "student_backward_no_kl": _summary(student_backward_no_kl_ms),
        },
    }

    report["timings_ms"]["derived"] = {
        "backward_kl_overhead_mean_ms": (
            report["timings_ms"]["student_backward_with_kl"]["mean_ms"]
            - report["timings_ms"]["student_backward_no_kl"]["mean_ms"]
        ),
        "forward_kl_overhead_mean_ms": (
            report["timings_ms"]["student_forward_with_kl"]["mean_ms"]
            - report["timings_ms"]["student_forward_no_kl"]["mean_ms"]
        ),
        "forward_kl_split_sum_mean_ms": (
            report["timings_ms"]["student_forward_grad"]["mean_ms"]
            + report["timings_ms"]["teacher_forward_in_kl"]["mean_ms"]
            + report["timings_ms"]["kl_loss_compute"]["mean_ms"]
        ),
        "total_student_with_kl_mean_ms": (
            report["timings_ms"]["student_forward_with_kl"]["mean_ms"]
            + report["timings_ms"]["student_backward_with_kl"]["mean_ms"]
        ),
        "total_student_no_kl_mean_ms": (
            report["timings_ms"]["student_forward_no_kl"]["mean_ms"]
            + report["timings_ms"]["student_backward_no_kl"]["mean_ms"]
        ),
    }

    print("\n=== Distill Latency Summary (ms) ===")
    for k, v in report["timings_ms"].items():
        if k == "derived":
            continue
        print(
            f"{k:28s} mean={v['mean_ms']:.2f}  med={v['median_ms']:.2f}  "
            f"p95={v['p95_ms']:.2f}  min={v['min_ms']:.2f}  max={v['max_ms']:.2f}"
        )
    print("\n=== Derived ===")
    for k, v in report["timings_ms"]["derived"].items():
        print(f"{k:35s} {v:.2f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nSaved report to {out_path}")

    # Helps with debugging if you want to diff the exact resolved settings.
    print("\nResolved model config snippet:")
    print(
        json.dumps(
            {
                "model_type": cfg.model_type,
                "model": {
                    "dim": cfg.model.dim,
                    "n_layers": cfg.model.n_layers,
                    "n_heads": cfg.model.n_heads,
                    "n_kv_heads": cfg.model.n_kv_heads,
                    "max_seqlen": cfg.model.max_seqlen,
                    "stem_layers": cfg.model.stem_layers,
                },
                "data": {"batch_size": cfg.data.batch_size, "seq_len": cfg.data.seq_len},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
