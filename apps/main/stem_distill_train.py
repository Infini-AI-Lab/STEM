from dataclasses import dataclass
from typing import Optional, Type

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from lingua.checkpoint import load_from_checkpoint
from lingua.transformer import cross_entropy
from apps.main import stem_train
from apps.main import train as base_train

# Chunk size along the sequence dimension when computing KL divergence.
# Smaller values reduce peak GPU memory at the cost of slightly more kernel
# launches. Tune based on vocab size: 128 works for V≥32K; increase for
# smaller vocab or if memory is not the bottleneck.
_KL_CHUNK_SIZE = 2048


def _chunked_kl_div(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    """KL divergence computed in sequence-length chunks to cap peak memory.

    Instead of materialising full [B, T, V] log-prob tensors twice, we iterate
    over chunks of size _KL_CHUNK_SIZE along T and accumulate a scalar sum.
    Peak extra allocation is 4 × [B, chunk, V] rather than 4 × [B, T, V].
    """
    B, T, _ = student_logits.shape
    kl_sum = student_logits.new_zeros(())
    for start in range(0, T, _KL_CHUNK_SIZE):
        s_chunk = student_logits[:, start : start + _KL_CHUNK_SIZE, :]
        t_chunk = teacher_logits[:, start : start + _KL_CHUNK_SIZE, :]
        s_lp = F.log_softmax(s_chunk / temp, dim=-1)
        t_lp = F.log_softmax(t_chunk.float() / temp, dim=-1)
        kl_sum = kl_sum + F.kl_div(s_lp, t_lp, reduction="sum", log_target=True)
    # Normalise to match the original batchmean-over-sequence convention and
    # apply the temperature-squared scaling from the Hinton et al. recipe.
    return kl_sum / (B * T) * (temp ** 2)


@dataclass
class DistillStemTrainArgs(stem_train.StemTrainArgs):
    ce_loss_weight: float = 1.0
    distill_loss_weight: float = 1.0
    distill_temperature: float = 1.0
    teacher_ckpt_path: str = ""


def build_distill_model_cls(
    base_model_cls: Type[torch.nn.Module],
    teacher_model_cls: Type[torch.nn.Module],
    ce_loss_weight: float,
    distill_loss_weight: float,
    distill_temperature: float,
    teacher_ckpt_path: str,
) -> Type[torch.nn.Module]:
    class DistillStemModel(base_model_cls):
        _ce_loss_weight = ce_loss_weight
        _distill_loss_weight = distill_loss_weight
        _distill_temperature = distill_temperature
        _teacher_model_cls = teacher_model_cls
        _teacher_ckpt_path = teacher_ckpt_path

        def _get_teacher_model(self) -> torch.nn.Module:
            teacher_model = getattr(self, "_teacher_model", None)
            if teacher_model is not None:
                return teacher_model

            if not self._teacher_ckpt_path:
                raise ValueError(
                    "Teacher checkpoint is required. Set `teacher_ckpt_path` or "
                    "`checkpoint.init_ckpt_path` in config."
                )

            teacher_model = self._teacher_model_cls(self.args).cuda()
            load_from_checkpoint(
                self._teacher_ckpt_path,
                teacher_model,
                model_key="model",
            )
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model = torch.compile(teacher_model)

            # Store as a non-registered attribute so it is not optimized/saved.
            object.__setattr__(self, "_teacher_model", teacher_model)
            return teacher_model

        def compute_teacher_logits(
            self,
            token_values: torch.Tensor,
            tok_idx: Optional[torch.Tensor] = None,
            mask=None,
            attn_impl: str = "sdpa",
        ) -> torch.Tensor:
            teacher_model = self._get_teacher_model()
            with torch.no_grad():
                # Cast to bf16 to halve memory bandwidth for the logit tensor;
                # the chunked KL div upcasts teacher log-probs to float32
                # internally so numerical precision is preserved.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return teacher_model(
                        token_values=token_values,
                        target=None,
                        tok_idx=tok_idx,
                        mask=mask,
                        attn_impl=attn_impl,
                    )

        def forward(
            self,
            token_values: torch.Tensor,
            target: torch.Tensor = None,
            tok_idx: torch.Tensor = None,
            mask=None,
            attn_impl: str = "sdpa",
            teacher_logits: torch.Tensor = None,
        ):
            student_logits = super().forward(
                token_values=token_values,
                target=None,
                tok_idx=tok_idx,
                mask=mask,
                attn_impl=attn_impl,
            )

            if target is None:
                return student_logits

            ce_loss = cross_entropy(student_logits, target)
            if teacher_logits is None:
                teacher_logits = self.compute_teacher_logits(
                    token_values=token_values,
                    tok_idx=tok_idx,
                    mask=mask,
                    attn_impl=attn_impl,
                )

            temp = self._distill_temperature
            distill_loss = _chunked_kl_div(student_logits, teacher_logits, temp)
            total_loss = self._ce_loss_weight * ce_loss + self._distill_loss_weight * distill_loss
            # loss.item() is CE (for loss/out); backward follows total_loss.
            loss_for_backward = total_loss + (ce_loss - total_loss).detach()
            return stem_train.StemTrainLossOut(
                loss=loss_for_backward,
                distill_loss=distill_loss.detach(),
            )

    DistillStemModel.__name__ = f"{base_model_cls.__name__}Distill"
    return DistillStemModel


def patch_registry_for_distillation(args: DistillStemTrainArgs):
    patched_registry = {}
    teacher_ckpt_path = args.teacher_ckpt_path or args.checkpoint.init_ckpt_path
    for model_type, (
        model_cls,
        args_cls,
        build_fsdp_plan,
        get_no_recompute_ops,
        get_num_flop_per_token,
    ) in stem_train.STEM_MODEL_REGISTRY.items():
        teacher_model_cls = base_train.MODEL_REGISTRY[model_type][0]
        distill_cls = build_distill_model_cls(
            model_cls,
            teacher_model_cls=teacher_model_cls,
            ce_loss_weight=args.ce_loss_weight,
            distill_loss_weight=args.distill_loss_weight,
            distill_temperature=args.distill_temperature,
            teacher_ckpt_path=teacher_ckpt_path,
        )
        patched_registry[model_type] = (
            distill_cls,
            args_cls,
            build_fsdp_plan,
            get_no_recompute_ops,
            get_num_flop_per_token,
        )
    return patched_registry


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(DistillStemTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    if cfg.distill_temperature <= 0:
        raise ValueError("distill_temperature must be > 0")
    if not (cfg.teacher_ckpt_path or cfg.checkpoint.init_ckpt_path):
        raise ValueError(
            "Provide `teacher_ckpt_path` (or `checkpoint.init_ckpt_path`) "
            "for the frozen baseline teacher."
        )

    original_registry = stem_train.STEM_MODEL_REGISTRY
    stem_train.STEM_MODEL_REGISTRY = patch_registry_for_distillation(cfg)
    try:
        stem_train.train(cfg)
    finally:
        stem_train.STEM_MODEL_REGISTRY = original_registry


if __name__ == "__main__":
    main()
