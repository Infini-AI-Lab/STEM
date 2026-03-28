from dataclasses import dataclass
from typing import Optional, Type

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from lingua.checkpoint import load_from_checkpoint
from lingua.transformer import cross_entropy
from apps.main import stem_train
from apps.main import train as base_train


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
        _compile_teacher = False

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
            if self._compile_teacher:
                teacher_model.compile()

            # Store as a non-registered attribute so it is not optimized/saved.
            object.__setattr__(self, "_teacher_model", teacher_model)
            return teacher_model

        @torch.compiler.disable
        def compute_teacher_logits(
            self,
            token_values: torch.Tensor,
            tok_idx: Optional[torch.Tensor] = None,
            mask=None,
            attn_impl: str = "sdpa",
        ) -> torch.Tensor:
            teacher_model = self._get_teacher_model()
            with torch.no_grad():
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
            total_loss = self._ce_loss_weight * ce_loss + self._distill_loss_weight * distill_loss
            # Keep logging comparable with other methods: forward value is CE,
            # but gradients come from the weighted distillation objective.
            return total_loss + (ce_loss - total_loss).detach()

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
        distill_cls._compile_teacher = bool(args.distributed.compile)
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
