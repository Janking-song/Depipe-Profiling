"""Quantization helpers for local stage export and profiling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


STAGE_QUANTIZATION_NONE = "none"
STAGE_QUANTIZATION_BNB_4BIT_NF4 = "bnb_4bit_nf4"
STAGE_QUANTIZATION_BNB_4BIT_FP4 = "bnb_4bit_fp4"
STAGE_QUANTIZATION_CHOICES = (
    STAGE_QUANTIZATION_NONE,
    STAGE_QUANTIZATION_BNB_4BIT_NF4,
    STAGE_QUANTIZATION_BNB_4BIT_FP4,
)


@dataclass(frozen=True)
class StageQuantizationSpec:
    method: str = STAGE_QUANTIZATION_NONE
    compute_dtype: torch.dtype | None = None

    @property
    def enabled(self) -> bool:
        return self.method != STAGE_QUANTIZATION_NONE

    @property
    def quant_type(self) -> str:
        if self.method == STAGE_QUANTIZATION_BNB_4BIT_NF4:
            return "nf4"
        if self.method == STAGE_QUANTIZATION_BNB_4BIT_FP4:
            return "fp4"
        raise ValueError(f"{self.method!r} does not define a bitsandbytes quant_type.")


def parse_stage_quantization_spec(
    method: str | None,
    *,
    compute_dtype: torch.dtype | None = None,
) -> StageQuantizationSpec:
    normalized = STAGE_QUANTIZATION_NONE if method is None else method.lower()
    if normalized not in STAGE_QUANTIZATION_CHOICES:
        supported = ", ".join(STAGE_QUANTIZATION_CHOICES)
        raise ValueError(f"Unsupported quantization={method!r}. Supported values: {supported}")
    return StageQuantizationSpec(method=normalized, compute_dtype=compute_dtype)


def replace_linear_with_bnb_linear4bit(stage: nn.Module, spec: StageQuantizationSpec) -> nn.Module:
    """Replace all nn.Linear modules in-place with bitsandbytes Linear4bit."""

    if not spec.enabled:
        return stage

    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for 4-bit stage quantization. Install it in the active environment first."
        ) from exc

    for module_name, child in list(stage.named_children()):
        if isinstance(child, nn.Linear):
            replacement = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=spec.compute_dtype,
                quant_type=spec.quant_type,
                compress_statistics=True,
                quant_storage=torch.uint8,
                device=child.weight.device,
            )
            replacement.source_cls = type(child)
            replacement.requires_grad_(False)
            stage._modules[module_name] = replacement
            continue
        replace_linear_with_bnb_linear4bit(child, spec)
    return stage


def iter_bnb_linear_weight_names(stage: nn.Module) -> list[str]:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return []

    names: list[str] = []
    for module_name, module in stage.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names.append(f"{module_name}.weight" if module_name else "weight")
    return sorted(names)

