"""Model-family adapter registry."""

from pathlib import Path

from .base import BaseModelAdapter
from .llama import LlamaAdapter
from .qwen2 import Qwen2Adapter


_ADAPTERS: dict[str, type[BaseModelAdapter]] = {
    "qwen2": Qwen2Adapter,
    "llama": LlamaAdapter,
}


def get_adapter(
    model_family: str,
    model_path: str | Path,
    *,
    attn_implementation: str = "eager",
) -> BaseModelAdapter:
    family = model_family.lower()
    try:
        adapter_cls = _ADAPTERS[family]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported model_family={model_family!r}. Supported families: {supported}") from exc
    return adapter_cls(Path(model_path), attn_implementation=attn_implementation)


__all__ = ["BaseModelAdapter", "LlamaAdapter", "Qwen2Adapter", "get_adapter"]
