"""Source-only, two-stage Latent Compaction for Qwen3 and Llama caches."""

from .memory import CompactedMemory
from .prompts import SOURCE_PROMPTS
from .models import DocumentCache, generate_from_memory, install_attention_adapter, load_model, resolve_model_name
from .methods.latent import FitConfig, fit_document, load_memory, save_memory

__all__ = [
    "CompactedMemory",
    "DocumentCache",
    "FitConfig",
    "SOURCE_PROMPTS",
    "fit_document",
    "generate_from_memory",
    "install_attention_adapter",
    "load_memory",
    "load_model",
    "resolve_model_name",
    "save_memory",
]
