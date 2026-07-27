import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nano_infer_engine.generation.async_engine import (
    AsyncInferenceEngine,
    AsyncPDInferenceEngine,
)
from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache

from .app import InferenceRuntime


DEFAULT_MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"
DEFAULT_KV_CACHE_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024


def _apply_max_model_len(model_config: LlamaConfig) -> None:
    configured_value = os.getenv("NANO_MAX_MODEL_LEN")
    if configured_value is None:
        return
    try:
        max_model_len = int(configured_value)
    except ValueError:
        raise ValueError("NANO_MAX_MODEL_LEN must be an integer") from None
    if max_model_len <= 0:
        raise ValueError("NANO_MAX_MODEL_LEN must be positive")
    if max_model_len > model_config.max_seq_len:
        raise ValueError(
            "NANO_MAX_MODEL_LEN cannot exceed the model's "
            f"max_position_embeddings ({model_config.max_seq_len})"
        )
    model_config.max_seq_len = max_model_len


def _resolve_served_model_name(model_path: Path) -> str:
    configured_name = os.getenv("NANO_SERVED_MODEL_NAME")
    if configured_name is None:
        return str(model_path)
    served_model_name = configured_name.strip()
    if not served_model_name:
        raise ValueError("NANO_SERVED_MODEL_NAME must not be empty")
    return served_model_name


def _resolve_gpu_memory_utilization(
    specific_env_name: str | None = None,
) -> float | None:
    configured_value = (
        os.getenv(specific_env_name) if specific_env_name is not None else None
    )
    if configured_value is None:
        configured_value = os.getenv("NANO_GPU_MEMORY_UTILIZATION")
    if configured_value is None:
        return None
    try:
        utilization = float(configured_value)
    except ValueError:
        raise ValueError(
            "GPU memory utilization must be a number"
        ) from None
    if utilization <= 0 or utilization > 1:
        raise ValueError("GPU memory utilization must be in the range (0, 1]")
    return utilization


def _resolve_kv_cache_safety_margin_bytes() -> int:
    configured_value = os.getenv("NANO_KV_CACHE_SAFETY_MARGIN_BYTES")
    if configured_value is None:
        return DEFAULT_KV_CACHE_SAFETY_MARGIN_BYTES
    try:
        safety_margin_bytes = int(configured_value)
    except ValueError:
        raise ValueError(
            "NANO_KV_CACHE_SAFETY_MARGIN_BYTES must be an integer"
        ) from None
    if safety_margin_bytes < 0:
        raise ValueError(
            "NANO_KV_CACHE_SAFETY_MARGIN_BYTES must be non-negative"
        )
    return safety_margin_bytes


def _kv_block_bytes(model: Llama3_2, block_size: int) -> int:
    return (
        2
        * len(model.decoders)
        * block_size
        * model.config.kv_head_num
        * model.config.head_dim
        * model.embed.weight.element_size()
    )


def _calculate_num_blocks_from_memory(
    model: Llama3_2,
    *,
    device: torch.device,
    block_size: int,
    utilization: float,
    safety_margin_bytes: int,
) -> int:
    if device.type != "cuda":
        raise ValueError("GPU memory utilization requires a CUDA device")

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    target_process_bytes = int(total_bytes * utilization)
    budget_within_target = target_process_bytes - reserved_bytes
    budget_within_free_memory = free_bytes - safety_margin_bytes
    available_kv_bytes = min(
        budget_within_target,
        budget_within_free_memory,
    )
    block_bytes = _kv_block_bytes(model, block_size)
    num_blocks = available_kv_bytes // block_bytes
    if num_blocks <= 0:
        raise ValueError(
            "GPU memory budget cannot fit a single KV cache block"
        )
    return num_blocks


def _resolve_num_blocks(
    model: Llama3_2,
    *,
    device: torch.device,
    block_size: int,
    explicit_env_name: str,
    default_num_blocks: int,
    utilization_env_name: str | None = None,
) -> int:
    configured_blocks = os.getenv(explicit_env_name)
    if configured_blocks is not None:
        try:
            num_blocks = int(configured_blocks)
        except ValueError:
            raise ValueError(
                f"{explicit_env_name} must be an integer"
            ) from None
        if num_blocks <= 0:
            raise ValueError(f"{explicit_env_name} must be positive")
        return num_blocks

    utilization = _resolve_gpu_memory_utilization(utilization_env_name)
    if utilization is None:
        return default_num_blocks
    return _calculate_num_blocks_from_memory(
        model,
        device=device,
        block_size=block_size,
        utilization=utilization,
        safety_margin_bytes=_resolve_kv_cache_safety_margin_bytes(),
    )


def _load_model(
    model_config,
    model_path: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> Llama3_2:
    model = Llama3_2(model_config).to(dtype=dtype)
    load_llama(model, model_path)
    return model.to(device).eval()


def _build_cache(
    model: Llama3_2,
    *,
    num_blocks: int,
    block_size: int,
    device: torch.device,
) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=device,
    )


def build_default_runtime() -> InferenceRuntime:
    """Load the configured model and construct one process-wide engine."""
    model_path = Path(os.getenv("NANO_MODEL_PATH", DEFAULT_MODEL_PATH))
    served_model_name = _resolve_served_model_name(model_path)
    device_name = os.getenv(
        "NANO_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    block_size = int(os.getenv("NANO_BLOCK_SIZE", "16"))
    max_batch_size = int(os.getenv("NANO_MAX_BATCH_SIZE", "8"))
    max_new_tokens = int(os.getenv("NANO_MAX_NEW_TOKENS", "128"))

    model_config = load_convert_hf_config(model_path)
    _apply_max_model_len(model_config)
    model = _load_model(model_config, model_path, dtype, device)
    num_blocks = _resolve_num_blocks(
        model,
        device=device,
        block_size=block_size,
        explicit_env_name="NANO_NUM_BLOCKS",
        default_num_blocks=512,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    paged_cache = _build_cache(
        model,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
    )
    engine = AsyncInferenceEngine(
        model,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        ),
        paged_cache,
        max_batch_size=max_batch_size,
    )
    return InferenceRuntime(
        engine=engine,
        tokenizer=tokenizer,
        device=device,
        served_model_name=served_model_name,
    )


def build_pd_runtime() -> InferenceRuntime:
    """Load a two-device P/D engine for one process-wide HTTP runtime."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("P/D runtime requires at least two CUDA devices")

    model_path = Path(os.getenv("NANO_MODEL_PATH", DEFAULT_MODEL_PATH))
    served_model_name = _resolve_served_model_name(model_path)
    prefill_device = torch.device(
        os.getenv("NANO_PREFILL_DEVICE", "cuda:0")
    )
    decode_device = torch.device(os.getenv("NANO_DECODE_DEVICE", "cuda:1"))
    if prefill_device == decode_device:
        raise ValueError("prefill and decode devices must be different")

    dtype = torch.bfloat16
    block_size = int(os.getenv("NANO_BLOCK_SIZE", "16"))
    max_batch_size = int(os.getenv("NANO_MAX_BATCH_SIZE", "8"))
    max_new_tokens = int(os.getenv("NANO_MAX_NEW_TOKENS", "128"))

    model_config = load_convert_hf_config(model_path)
    _apply_max_model_len(model_config)
    prefill_model = _load_model(
        model_config,
        model_path,
        dtype,
        prefill_device,
    )
    decode_model = _load_model(
        model_config,
        model_path,
        dtype,
        decode_device,
    )
    prefill_num_blocks = _resolve_num_blocks(
        prefill_model,
        device=prefill_device,
        block_size=block_size,
        explicit_env_name="NANO_PREFILL_NUM_BLOCKS",
        default_num_blocks=128,
        utilization_env_name="NANO_PREFILL_GPU_MEMORY_UTILIZATION",
    )
    decode_num_blocks = _resolve_num_blocks(
        decode_model,
        device=decode_device,
        block_size=block_size,
        explicit_env_name="NANO_DECODE_NUM_BLOCKS",
        default_num_blocks=512,
        utilization_env_name="NANO_DECODE_GPU_MEMORY_UTILIZATION",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    engine = AsyncPDInferenceEngine(
        prefill_model,
        decode_model,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        ),
        _build_cache(
            prefill_model,
            num_blocks=prefill_num_blocks,
            block_size=block_size,
            device=prefill_device,
        ),
        _build_cache(
            decode_model,
            num_blocks=decode_num_blocks,
            block_size=block_size,
            device=decode_device,
        ),
        max_batch_size=max_batch_size,
    )
    return InferenceRuntime(
        engine=engine,
        tokenizer=tokenizer,
        device=prefill_device,
        served_model_name=served_model_name,
    )
