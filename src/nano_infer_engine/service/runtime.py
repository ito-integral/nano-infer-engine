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
from nano_infer_engine.models.llama import Llama3_2
from nano_infer_engine.paged_cache import PagedKVCache

from .app import InferenceRuntime


DEFAULT_MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"


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
    device_name = os.getenv(
        "NANO_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    block_size = int(os.getenv("NANO_BLOCK_SIZE", "16"))
    num_blocks = int(os.getenv("NANO_NUM_BLOCKS", "512"))
    max_batch_size = int(os.getenv("NANO_MAX_BATCH_SIZE", "8"))
    max_new_tokens = int(os.getenv("NANO_MAX_NEW_TOKENS", "128"))

    model_config = load_convert_hf_config(model_path)
    model = _load_model(model_config, model_path, dtype, device)
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
    )


def build_pd_runtime() -> InferenceRuntime:
    """Load a two-device P/D engine for one process-wide HTTP runtime."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("P/D runtime requires at least two CUDA devices")

    model_path = Path(os.getenv("NANO_MODEL_PATH", DEFAULT_MODEL_PATH))
    prefill_device = torch.device(
        os.getenv("NANO_PREFILL_DEVICE", "cuda:0")
    )
    decode_device = torch.device(os.getenv("NANO_DECODE_DEVICE", "cuda:1"))
    if prefill_device == decode_device:
        raise ValueError("prefill and decode devices must be different")

    dtype = torch.bfloat16
    block_size = int(os.getenv("NANO_BLOCK_SIZE", "16"))
    prefill_num_blocks = int(
        os.getenv("NANO_PREFILL_NUM_BLOCKS", "128")
    )
    decode_num_blocks = int(
        os.getenv("NANO_DECODE_NUM_BLOCKS", "512")
    )
    max_batch_size = int(os.getenv("NANO_MAX_BATCH_SIZE", "8"))
    max_new_tokens = int(os.getenv("NANO_MAX_NEW_TOKENS", "128"))

    model_config = load_convert_hf_config(model_path)
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
    )
