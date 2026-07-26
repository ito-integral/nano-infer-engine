import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nano_infer_engine.generation.async_engine import AsyncInferenceEngine
from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2
from nano_infer_engine.paged_cache import PagedKVCache

from .app import InferenceRuntime


DEFAULT_MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"


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
    model = Llama3_2(model_config).to(dtype=dtype)
    load_llama(model, model_path)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    paged_cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
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
