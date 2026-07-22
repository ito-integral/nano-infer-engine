import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2


MODEL_PATH = Path(
    os.getenv("NANO_INFER_MODEL_PATH", "/home/a/dm/models/Llama-3.2-1B-Instruct")
)


def test_cached_logits_match_no_cache() -> None:
    if not (MODEL_PATH / "config.json").is_file():
        pytest.skip(f"model not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)(
        "The capital of France is", return_tensors="pt"
    ).input_ids.to(device)
    model = Llama3_2(load_convert_hf_config(MODEL_PATH))
    load_llama(model, MODEL_PATH)
    model = model.to(device).eval()

    with torch.inference_mode():
        expected = model(tokens).float()
        cached_steps = []
        k_caches = v_caches = None
        for position in range(tokens.shape[1]):
            logits, k_caches, v_caches = model(
                tokens[:, position : position + 1],
                k_caches=k_caches,
                v_caches=v_caches,
                use_cache=True,
                cache_position=position,
                cache_capacity=tokens.shape[1] if position == 0 else None,
            )
            cached_steps.append(logits)
        actual = torch.cat(cached_steps, dim=1).float()

    diff = (actual - expected).abs()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    metrics = {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_similarity": min(1.0, max(-1.0, cosine)),
        "top1_token_match": actual.argmax(-1)
        .eq(expected.argmax(-1))
        .float()
        .mean()
        .item(),
    }
    print(metrics)
    assert metrics["max_abs_diff"] < 1e-3
    assert metrics["mean_abs_diff"] < 1e-4
    assert metrics["cosine_similarity"] > 0.999999
    assert metrics["top1_token_match"] == 1.0
