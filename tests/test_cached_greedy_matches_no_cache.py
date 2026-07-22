import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nano_infer_engine.cache import KVCache
from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2


MODEL_PATH = Path(
    os.getenv("NANO_INFER_MODEL_PATH", "/home/a/dm/models/Llama-3.2-1B-Instruct")
)
MAX_NEW_TOKENS = 4


@torch.inference_mode()
def greedy_trace(model, tokens, use_cache):
    sequence = tokens.clone()
    trace = []
    if use_cache:
        kv_cache = KVCache(
            num_layers=len(model.decoders),
            batch_size=tokens.shape[0],
            capacity=tokens.shape[1] + MAX_NEW_TOKENS,
            kv_head_num=model.config.kv_head_num,
            head_dim=model.config.head_dim,
            dtype=model.embed.weight.dtype,
            device=tokens.device,
        )
        logits = model(sequence, kv_cache=kv_cache)
    for step in range(MAX_NEW_TOKENS):
        if not use_cache:
            logits = model(sequence)
        last_logits = logits[:, -1]
        trace.append(last_logits)
        next_token = last_logits.argmax(-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
        if use_cache and step + 1 < MAX_NEW_TOKENS:
            logits = model(next_token, kv_cache=kv_cache)
    return sequence, torch.stack(trace, dim=1).float()


def test_cached_greedy_matches_no_cache() -> None:
    if not (MODEL_PATH / "config.json").is_file():
        pytest.skip(f"model not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)(
        "The capital of France is", return_tensors="pt"
    ).input_ids.to(device)
    model = Llama3_2(load_convert_hf_config(MODEL_PATH))
    load_llama(model, MODEL_PATH)
    model = model.to(device).eval()

    expected_tokens, expected = greedy_trace(model, tokens, use_cache=False)
    actual_tokens, actual = greedy_trace(model, tokens, use_cache=True)
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
    assert torch.equal(actual_tokens, expected_tokens)
    assert metrics["max_abs_diff"] < 1e-3
    assert metrics["mean_abs_diff"] < 1e-4
    assert metrics["cosine_similarity"] > 0.999999
    assert metrics["top1_token_match"] == 1.0
