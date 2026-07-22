import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2


MODEL_PATH = Path(
    os.getenv("NANO_INFER_MODEL_PATH", "/home/a/dm/models/Llama-3.2-1B-Instruct")
)


def test_bf16_top1_matches_reference() -> None:
    if not (MODEL_PATH / "config.json").is_file():
        pytest.skip(f"model not found: {MODEL_PATH}")
    if not torch.cuda.is_available():
        pytest.skip("the bfloat16 parity test requires CUDA")
    device = torch.device("cuda")
    input_ids = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)(
        "The capital of France is", return_tensors="pt"
    ).input_ids.to(device)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    model = Llama3_2(load_convert_hf_config(MODEL_PATH))
    load_llama(model, MODEL_PATH)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    with torch.inference_mode():
        expected = reference(input_ids).logits.float()
        actual = model(input_ids).float()

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
    assert metrics["cosine_similarity"] > 0.999
    assert metrics["top1_token_match"] == 1.0
