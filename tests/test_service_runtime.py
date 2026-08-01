from pathlib import Path

import pytest
import torch

from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.service.runtime import (
    DEFAULT_KV_CACHE_SAFETY_MARGIN_BYTES,
    _apply_max_model_len,
    _calculate_num_blocks_from_memory,
    _kv_block_bytes,
    _resolve_gpu_memory_utilization,
    _resolve_kv_cache_safety_margin_bytes,
    _resolve_num_blocks,
    _resolve_optional_positive_int,
    _resolve_prefill_scheduling,
    _resolve_served_model_name,
)


def _tiny_model() -> Llama3_2:
    return Llama3_2(
        LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            mlp_inner_size=16,
            num_layers=2,
            q_head_num=2,
            kv_head_num=1,
            max_seq_len=16,
            tie_word_embeddings=False,
        )
    )


def test_max_model_len_defaults_to_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANO_MAX_MODEL_LEN", raising=False)
    model_config = LlamaConfig(max_seq_len=128)

    _apply_max_model_len(model_config)

    assert model_config.max_seq_len == 128


def test_max_model_len_can_limit_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANO_MAX_MODEL_LEN", "96")
    model_config = LlamaConfig(max_seq_len=128)

    _apply_max_model_len(model_config)

    assert model_config.max_seq_len == 96


@pytest.mark.parametrize(
    ("configured_value", "message"),
    [
        ("invalid", "must be an integer"),
        ("0", "must be positive"),
        ("129", "cannot exceed"),
    ],
)
def test_max_model_len_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str,
    message: str,
) -> None:
    monkeypatch.setenv("NANO_MAX_MODEL_LEN", configured_value)
    model_config = LlamaConfig(max_seq_len=128)

    with pytest.raises(ValueError, match=message):
        _apply_max_model_len(model_config)


def test_optional_positive_int_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANO_PREFILL_CHUNK_SIZE", raising=False)

    assert _resolve_optional_positive_int("NANO_PREFILL_CHUNK_SIZE") is None


@pytest.mark.parametrize("configured_value", ["invalid", "0", "-1"])
def test_optional_positive_int_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str,
) -> None:
    monkeypatch.setenv("NANO_PREFILL_CHUNK_SIZE", configured_value)

    with pytest.raises(ValueError, match="NANO_PREFILL_CHUNK_SIZE"):
        _resolve_optional_positive_int("NANO_PREFILL_CHUNK_SIZE")


def test_prefill_scheduling_uses_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANO_PREFILL_CHUNK_SIZE", "128")
    monkeypatch.setenv("NANO_MAX_PREFILL_TOKENS_PER_STEP", "512")

    assert _resolve_prefill_scheduling() == (128, 512)


def test_prefill_budget_requires_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANO_PREFILL_CHUNK_SIZE", raising=False)
    monkeypatch.setenv("NANO_MAX_PREFILL_TOKENS_PER_STEP", "512")

    with pytest.raises(ValueError, match="requires NANO_PREFILL_CHUNK_SIZE"):
        _resolve_prefill_scheduling()


def test_served_model_name_defaults_to_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANO_SERVED_MODEL_NAME", raising=False)

    assert _resolve_served_model_name(Path("/models/llama")) == "/models/llama"


def test_served_model_name_uses_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANO_SERVED_MODEL_NAME", " llama-3.2-1b ")

    assert _resolve_served_model_name(Path("/models/llama")) == "llama-3.2-1b"


def test_served_model_name_rejects_empty_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANO_SERVED_MODEL_NAME", "  ")

    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_served_model_name(Path("/models/llama"))


def test_gpu_memory_utilization_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANO_GPU_MEMORY_UTILIZATION", raising=False)

    assert _resolve_gpu_memory_utilization() is None


def test_specific_gpu_memory_utilization_overrides_global_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANO_GPU_MEMORY_UTILIZATION", "0.8")
    monkeypatch.setenv("NANO_DECODE_GPU_MEMORY_UTILIZATION", "0.9")

    assert (
        _resolve_gpu_memory_utilization(
            "NANO_DECODE_GPU_MEMORY_UTILIZATION"
        )
        == 0.9
    )


@pytest.mark.parametrize("configured_value", ["invalid", "0", "1.1"])
def test_gpu_memory_utilization_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str,
) -> None:
    monkeypatch.setenv("NANO_GPU_MEMORY_UTILIZATION", configured_value)

    with pytest.raises(ValueError, match="GPU memory utilization"):
        _resolve_gpu_memory_utilization()


def test_kv_cache_safety_margin_defaults_to_512_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANO_KV_CACHE_SAFETY_MARGIN_BYTES", raising=False)

    assert (
        _resolve_kv_cache_safety_margin_bytes()
        == DEFAULT_KV_CACHE_SAFETY_MARGIN_BYTES
    )


def test_kv_block_bytes_includes_all_layers_keys_and_values() -> None:
    model = _tiny_model()

    assert _kv_block_bytes(model, block_size=4) == 256


def test_num_blocks_is_calculated_from_available_gpu_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model()
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (8_000, 10_000),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: 2_000,
    )

    num_blocks = _calculate_num_blocks_from_memory(
        model,
        device=torch.device("cuda:0"),
        block_size=4,
        utilization=0.8,
        safety_margin_bytes=1_000,
    )

    assert num_blocks == 23


def test_explicit_num_blocks_takes_precedence_over_memory_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANO_NUM_BLOCKS", "7")
    monkeypatch.setenv("NANO_GPU_MEMORY_UTILIZATION", "0.9")

    assert (
        _resolve_num_blocks(
            _tiny_model(),
            device=torch.device("cpu"),
            block_size=4,
            explicit_env_name="NANO_NUM_BLOCKS",
            default_num_blocks=512,
        )
        == 7
    )


def test_num_blocks_uses_default_without_memory_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANO_NUM_BLOCKS", raising=False)
    monkeypatch.delenv("NANO_GPU_MEMORY_UTILIZATION", raising=False)

    assert (
        _resolve_num_blocks(
            _tiny_model(),
            device=torch.device("cpu"),
            block_size=4,
            explicit_env_name="NANO_NUM_BLOCKS",
            default_num_blocks=512,
        )
        == 512
    )
