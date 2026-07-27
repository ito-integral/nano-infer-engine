import pytest

from nano_infer_engine.models.llama import LlamaConfig
from nano_infer_engine.service.runtime import _apply_max_model_len


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
