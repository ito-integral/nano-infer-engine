import pytest

from nano_infer_engine.generation.config import GenerationConfig


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_prefill_chunk_size_must_be_positive(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="prefill_chunk_size must be positive"):
        GenerationConfig(prefill_chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [True, 1.5, "4"])
def test_prefill_chunk_size_must_be_an_integer(chunk_size: object) -> None:
    with pytest.raises(TypeError, match="prefill_chunk_size must be an integer"):
        GenerationConfig(prefill_chunk_size=chunk_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("token_budget", [0, -1])
def test_max_prefill_tokens_per_step_must_be_positive(token_budget: int) -> None:
    with pytest.raises(
        ValueError, match="max_prefill_tokens_per_step must be positive"
    ):
        GenerationConfig(
            prefill_chunk_size=2,
            max_prefill_tokens_per_step=token_budget,
        )


@pytest.mark.parametrize("token_budget", [True, 1.5, "4"])
def test_max_prefill_tokens_per_step_must_be_an_integer(
    token_budget: object,
) -> None:
    with pytest.raises(
        TypeError, match="max_prefill_tokens_per_step must be an integer"
    ):
        GenerationConfig(
            prefill_chunk_size=2,
            max_prefill_tokens_per_step=token_budget,  # type: ignore[arg-type]
        )


def test_max_prefill_tokens_per_step_requires_chunking() -> None:
    with pytest.raises(
        ValueError, match="max_prefill_tokens_per_step requires prefill_chunk_size"
    ):
        GenerationConfig(max_prefill_tokens_per_step=4)
