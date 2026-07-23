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
