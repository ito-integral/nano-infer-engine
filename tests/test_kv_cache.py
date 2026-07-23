import pytest
import torch

from nano_infer_engine.cache import KVCache


def _build_cache(**overrides: object) -> KVCache:
    arguments = {
        "num_layers": 2,
        "batch_size": 3,
        "capacity": 5,
        "kv_head_num": 2,
        "head_dim": 4,
        "dtype": torch.float32,
        "device": "cpu",
    }
    arguments.update(overrides)
    return KVCache(**arguments)  # type: ignore[arg-type]


def test_initializes_per_layer_storage() -> None:
    cache = _build_cache()
    expected_shape = (3, 5, 2, 4)

    assert cache.position == 0
    assert cache.capacity == 5
    assert len(cache.keys) == 2
    assert len(cache.values) == 2

    for tensor in [*cache.keys, *cache.values]:
        assert tensor.shape == expected_shape
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"

    assert cache.keys[0] is not cache.keys[1]
    assert cache.values[0] is not cache.values[1]
    key, value = cache.get(0)
    assert key is cache.keys[0]
    assert value is cache.values[0]


def test_advance_tracks_position_and_allows_exact_capacity() -> None:
    cache = _build_cache(capacity=5)

    cache.advance(2)
    assert cache.position == 2

    cache.advance(3)
    assert cache.position == 5


def test_advance_rejects_capacity_overflow_without_changing_position() -> None:
    cache = _build_cache(capacity=5)
    cache.advance(4)

    with pytest.raises(ValueError, match="KV cache capacity exceeded"):
        cache.advance(2)

    assert cache.position == 4


def test_reset_preserves_allocated_storage() -> None:
    cache = _build_cache()
    keys = list(cache.keys)
    values = list(cache.values)
    cache.advance(3)

    cache.reset()

    assert cache.position == 0
    assert all(actual is original for actual, original in zip(cache.keys, keys))
    assert all(actual is original for actual, original in zip(cache.values, values))


@pytest.mark.parametrize(
    "name",
    ["num_layers", "batch_size", "capacity", "kv_head_num", "head_dim"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_dimensions_must_be_positive(name: str, value: int) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        _build_cache(**{name: value})


@pytest.mark.parametrize(
    "name",
    ["num_layers", "batch_size", "capacity", "kv_head_num", "head_dim"],
)
@pytest.mark.parametrize("value", [True, 1.5])
def test_dimensions_must_be_integers(name: str, value: object) -> None:
    with pytest.raises(TypeError, match=rf"{name} must be an integer"):
        _build_cache(**{name: value})


@pytest.mark.parametrize("token_count", [0, -1])
def test_advance_requires_positive_token_count(token_count: int) -> None:
    cache = _build_cache()

    with pytest.raises(ValueError, match="token_count must be positive"):
        cache.advance(token_count)

    assert cache.position == 0


@pytest.mark.parametrize("token_count", [True, 1.5])
def test_advance_requires_integer_token_count(token_count: object) -> None:
    cache = _build_cache()

    with pytest.raises(TypeError, match="token_count must be an integer"):
        cache.advance(token_count)  # type: ignore[arg-type]

    assert cache.position == 0
