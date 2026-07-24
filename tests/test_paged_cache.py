import pytest
import torch

from nano_infer_engine.cache import KVCache
from nano_infer_engine.paged_cache import PagedKVCache


def _build_cache(**overrides: object) -> PagedKVCache:
    arguments = {
        "num_blocks": 4,
        "block_size": 4,
        "num_layers": 2,
        "kv_head_num": 2,
        "head_dim": 4,
        "dtype": torch.float32,
        "device": "cpu",
    }
    arguments.update(overrides)
    return PagedKVCache(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("required_tokens", "expected_blocks"),
    [(1, 1), (4, 1), (5, 2), (8, 2), (9, 3)],
)
def test_ensure_capacity_allocates_required_blocks(
    required_tokens: int,
    expected_blocks: int,
) -> None:
    cache = _build_cache(num_blocks=4)

    cache.ensure_capacity("request-a", required_tokens)

    assert len(cache.get_block_table("request-a")) == expected_blocks
    assert cache.allocator.allocated_block_count == expected_blocks


def test_ensure_capacity_only_allocates_missing_blocks() -> None:
    cache = _build_cache(num_blocks=4)
    cache.ensure_capacity("request-a", required_tokens=2)
    first_block_table = cache.get_block_table("request-a")

    cache.ensure_capacity("request-a", required_tokens=4)
    assert cache.get_block_table("request-a") == first_block_table

    cache.ensure_capacity("request-a", required_tokens=5)
    assert cache.get_block_table("request-a")[:1] == first_block_table
    assert len(cache.get_block_table("request-a")) == 2


def test_ensure_capacity_keeps_sequence_block_tables_separate() -> None:
    cache = _build_cache(num_blocks=4)

    cache.ensure_capacity("request-a", required_tokens=5)
    cache.ensure_capacity("request-b", required_tokens=5)

    blocks_a = set(cache.get_block_table("request-a"))
    blocks_b = set(cache.get_block_table("request-b"))
    assert blocks_a.isdisjoint(blocks_b)


def test_ensure_capacity_failure_does_not_create_partial_block_table() -> None:
    cache = _build_cache(num_blocks=2)
    cache.ensure_capacity("request-a", required_tokens=4)
    existing_table = cache.get_block_table("request-a")

    with pytest.raises(ValueError, match="not enough free blocks"):
        cache.ensure_capacity("request-b", required_tokens=8)

    with pytest.raises(KeyError):
        cache.get_block_table("request-b")
    assert cache.get_block_table("request-a") == existing_table
    assert cache.allocator.free_block_count == 1


@pytest.mark.parametrize("required_tokens", [0, -1])
def test_required_tokens_must_be_positive(required_tokens: int) -> None:
    cache = _build_cache(num_blocks=2)

    with pytest.raises(ValueError, match="required_tokens must be positive"):
        cache.ensure_capacity("request-a", required_tokens)


@pytest.mark.parametrize("required_tokens", [True, 1.5])
def test_required_tokens_must_be_an_integer(required_tokens: object) -> None:
    cache = _build_cache(num_blocks=2)

    with pytest.raises(TypeError, match="required_tokens must be an integer"):
        cache.ensure_capacity("request-a", required_tokens)  # type: ignore[arg-type]


def test_get_block_table_returns_an_immutable_snapshot() -> None:
    cache = _build_cache(num_blocks=3)
    cache.ensure_capacity("request-a", required_tokens=4)

    snapshot = cache.get_block_table("request-a")
    cache.ensure_capacity("request-a", required_tokens=5)

    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 1
    assert len(cache.get_block_table("request-a")) == 2


def test_get_block_table_rejects_unknown_sequence() -> None:
    cache = _build_cache(num_blocks=2)

    with pytest.raises(KeyError, match="request-a"):
        cache.get_block_table("request-a")


@pytest.mark.parametrize("sequence_id", [None, 1])
def test_get_block_table_requires_string_sequence_id(sequence_id: object) -> None:
    cache = _build_cache(num_blocks=2)

    with pytest.raises(TypeError, match="sequence_id must be a string"):
        cache.get_block_table(sequence_id)  # type: ignore[arg-type]


def test_get_block_table_rejects_empty_sequence_id() -> None:
    cache = _build_cache(num_blocks=2)

    with pytest.raises(ValueError, match="sequence_id must not be empty"):
        cache.get_block_table("")


def test_release_returns_all_sequence_blocks() -> None:
    cache = _build_cache(num_blocks=3)
    cache.ensure_capacity("request-a", required_tokens=5)
    released_blocks = cache.get_block_table("request-a")

    cache.release("request-a")

    assert cache.allocator.free_block_count == 3
    assert cache.allocator.allocated_block_count == 0
    with pytest.raises(KeyError, match="request-a"):
        cache.get_block_table("request-a")
    assert cache.allocator.allocate(2) == list(released_blocks)


def test_release_only_returns_requested_sequence_blocks() -> None:
    cache = _build_cache(num_blocks=4)
    cache.ensure_capacity("request-a", required_tokens=5)
    cache.ensure_capacity("request-b", required_tokens=5)
    blocks_b = cache.get_block_table("request-b")

    cache.release("request-a")

    assert cache.get_block_table("request-b") == blocks_b
    assert cache.allocator.allocated_block_count == 2


def test_release_keeps_block_table_when_allocator_free_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _build_cache(num_blocks=2)
    cache.ensure_capacity("request-a", required_tokens=4)
    existing_table = cache.get_block_table("request-a")

    def fail_to_free(block_ids: object) -> None:
        raise RuntimeError("simulated allocator failure")

    monkeypatch.setattr(cache.allocator, "free", fail_to_free)

    with pytest.raises(RuntimeError, match="simulated allocator failure"):
        cache.release("request-a")

    assert cache.get_block_table("request-a") == existing_table


def test_release_rejects_unknown_or_already_released_sequence() -> None:
    cache = _build_cache(num_blocks=2)

    with pytest.raises(KeyError, match="request-a"):
        cache.release("request-a")

    cache.ensure_capacity("request-a", required_tokens=1)
    cache.release("request-a")
    with pytest.raises(KeyError, match="request-a"):
        cache.release("request-a")


def test_gather_matches_contiguous_kv_cache() -> None:
    token_count = 6
    layer_index = 1
    keys = torch.arange(token_count * 2 * 4, dtype=torch.float32).reshape(
        token_count,
        2,
        4,
    )
    values = keys + 100

    contiguous_cache = KVCache(
        num_layers=2,
        batch_size=1,
        capacity=token_count,
        kv_head_num=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    contiguous_keys, contiguous_values = contiguous_cache.get(layer_index)
    contiguous_keys[0, :token_count].copy_(keys)
    contiguous_values[0, :token_count].copy_(values)
    contiguous_cache.advance(token_count)

    paged_cache = _build_cache(num_blocks=6)

    # Create a non-contiguous physical block table for the target sequence.
    paged_cache.ensure_capacity("temporary-a", 1)
    paged_cache.ensure_capacity("temporary-b", 1)
    paged_cache.ensure_capacity("temporary-c", 1)
    paged_cache.release("temporary-a")
    paged_cache.release("temporary-c")

    paged_cache.ensure_capacity("request-a", token_count)
    assert paged_cache.get_block_table("request-a") == (2, 0)
    paged_cache.write(layer_index, "request-a", 0, keys, values)
    gathered_keys, gathered_values = paged_cache.gather(
        layer_index,
        "request-a",
        token_count,
    )

    torch.testing.assert_close(gathered_keys, contiguous_keys[0, :token_count])
    torch.testing.assert_close(gathered_values, contiguous_values[0, :token_count])
