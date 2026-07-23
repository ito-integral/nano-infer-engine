import pytest

from nano_infer_engine.block_allocator import BlockAllocator


def test_allocates_and_tracks_blocks() -> None:
    allocator = BlockAllocator(num_blocks=4)

    assert allocator.free_block_count == 4
    assert allocator.allocated_block_count == 0

    assert allocator.allocate(2) == [0, 1]
    assert allocator.free_block_count == 2
    assert allocator.allocated_block_count == 2


def test_reuses_freed_blocks() -> None:
    allocator = BlockAllocator(num_blocks=4)
    allocated = allocator.allocate(4)

    allocator.free([allocated[1], allocated[3]])

    assert allocator.allocate(2) == [allocated[1], allocated[3]]


def test_allocation_failure_does_not_consume_blocks() -> None:
    allocator = BlockAllocator(num_blocks=2)

    with pytest.raises(ValueError, match="not enough free blocks"):
        allocator.allocate(3)

    assert allocator.free_block_count == 2
    assert allocator.allocated_block_count == 0


def test_rejects_duplicate_free_without_changing_state() -> None:
    allocator = BlockAllocator(num_blocks=2)
    block_id = allocator.allocate()[0]

    with pytest.raises(ValueError, match="same block more than once"):
        allocator.free([block_id, block_id])

    assert allocator.free_block_count == 1
    assert allocator.allocated_block_count == 1


def test_rejects_freeing_unallocated_block() -> None:
    allocator = BlockAllocator(num_blocks=2)

    with pytest.raises(ValueError, match="block is not allocated"):
        allocator.free([0])


def test_rejects_double_free() -> None:
    allocator = BlockAllocator(num_blocks=2)
    block_id = allocator.allocate()[0]
    allocator.free([block_id])

    with pytest.raises(ValueError, match="block is not allocated"):
        allocator.free([block_id])


def test_failed_multi_block_free_is_atomic() -> None:
    allocator = BlockAllocator(num_blocks=3)
    block_id = allocator.allocate()[0]

    with pytest.raises(ValueError, match="block is not allocated"):
        allocator.free([block_id, 1])

    assert allocator.free_block_count == 2
    assert allocator.allocated_block_count == 1


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_num_blocks_must_be_positive(num_blocks: int) -> None:
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        BlockAllocator(num_blocks)


@pytest.mark.parametrize("num_blocks", [True, 1.5])
def test_num_blocks_must_be_an_integer(num_blocks: object) -> None:
    with pytest.raises(TypeError, match="num_blocks must be an integer"):
        BlockAllocator(num_blocks)  # type: ignore[arg-type]


@pytest.mark.parametrize("block_count", [0, -1])
def test_block_count_must_be_positive(block_count: int) -> None:
    allocator = BlockAllocator(2)

    with pytest.raises(ValueError, match="block_count must be positive"):
        allocator.allocate(block_count)


@pytest.mark.parametrize("block_count", [True, 1.5])
def test_block_count_must_be_an_integer(block_count: object) -> None:
    allocator = BlockAllocator(2)

    with pytest.raises(TypeError, match="block_count must be an integer"):
        allocator.allocate(block_count)  # type: ignore[arg-type]


@pytest.mark.parametrize("block_id", [-1, 2])
def test_free_rejects_out_of_range_block_id(block_id: int) -> None:
    allocator = BlockAllocator(2)

    with pytest.raises(ValueError, match="block ID out of range"):
        allocator.free([block_id])


@pytest.mark.parametrize("block_id", [True, 1.5])
def test_free_requires_integer_block_ids(block_id: object) -> None:
    allocator = BlockAllocator(2)

    with pytest.raises(TypeError, match="block IDs must be integers"):
        allocator.free([block_id])  # type: ignore[list-item]
