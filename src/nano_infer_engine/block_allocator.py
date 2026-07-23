from collections.abc import Iterable


class BlockAllocator:
    """Allocate and recycle integer IDs for preallocated KV cache blocks."""

    def __init__(self, num_blocks: int) -> None:
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool):
            raise TypeError("num_blocks must be an integer")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.num_blocks = num_blocks
        # Reverse the initial list so pop() returns 0, 1, 2, ... cheaply.
        self._free_blocks = list(range(num_blocks - 1, -1, -1))
        self._allocated_blocks: set[int] = set()

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def allocated_block_count(self) -> int:
        return len(self._allocated_blocks)

    def allocate(self, block_count: int = 1) -> list[int]:
        """Return physical block IDs, without requiring them to be contiguous."""
        if not isinstance(block_count, int) or isinstance(block_count, bool):
            raise TypeError("block_count must be an integer")
        if block_count <= 0:
            raise ValueError("block_count must be positive")
        if block_count > self.free_block_count:
            raise ValueError("not enough free blocks")

        block_ids = [self._free_blocks.pop() for _ in range(block_count)]
        self._allocated_blocks.update(block_ids)
        return block_ids

    def free(self, block_ids: Iterable[int]) -> None:
        """Return previously allocated physical block IDs to the free list."""
        block_ids = list(block_ids)

        if any(
            not isinstance(block_id, int) or isinstance(block_id, bool)
            for block_id in block_ids
        ):
            raise TypeError("block IDs must be integers")
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("cannot free the same block more than once")

        for block_id in block_ids:
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(f"block ID out of range: {block_id}")
            if block_id not in self._allocated_blocks:
                raise ValueError(f"block is not allocated: {block_id}")

        # Validate every ID before mutating state, so a failed free is atomic.
        for block_id in block_ids:
            self._allocated_blocks.remove(block_id)
        self._free_blocks.extend(reversed(block_ids))
