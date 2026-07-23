from nano_infer_engine.block_allocator import BlockAllocator


class PagedKVCache:
    """Manage per-sequence block tables for a paged KV cache.

    This first step manages metadata only. Actual K/V tensors and attention
    integration will be added after block-table behavior is implemented and
    tested.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise TypeError("block_size must be an integer")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

        # request ID -> physical block IDs in logical-block order
        # Example: {"request-a": [3, 7, 1]}
        self._block_tables: dict[str, list[int]] = {}

    def ensure_capacity(self, sequence_id: str, required_tokens: int) -> None:
        """Ensure that a sequence owns enough blocks for required_tokens."""
        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")
        if not isinstance(required_tokens, int) or isinstance(required_tokens, bool):
            raise TypeError("required_tokens must be an integer")
        if required_tokens <= 0:
            raise ValueError("required_tokens must be positive")

        required_blocks = (
            required_tokens + self.block_size - 1
        ) // self.block_size
        block_table = self._block_tables.get(sequence_id)
        current_blocks = 0 if block_table is None else len(block_table)
        missing_blocks = required_blocks - current_blocks

        if missing_blocks <= 0:
            return

        # allocate() validates capacity before changing allocator state. Do not
        # create the dictionary entry until allocation succeeds either.
        new_block_ids = self.allocator.allocate(missing_blocks)
        if block_table is None:
            self._block_tables[sequence_id] = new_block_ids
        else:
            block_table.extend(new_block_ids)

    def get_block_table(self, sequence_id: str) -> tuple[int, ...]:
        """Return a read-only snapshot of a sequence's block table."""
        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")

        return tuple(self._block_tables[sequence_id])

    def release(self, sequence_id: str) -> None:
        """Release every physical block owned by a sequence."""
        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")

        block_ids = self._block_tables[sequence_id]
        self.allocator.free(block_ids)
        del self._block_tables[sequence_id]
