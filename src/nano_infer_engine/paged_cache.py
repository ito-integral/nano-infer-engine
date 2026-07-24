import torch

from nano_infer_engine.block_allocator import BlockAllocator


class PagedKVCache:
    """Manage physical K/V block storage and per-sequence block tables."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        kv_head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise TypeError("block_size must be an integer")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool):
            raise TypeError("num_blocks must be an integer")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if not isinstance(num_layers, int) or isinstance(num_layers, bool):
            raise TypeError("num_layers must be an integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if not isinstance(kv_head_num, int) or isinstance(kv_head_num, bool):
            raise TypeError("kv_head_num must be an integer")
        if kv_head_num <= 0:
            raise ValueError("kv_head_num must be positive")
        if not isinstance(head_dim, int) or isinstance(head_dim, bool):
            raise TypeError("head_dim must be an integer")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

        self.num_layers = num_layers
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim

        # request ID -> physical block IDs in logical-block order
        # Example: {"request-a": [3, 7, 1]}
        self._block_tables: dict[str, list[int]] = {}
        self._sequence_lengths: dict[str, int] = {}

        storage_shape = (
            num_layers,
            num_blocks,
            block_size,
            kv_head_num,
            head_dim,
        )
        self.keys = torch.empty(storage_shape, dtype=dtype, device=device)
        self.values = torch.empty(storage_shape, dtype=dtype, device=device)

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

        required_blocks = (required_tokens + self.block_size - 1) // self.block_size
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
            self._sequence_lengths[sequence_id] = 0
        else:
            block_table.extend(new_block_ids)

    def get_sequence_length(self, sequence_id: str) -> int:
        """Return the number of cached tokens for a sequence."""
        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")

        return self._sequence_lengths.get(sequence_id, 0)

    def advance(self, sequence_id: str, token_count: int) -> None:
        """Advance a sequence after every layer has written its new K/V tokens."""
        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an integer")
        if token_count <= 0:
            raise ValueError("token_count must be positive")

        current_length = self._sequence_lengths[sequence_id]
        new_length = current_length + token_count

        # Capacity must be allocated before the sequence length is advanced.
        self._resolve_position(sequence_id, new_length - 1)
        self._sequence_lengths[sequence_id] = new_length

    def get_block_table(self, sequence_id: str) -> tuple[int, ...]:
        """Return a read-only snapshot of a sequence's block table."""
        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")

        return tuple(self._block_tables[sequence_id])

    def _resolve_position(
        self,
        sequence_id: str,
        token_position: int,
    ) -> tuple[int, int]:
        """先调用ensure_capacity()确保有足够的块，然后调用这个函数来获取物理块和偏移量"""
        if not isinstance(token_position, int) or isinstance(token_position, bool):
            raise TypeError("token_position must be an integer")
        if token_position < 0:
            raise ValueError("token_position must be non-negative")

        block_table = self.get_block_table(sequence_id)

        # Map the token position to its logical block and in-block offset.
        logical_block = token_position // self.block_size
        block_offset = token_position % self.block_size

        if logical_block >= len(block_table):
            raise IndexError("token position exceeds allocated capacity")

        # Translate the logical block through the sequence's block table.
        physical_block = block_table[logical_block]
        return physical_block, block_offset

    def write(
        self,
        layer_index: int,
        sequence_id: str,
        start_position: int,
        keys,
        values,
    ) -> None:
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("layer_index must be an integer")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError("layer_index out of range")

        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")

        if not isinstance(start_position, int) or isinstance(start_position, bool):
            raise TypeError("start_position must be an integer")
        if start_position < 0:
            raise ValueError("start_position must be non-negative")

        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("keys and values must be torch.Tensor objects")
        if keys.shape != values.shape:
            raise ValueError("keys and values must have identical shapes")
        if keys.ndim != 3:
            raise ValueError("keys and values must be 3D tensors")

        token_count, kv_head_num, head_dim = keys.shape
        if token_count <= 0:
            raise ValueError("keys and values must contain at least one token")
        if kv_head_num != self.kv_head_num or head_dim != self.head_dim:
            raise ValueError(
                "keys and values must match the cache KV head count and head dimension"
            )
        if keys.dtype != self.keys.dtype or values.dtype != self.values.dtype:
            raise ValueError("keys and values must match the cache dtype")
        if keys.device != self.keys.device or values.device != self.values.device:
            raise ValueError("keys and values must be on the cache device")

        # 写入前检查最后一个 token 是否已有对应的物理 block。
        end_position = start_position + token_count
        self._resolve_position(sequence_id, end_position - 1)

        # 逐 token 解析物理位置并写入预分配的 K/V 存储。
        for token_offset in range(token_count):
            token_position = start_position + token_offset
            physical_block, block_offset = self._resolve_position(
                sequence_id,
                token_position,
            )
            self.keys[layer_index, physical_block, block_offset].copy_(
                keys[token_offset]
            )
            self.values[layer_index, physical_block, block_offset].copy_(
                values[token_offset]
            )

    def gather(
        self,
        layer_index: int,
        sequence_id: str,
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged K/V tokens into contiguous tensors."""
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("layer_index must be an integer")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError("layer_index out of range")

        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")

        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an integer")
        if token_count <= 0:
            raise ValueError("token_count must be positive")

        # 读取前检查完整 token 范围是否已分配物理 block。
        self._resolve_position(sequence_id, token_count - 1)

        key_tokens = []
        value_tokens = []

        # 按逻辑 token 顺序从离散的物理 block 中收集 K/V。
        for token_position in range(token_count):
            physical_block, block_offset = self._resolve_position(
                sequence_id,
                token_position,
            )
            key_tokens.append(self.keys[layer_index, physical_block, block_offset])
            value_tokens.append(self.values[layer_index, physical_block, block_offset])
        return torch.stack(key_tokens), torch.stack(
            value_tokens
        )  # (token_count, kv_head_num, head_dim)

    def release(self, sequence_id: str) -> None:
        """Release every physical block owned by a sequence."""
        if not isinstance(sequence_id, str):
            raise TypeError("sequence_id must be a string")
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")

        block_ids = self._block_tables[sequence_id]
        self.allocator.free(block_ids)
        del self._block_tables[sequence_id]
        del self._sequence_lengths[sequence_id]
