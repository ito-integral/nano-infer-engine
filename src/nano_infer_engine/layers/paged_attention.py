import torch


def paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: tuple[int, ...],
    sequence_length: int,
    layer_index: int,
) -> torch.Tensor:
    """Compute single-token decode attention directly from paged K/V blocks.

    Shapes:
        query: (q_head_num, head_dim)
        key_cache/value_cache:
            (num_layers, num_blocks, block_size, kv_head_num, head_dim)
        output: (q_head_num, head_dim)

    The current token's K/V must already be present in the cache. Because this
    function handles a single decode token at the end of the sequence, that
    query can attend to every cached token and needs no additional causal mask.
    """
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not isinstance(key_cache, torch.Tensor) or not isinstance(
        value_cache, torch.Tensor
    ):
        raise TypeError("key_cache and value_cache must be torch.Tensor objects")
    if query.ndim != 2:
        raise ValueError("query must be a 2D tensor")
    if key_cache.ndim != 5 or value_cache.ndim != 5:
        raise ValueError("key_cache and value_cache must be 5D tensors")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have identical shapes")
    if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
        raise ValueError("query and K/V caches must have the same dtype")
    if query.device != key_cache.device or query.device != value_cache.device:
        raise ValueError("query and K/V caches must be on the same device")
    if not query.is_floating_point():
        raise ValueError("query and K/V caches must use a floating-point dtype")

    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
        raise TypeError("sequence_length must be an integer")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not isinstance(layer_index, int) or isinstance(layer_index, bool):
        raise TypeError("layer_index must be an integer")
    if not isinstance(block_table, tuple):
        raise TypeError("block_table must be a tuple")

    num_layers, num_blocks, block_size, kv_head_num, cache_head_dim = key_cache.shape
    if min(key_cache.shape) <= 0:
        raise ValueError("K/V cache dimensions must be positive")
    if layer_index < 0 or layer_index >= num_layers:
        raise IndexError("layer_index out of range")

    q_head_num, head_dim = query.shape
    if head_dim != cache_head_dim:
        raise ValueError("query and K/V caches must have the same head dimension")
    if q_head_num % kv_head_num != 0:
        raise ValueError("q_head_num must be divisible by kv_head_num")

    required_blocks = (sequence_length + block_size - 1) // block_size
    if len(block_table) < required_blocks:
        raise ValueError("block_table does not cover sequence_length")

    used_block_ids = block_table[:required_blocks]
    if any(
        not isinstance(block_id, int) or isinstance(block_id, bool)
        for block_id in used_block_ids
    ):
        raise TypeError("block IDs must be integers")
    if any(block_id < 0 or block_id >= num_blocks for block_id in used_block_ids):
        raise ValueError("block ID out of range")

    group_num = q_head_num // kv_head_num

    score_blocks: list[torch.Tensor] = []
    value_blocks: list[torch.Tensor] = []

    for logical_block, physical_block in enumerate(used_block_ids):
        block_start = logical_block * block_size
        valid_token_count = min(block_size, sequence_length - block_start)

        # Shapes: (valid_token_count, kv_head_num, head_dim).
        key_block = key_cache[layer_index, physical_block, :valid_token_count]
        value_block = value_cache[layer_index, physical_block, :valid_token_count]

        # Expand KV heads to query heads for grouped-query attention.
        # Shapes: (valid_token_count, q_head_num, head_dim).
        key_block = key_block.repeat_interleave(group_num, dim=1)
        value_block = value_block.repeat_interleave(group_num, dim=1)

        # Compare every query head with the matching key head at each token.
        # Shape: (q_head_num, valid_token_count).
        block_scores = torch.einsum("hd,thd->ht", query, key_block)
        block_scores = block_scores * (head_dim**-0.5)

        score_blocks.append(block_scores)
        value_blocks.append(value_block)

    # Restore logical token order across physical blocks.
    scores = torch.cat(score_blocks, dim=-1)  # Shape: (q_head_num, sequence_length)
    values = torch.cat(
        value_blocks, dim=0
    )  # Shape: (sequence_length, q_head_num, head_dim)

    weights = torch.softmax(
        scores, dim=-1, dtype=torch.float32
    )  # (q_head_num, sequence_length)
    weights = weights.to(values.dtype)

    return torch.einsum("ht,thd->hd", weights, values)  # Shape: (q_head_num, head_dim)


def batched_paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: tuple[tuple[int, ...], ...],
    sequence_lengths: tuple[int, ...],
    layer_index: int,
) -> torch.Tensor:
    """Compute single-token decode attention for a fixed batch of sequences.

    Expected shapes:
        query: (batch_size, q_head_num, head_dim)
        key_cache/value_cache:
            (num_layers, num_blocks, block_size, kv_head_num, head_dim)
        output: (batch_size, q_head_num, head_dim)

    Each batch row has its own block table and sequence length. This reference
    version reuses ``paged_attention_reference`` per sequence; it is a
    correctness baseline rather than a vectorized implementation.
    """
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if query.ndim != 3:
        raise ValueError("query must be a 3D tensor")

    batch_size = query.shape[0]
    if batch_size <= 0:
        raise ValueError("query batch size must be positive")

    if not isinstance(block_tables, tuple):
        raise TypeError("block_tables must be a tuple")
    if not isinstance(sequence_lengths, tuple):
        raise TypeError("sequence_lengths must be a tuple")
    if len(block_tables) != batch_size:
        raise ValueError("block_tables must match the query batch size")
    if len(sequence_lengths) != batch_size:
        raise ValueError("sequence_lengths must match the query batch size")
    if any(not isinstance(block_table, tuple) for block_table in block_tables):
        raise TypeError("each block table must be a tuple")
    if any(
        not isinstance(sequence_length, int) or isinstance(sequence_length, bool)
        for sequence_length in sequence_lengths
    ):
        raise TypeError("sequence lengths must be integers")
    if any(sequence_length <= 0 for sequence_length in sequence_lengths):
        raise ValueError("sequence lengths must be positive")

    outputs = []
    for batch_index in range(batch_size):
        output = paged_attention_reference(
            query=query[batch_index],
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_tables[batch_index],
            sequence_length=sequence_lengths[batch_index],
            layer_index=layer_index,
        )
        outputs.append(output)

    return torch.stack(outputs, dim=0)
