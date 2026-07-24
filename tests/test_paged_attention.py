import torch

from nano_infer_engine.layers.paged_attention import paged_attention_reference


def test_paged_attention_matches_contiguous_attention() -> None:
    torch.manual_seed(0)

    sequence_length = 5
    block_size = 2
    q_head_num = 4
    kv_head_num = 2
    head_dim = 3
    layer_index = 1
    block_table = (3, 0, 2)

    query = torch.randn(q_head_num, head_dim)
    contiguous_keys = torch.randn(sequence_length, kv_head_num, head_dim)
    contiguous_values = torch.randn(sequence_length, kv_head_num, head_dim)

    key_cache = torch.randn(2, 4, block_size, kv_head_num, head_dim)
    value_cache = torch.randn_like(key_cache)
    for logical_block, physical_block in enumerate(block_table):
        block_start = logical_block * block_size
        block_end = min(block_start + block_size, sequence_length)
        valid_token_count = block_end - block_start
        key_cache[layer_index, physical_block, :valid_token_count].copy_(
            contiguous_keys[block_start:block_end]
        )
        value_cache[layer_index, physical_block, :valid_token_count].copy_(
            contiguous_values[block_start:block_end]
        )

    group_num = q_head_num // kv_head_num
    expanded_keys = contiguous_keys.repeat_interleave(group_num, dim=1)
    expanded_values = contiguous_values.repeat_interleave(group_num, dim=1)
    expected_scores = torch.einsum("hd,thd->ht", query, expanded_keys)
    expected_scores = expected_scores * (head_dim**-0.5)
    expected_weights = torch.softmax(expected_scores, dim=-1, dtype=torch.float32)
    expected = torch.einsum("ht,thd->hd", expected_weights, expanded_values)

    actual = paged_attention_reference(
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_length,
        layer_index,
    )

    torch.testing.assert_close(actual, expected)
