import torch

from nano_infer_engine.layers.attention import GroupedQueryAttention
from nano_infer_engine.layers.paged_attention import (
    batched_paged_attention_reference,
    paged_attention_reference,
)
from nano_infer_engine.paged_cache import PagedKVCache


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


def test_batched_paged_attention_matches_per_sequence_reference() -> None:
    torch.manual_seed(1)

    queries = torch.randn(3, 4, 3)
    key_cache = torch.randn(2, 7, 2, 2, 3)
    value_cache = torch.randn_like(key_cache)
    block_tables = (
        (5, 1),
        (4, 0, 6),
        (2,),
    )
    sequence_lengths = (3, 5, 2)
    layer_index = 1

    expected = torch.stack(
        [
            paged_attention_reference(
                query=queries[batch_index],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_tables[batch_index],
                sequence_length=sequence_lengths[batch_index],
                layer_index=layer_index,
            )
            for batch_index in range(queries.shape[0])
        ],
        dim=0,
    )

    actual = batched_paged_attention_reference(
        query=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        sequence_lengths=sequence_lengths,
        layer_index=layer_index,
    )

    assert actual.shape == queries.shape
    torch.testing.assert_close(actual, expected)


def test_grouped_query_attention_supports_batched_paged_decode() -> None:
    torch.manual_seed(2)

    attention = GroupedQueryAttention(
        q_head_num=4,
        kv_head_num=2,
        hidden_size=16,
    ).eval()
    sequence_ids = ("request-a", "request-b")
    sequence_lengths = (2, 4)
    historical_keys = (
        torch.randn(2, 2, 4),
        torch.randn(4, 2, 4),
    )
    historical_values = (
        torch.randn(2, 2, 4),
        torch.randn(4, 2, 4),
    )

    def build_cache(
        selected_sequence_ids: tuple[str, ...],
    ) -> PagedKVCache:
        required_blocks = sum(
            (sequence_lengths[sequence_ids.index(current_sequence_id)] + 2) // 2
            for current_sequence_id in selected_sequence_ids
        )
        cache = PagedKVCache(
            num_blocks=required_blocks,
            block_size=2,
            num_layers=1,
            kv_head_num=2,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
        )
        for current_sequence_id in selected_sequence_ids:
            source_index = sequence_ids.index(current_sequence_id)
            sequence_length = sequence_lengths[source_index]
            cache.ensure_capacity(current_sequence_id, sequence_length + 1)
            cache.write(
                0,
                current_sequence_id,
                0,
                historical_keys[source_index],
                historical_values[source_index],
            )
            cache.advance(current_sequence_id, sequence_length)
        return cache

    x = torch.randn(2, 1, 16)
    batched_cache = build_cache(sequence_ids)
    individual_caches = tuple(
        build_cache((current_sequence_id,)) for current_sequence_id in sequence_ids
    )

    with torch.inference_mode():
        actual = attention(
            x,
            paged_cache=batched_cache,
            layer_index=0,
            sequence_ids=sequence_ids,
        )
        expected = torch.cat(
            [
                attention(
                    x[batch_index : batch_index + 1],
                    paged_cache=individual_caches[batch_index],
                    layer_index=0,
                    sequence_id=current_sequence_id,
                )
                for batch_index, current_sequence_id in enumerate(sequence_ids)
            ],
            dim=0,
        )

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)
