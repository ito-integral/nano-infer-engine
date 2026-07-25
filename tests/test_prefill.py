from unittest.mock import patch

import torch

from nano_infer_engine.cache import KVCache
from nano_infer_engine.generation.prefill import prefill
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache


def _build_tiny_model() -> Llama3_2:
    torch.manual_seed(0)
    return Llama3_2(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            mlp_inner_size=32,
            num_layers=2,
            q_head_num=4,
            kv_head_num=2,
            rope_type="llama3",
            max_seq_len=16,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()


def _build_cache(model: Llama3_2, input_ids: torch.Tensor) -> KVCache:
    return KVCache(
        num_layers=len(model.decoders),
        batch_size=input_ids.shape[0],
        capacity=input_ids.shape[1],
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=input_ids.device,
    )


def test_one_shot_prefill_matches_no_cache_logits() -> None:
    model = _build_tiny_model()
    input_ids = torch.tensor([[1, 4, 7, 9], [1, 5, 8, 11]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    kv_cache = _build_cache(model, input_ids)

    with torch.inference_mode():
        expected = model(input_ids, attention_mask=attention_mask)
        actual = prefill(model, input_ids, attention_mask, kv_cache)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        model.config.vocab_size,
    )
    assert kv_cache.position == input_ids.shape[1]


def test_paged_cache_matches_contiguous_cache_for_prefill_and_decode() -> None:
    model = _build_tiny_model()
    prefill_ids = torch.tensor([[1, 4, 7, 9]])
    decode_ids = torch.tensor([[11]])
    capacity = prefill_ids.shape[1] + decode_ids.shape[1]

    contiguous_cache = KVCache(
        num_layers=len(model.decoders),
        batch_size=1,
        capacity=capacity,
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=prefill_ids.device,
    )
    paged_cache = PagedKVCache(
        num_blocks=3,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=prefill_ids.device,
    )

    with torch.inference_mode():
        contiguous_prefill = model(prefill_ids, kv_cache=contiguous_cache)
        paged_prefill = model(prefill_ids, kv_cache=paged_cache)
        contiguous_decode = model(decode_ids, kv_cache=contiguous_cache)
        with patch.object(
            paged_cache,
            "gather",
            side_effect=AssertionError("decode must not gather paged K/V"),
        ):
            paged_decode = model(decode_ids, kv_cache=paged_cache)

    torch.testing.assert_close(paged_prefill, contiguous_prefill)
    torch.testing.assert_close(paged_decode, contiguous_decode)
    assert paged_cache.get_sequence_length("default") == capacity

    for layer_index in range(len(model.decoders)):
        paged_keys, paged_values = paged_cache.gather(
            layer_index,
            "default",
            capacity,
        )
        contiguous_keys, contiguous_values = contiguous_cache.get(layer_index)
        torch.testing.assert_close(paged_keys, contiguous_keys[0, :capacity])
        torch.testing.assert_close(paged_values, contiguous_values[0, :capacity])


def test_paged_cache_supports_equal_length_batch_inference() -> None:
    model = _build_tiny_model()
    prefill_ids = torch.tensor(
        [
            [1, 4, 7],
            [1, 5, 8],
        ]
    )
    decode_ids = torch.tensor([[9], [11]])
    sequence_ids = ("request-a", "request-b")
    capacity = prefill_ids.shape[1] + decode_ids.shape[1]

    contiguous_cache = KVCache(
        num_layers=len(model.decoders),
        batch_size=2,
        capacity=capacity,
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=prefill_ids.device,
    )
    paged_cache = PagedKVCache(
        num_blocks=4,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=prefill_ids.device,
    )

    with torch.inference_mode():
        contiguous_prefill = model(prefill_ids, kv_cache=contiguous_cache)
        paged_prefill = model(
            prefill_ids,
            kv_cache=paged_cache,
            sequence_ids=sequence_ids,
        )
        contiguous_decode = model(decode_ids, kv_cache=contiguous_cache)
        paged_decode = model(
            decode_ids,
            kv_cache=paged_cache,
            sequence_ids=sequence_ids,
        )

    torch.testing.assert_close(paged_prefill, contiguous_prefill)
    torch.testing.assert_close(paged_decode, contiguous_decode)

    for batch_index, sequence_id in enumerate(sequence_ids):
        assert paged_cache.get_sequence_length(sequence_id) == capacity
        for layer_index in range(len(model.decoders)):
            paged_keys, paged_values = paged_cache.gather(
                layer_index,
                sequence_id,
                capacity,
            )
            contiguous_keys, contiguous_values = contiguous_cache.get(layer_index)
            torch.testing.assert_close(
                paged_keys,
                contiguous_keys[batch_index, :capacity],
            )
            torch.testing.assert_close(
                paged_values,
                contiguous_values[batch_index, :capacity],
            )


def test_paged_cache_supports_unequal_length_batched_decode() -> None:
    model = _build_tiny_model()
    prompts = (
        torch.tensor([[1, 4]]),
        torch.tensor([[1, 5, 8, 11]]),
    )
    decode_ids = torch.tensor([[9], [12]])
    sequence_ids = ("request-a", "request-b")
    paged_cache = PagedKVCache(
        num_blocks=5,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=decode_ids.device,
    )

    with torch.inference_mode():
        for prompt, sequence_id in zip(prompts, sequence_ids):
            model(
                prompt,
                kv_cache=paged_cache,
                sequence_id=sequence_id,
            )

        actual = model(
            decode_ids,
            kv_cache=paged_cache,
            sequence_ids=sequence_ids,
        )
        expected = torch.cat(
            [
                model(torch.cat((prompt, decode_ids[index : index + 1]), dim=1))[:, -1:]
                for index, prompt in enumerate(prompts)
            ],
            dim=0,
        )

    torch.testing.assert_close(actual, expected)
    assert paged_cache.get_sequence_length("request-a") == 3
    assert paged_cache.get_sequence_length("request-b") == 5


def test_paged_cache_supports_multi_step_unequal_length_batched_decode() -> None:
    model = _build_tiny_model()
    sequences = [
        torch.tensor([[1, 4]]),
        torch.tensor([[1, 5, 8, 11]]),
    ]
    sequence_ids = ("request-a", "request-b")
    decode_steps = 4
    paged_cache = PagedKVCache(
        num_blocks=7,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=sequences[0].device,
    )

    with torch.inference_mode():
        prefill_logits = [
            model(
                sequence,
                kv_cache=paged_cache,
                sequence_id=sequence_id,
            )
            for sequence, sequence_id in zip(sequences, sequence_ids)
        ]
        next_tokens = torch.cat(
            [logits[:, -1].argmax(dim=-1, keepdim=True) for logits in prefill_logits],
            dim=0,
        )

        for _ in range(decode_steps):
            actual = model(
                next_tokens,
                kv_cache=paged_cache,
                sequence_ids=sequence_ids,
            )
            sequences = [
                torch.cat((sequence, next_tokens[index : index + 1]), dim=1)
                for index, sequence in enumerate(sequences)
            ]
            expected = torch.cat(
                [model(sequence)[:, -1:] for sequence in sequences],
                dim=0,
            )
            torch.testing.assert_close(actual, expected)
            next_tokens = actual[:, -1].argmax(dim=-1, keepdim=True)

    assert paged_cache.get_sequence_length("request-a") == 6
    assert paged_cache.get_sequence_length("request-b") == 8


def test_one_shot_prefill_supports_left_padded_batch() -> None:
    model = _build_tiny_model()
    input_ids = torch.tensor([[0, 0, 1, 4], [1, 5, 8, 11]])
    attention_mask = torch.tensor(
        [[0, 0, 1, 1], [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    kv_cache = _build_cache(model, input_ids)

    with torch.inference_mode():
        actual = prefill(model, input_ids, attention_mask, kv_cache)
        expected_last_logits = torch.cat(
            [
                model(input_ids[index, mask][None, :])[:, -1]
                for index, mask in enumerate(attention_mask)
            ],
            dim=0,
        )

    torch.testing.assert_close(
        actual[:, -1],
        expected_last_logits,
        rtol=1e-5,
        atol=1e-5,
    )
    assert kv_cache.position == input_ids.shape[1]
