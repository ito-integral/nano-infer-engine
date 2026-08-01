import pytest
import torch

from nano_infer_engine.generation.paged_prefill import (
    paged_prefill,
    paged_prefill_chunks,
)
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache


def _build_model() -> Llama3_2:
    torch.manual_seed(0)
    return Llama3_2(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            mlp_inner_size=32,
            num_layers=2,
            q_head_num=4,
            kv_head_num=2,
            rope_type="default",
            max_seq_len=16,
            tie_word_embeddings=False,
        )
    ).eval()


def _build_cache(model: Llama3_2, num_blocks: int = 4) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device="cpu",
    )


def test_paged_prefill_matches_individual_prompt_logits() -> None:
    model = _build_model()
    cache = _build_cache(model)
    prompts = (
        torch.tensor([[1, 4]]),
        torch.tensor([[1, 5, 8, 11]]),
    )
    sequence_ids = ("request-a", "request-b")

    with torch.inference_mode():
        expected = torch.cat(
            [model(prompt)[:, -1] for prompt in prompts],
            dim=0,
        )
        actual = paged_prefill(model, prompts, cache, sequence_ids)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, model.config.vocab_size)
    assert cache.get_sequence_length("request-a") == 2
    assert cache.get_sequence_length("request-b") == 4
    assert set(cache.get_block_table("request-a")).isdisjoint(
        cache.get_block_table("request-b")
    )


def test_paged_prefill_batches_each_prompt_length_once() -> None:
    model = _build_model()
    cache = _build_cache(model, num_blocks=8)
    prompts = (
        torch.tensor([[1, 4]]),
        torch.tensor([[1, 5, 8]]),
        torch.tensor([[1, 6]]),
        torch.tensor([[1, 7, 10]]),
    )
    sequence_ids = ("request-a", "request-b", "request-c", "request-d")
    calls: list[tuple[tuple[int, int], tuple[str, ...]]] = []

    class RecordingModel:
        def __call__(
            self,
            input_ids: torch.Tensor,
            *,
            kv_cache: PagedKVCache,
            sequence_id: str = "default",
            sequence_ids: tuple[str, ...] | None = None,
        ) -> torch.Tensor:
            current_sequence_ids = sequence_ids or (sequence_id,)
            calls.append((tuple(input_ids.shape), current_sequence_ids))
            return model(
                input_ids,
                kv_cache=kv_cache,
                sequence_id=sequence_id,
                sequence_ids=sequence_ids,
            )

    with torch.inference_mode():
        expected = torch.cat(
            [model(prompt)[:, -1] for prompt in prompts],
            dim=0,
        )
        actual = paged_prefill(RecordingModel(), prompts, cache, sequence_ids)

    assert calls == [
        ((2, 2), ("request-a", "request-c")),
        ((2, 3), ("request-b", "request-d")),
    ]
    torch.testing.assert_close(actual, expected)
    assert tuple(
        cache.get_sequence_length(sequence_id) for sequence_id in sequence_ids
    ) == (2, 3, 2, 3)


def test_chunked_prefill_matches_one_shot_prefill() -> None:
    model = _build_model()
    prompt = torch.tensor([[1, 4, 7, 10, 13]])
    one_shot_cache = _build_cache(model, num_blocks=4)
    chunked_cache = _build_cache(model, num_blocks=4)

    with torch.inference_mode():
        expected = paged_prefill(
            model, (prompt,), one_shot_cache, ("one-shot",)
        )[0]
        paged_prefill_chunks(
            model, (prompt[:, :2],), chunked_cache, ("chunked",)
        )
        paged_prefill_chunks(
            model, (prompt[:, 2:4],), chunked_cache, ("chunked",)
        )
        actual = paged_prefill_chunks(
            model, (prompt[:, 4:],), chunked_cache, ("chunked",)
        )[0]

    torch.testing.assert_close(actual, expected)
    assert chunked_cache.get_sequence_length("chunked") == prompt.shape[1]
    for layer_index in range(len(model.decoders)):
        expected_keys, expected_values = one_shot_cache.gather(
            layer_index, "one-shot", prompt.shape[1]
        )
        actual_keys, actual_values = chunked_cache.gather(
            layer_index, "chunked", prompt.shape[1]
        )
        torch.testing.assert_close(actual_keys, expected_keys)
        torch.testing.assert_close(actual_values, expected_values)


def test_ragged_prefill_batches_different_tail_lengths() -> None:
    model = _build_model()
    cache = _build_cache(model, num_blocks=8)
    reference_cache = _build_cache(model, num_blocks=8)
    prompt_a = torch.tensor([[1, 2, 3, 4, 5]])
    prompt_b = torch.tensor([[6, 7, 8, 9, 10, 11]])
    sequence_ids = ("request-a", "request-b")

    paged_prefill(
        model,
        (prompt_a[:, :4], prompt_b[:, :4]),
        cache,
        sequence_ids,
    )
    paged_prefill(
        model,
        (prompt_a[:, :4], prompt_b[:, :4]),
        reference_cache,
        ("reference-a", "reference-b"),
    )
    expected = torch.cat(
        (
            model(
                prompt_a[:, 4:],
                kv_cache=reference_cache,
                sequence_id="reference-a",
            )[:, -1],
            model(
                prompt_b[:, 4:],
                kv_cache=reference_cache,
                sequence_id="reference-b",
            )[:, -1],
        ),
        dim=0,
    )
    call_count = 0
    original_forward_ragged = model.forward_ragged

    def recording_forward_ragged(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward_ragged(*args, **kwargs)

    model.forward_ragged = recording_forward_ragged  # type: ignore[method-assign]
    logits = paged_prefill_chunks(
        model,
        (prompt_a[:, 4:], prompt_b[:, 4:]),
        cache,
        sequence_ids,
    )

    assert call_count == 1
    assert logits.shape == (2, model.config.vocab_size)
    torch.testing.assert_close(logits, expected)
    assert cache.get_sequence_length("request-a") == 5
    assert cache.get_sequence_length("request-b") == 6


def test_paged_prefill_rejects_mismatched_metadata() -> None:
    model = _build_model()
    cache = _build_cache(model)

    with pytest.raises(ValueError, match="must have the same length"):
        paged_prefill(
            model,
            (torch.tensor([[1, 4]]),),
            cache,
            ("request-a", "request-b"),
        )


def test_paged_prefill_rejects_duplicate_sequence_ids() -> None:
    model = _build_model()
    cache = _build_cache(model)

    with pytest.raises(ValueError, match="sequence IDs must be unique"):
        paged_prefill(
            model,
            (torch.tensor([[1]]), torch.tensor([[2]])),
            cache,
            ("request-a", "request-a"),
        )


def test_paged_prefill_rejects_existing_sequence_before_writing() -> None:
    model = _build_model()
    cache = _build_cache(model)
    cache.ensure_capacity("request-b", 1)

    with pytest.raises(ValueError, match="sequence ID already exists: request-b"):
        paged_prefill(
            model,
            (torch.tensor([[1]]), torch.tensor([[2]])),
            cache,
            ("request-a", "request-b"),
        )

    with pytest.raises(KeyError):
        cache.get_block_table("request-a")


def test_paged_prefill_checks_total_capacity_before_writing() -> None:
    model = _build_model()
    cache = _build_cache(model, num_blocks=2)

    with pytest.raises(ValueError, match="not enough free blocks"):
        paged_prefill(
            model,
            (torch.tensor([[1, 2, 3]]), torch.tensor([[4]])),
            cache,
            ("request-a", "request-b"),
        )

    with pytest.raises(KeyError):
        cache.get_block_table("request-a")
    with pytest.raises(KeyError):
        cache.get_block_table("request-b")


def test_decode_after_kv_transfer_matches_decode_without_transfer() -> None:
    model = _build_model()
    prompt = torch.tensor([[1, 4, 7]])
    reference_cache = _build_cache(model)
    source_cache = _build_cache(model)
    destination_cache = _build_cache(model)

    # Make the destination use a different physical block mapping.
    destination_cache.ensure_capacity("temporary", 1)

    with torch.inference_mode():
        reference_prefill_logits = paged_prefill(
            model,
            (prompt,),
            reference_cache,
            ("reference",),
        )
        source_prefill_logits = paged_prefill(
            model,
            (prompt,),
            source_cache,
            ("migrated",),
        )
        next_token = reference_prefill_logits.argmax(dim=-1, keepdim=True)

        expected_decode_logits = model(
            next_token,
            kv_cache=reference_cache,
            sequence_id="reference",
        )

        transfer = source_cache.export_sequence("migrated")
        destination_cache.import_sequence("migrated", transfer)
        actual_decode_logits = model(
            next_token,
            kv_cache=destination_cache,
            sequence_id="migrated",
        )

    torch.testing.assert_close(
        source_prefill_logits,
        reference_prefill_logits,
    )
    torch.testing.assert_close(actual_decode_logits, expected_decode_logits)
    assert destination_cache.get_block_table(
        "migrated"
    ) != reference_cache.get_block_table("reference")

    expected_length = prompt.shape[1] + 1
    assert reference_cache.get_sequence_length("reference") == expected_length
    assert destination_cache.get_sequence_length("migrated") == expected_length
    for layer_index in range(len(model.decoders)):
        reference_keys, reference_values = reference_cache.gather(
            layer_index,
            "reference",
            expected_length,
        )
        migrated_keys, migrated_values = destination_cache.gather(
            layer_index,
            "migrated",
            expected_length,
        )
        torch.testing.assert_close(migrated_keys, reference_keys)
        torch.testing.assert_close(migrated_values, reference_values)
