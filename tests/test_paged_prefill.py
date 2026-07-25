import pytest
import torch

from nano_infer_engine.generation.paged_prefill import paged_prefill
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
