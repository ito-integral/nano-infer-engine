import pytest
import torch

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.greedy import greedy_generate
from nano_infer_engine.generation.paged_greedy import paged_greedy_generate
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache


def test_paged_greedy_matches_individual_generation_for_ragged_prompts() -> None:
    torch.manual_seed(0)
    model = Llama3_2(
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
    prompts = (
        torch.tensor([[1, 4]]),
        torch.tensor([[1, 5, 8, 11]]),
    )
    sequence_ids = ("request-a", "request-b")
    config = GenerationConfig(
        max_new_tokens=4,
        eos_token_id=None,
        use_cache=True,
    )
    paged_cache = PagedKVCache(
        num_blocks=7,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device="cpu",
    )

    expected_sequences = tuple(
        greedy_generate(model, prompt, config).sequences for prompt in prompts
    )
    actual = paged_greedy_generate(
        model,
        prompts,
        config,
        paged_cache,
        sequence_ids,
    )

    for actual_sequence, expected_sequence in zip(
        actual.sequences,
        expected_sequences,
    ):
        assert torch.equal(actual_sequence, expected_sequence)

    assert torch.equal(actual.generated_tokens, torch.tensor([4, 4]))
    assert not actual.stopped_by_eos.any()
    assert paged_cache.get_sequence_length("request-a") == 5
    assert paged_cache.get_sequence_length("request-b") == 7


def test_paged_greedy_checks_capacity_before_prefill() -> None:
    torch.manual_seed(0)
    model = Llama3_2(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            mlp_inner_size=16,
            num_layers=1,
            q_head_num=2,
            kv_head_num=1,
            rope_type="default",
            max_seq_len=16,
            tie_word_embeddings=False,
        )
    ).eval()
    cache = PagedKVCache(
        num_blocks=2,
        block_size=2,
        num_layers=1,
        kv_head_num=1,
        head_dim=4,
        dtype=model.embed.weight.dtype,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="not enough free blocks for paged generation",
    ):
        paged_greedy_generate(
            model,
            (torch.tensor([[1, 2]]), torch.tensor([[3, 4]])),
            GenerationConfig(max_new_tokens=2),
            cache,
            ("request-a", "request-b"),
        )

    assert cache.allocator.allocated_block_count == 0
