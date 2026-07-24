import torch

from nano_infer_engine.cache import KVCache
from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.greedy import greedy_generate
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache


def test_paged_greedy_matches_contiguous_cache_across_blocks() -> None:
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
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()
    input_ids = torch.tensor([[1, 4, 7]])
    config = GenerationConfig(
        max_new_tokens=5,
        eos_token_id=None,
        use_cache=True,
    )
    capacity = input_ids.shape[1] + config.max_new_tokens

    contiguous_cache = KVCache(
        num_layers=len(model.decoders),
        batch_size=1,
        capacity=capacity,
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=input_ids.device,
    )
    paged_cache = PagedKVCache(
        num_blocks=4,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=input_ids.device,
    )

    contiguous_output = greedy_generate(
        model,
        input_ids,
        config,
        kv_cache=contiguous_cache,
    )
    paged_output = greedy_generate(
        model,
        input_ids,
        config,
        kv_cache=paged_cache,
    )

    assert torch.equal(paged_output.sequences, contiguous_output.sequences)
    assert torch.equal(
        paged_output.generated_tokens,
        contiguous_output.generated_tokens,
    )

    # The final generated token is not cached until another decode step needs it.
    cached_token_count = capacity - 1
    assert contiguous_cache.position == cached_token_count
    assert paged_cache.get_sequence_length("default") == cached_token_count
    assert len(paged_cache.get_block_table("default")) == 4

    for layer_index in range(len(model.decoders)):
        paged_keys, paged_values = paged_cache.gather(
            layer_index,
            "default",
            cached_token_count,
        )
        contiguous_keys, contiguous_values = contiguous_cache.get(layer_index)
        torch.testing.assert_close(
            paged_keys,
            contiguous_keys[0, :cached_token_count],
        )
        torch.testing.assert_close(
            paged_values,
            contiguous_values[0, :cached_token_count],
        )
