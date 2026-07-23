import torch

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.greedy import greedy_generate
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig


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


def test_left_padded_batch_matches_individual_generation() -> None:
    model = _build_tiny_model()
    input_ids = torch.tensor(
        [
            [0, 0, 1, 4],
            [1, 5, 8, 11],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    cached_config = GenerationConfig(
        max_new_tokens=4,
        eos_token_id=None,
        use_cache=True,
    )
    no_cache_config = GenerationConfig(
        max_new_tokens=4,
        eos_token_id=None,
        use_cache=False,
    )

    cached_batch = greedy_generate(
        model,
        input_ids,
        cached_config,
        attention_mask=attention_mask,
    )
    no_cache_batch = greedy_generate(
        model,
        input_ids,
        no_cache_config,
        attention_mask=attention_mask,
    )
    individual_generated = torch.cat(
        [
            greedy_generate(
                model,
                input_ids[index, mask][None, :],
                cached_config,
            ).sequences[:, -cached_config.max_new_tokens :]
            for index, mask in enumerate(attention_mask)
        ],
        dim=0,
    )

    assert torch.equal(cached_batch.sequences, no_cache_batch.sequences)
    assert torch.equal(
        cached_batch.sequences[:, -cached_config.max_new_tokens :],
        individual_generated,
    )
    assert torch.equal(cached_batch.generated_tokens, torch.tensor([4, 4]))


def test_variable_length_batch_rejects_right_padding() -> None:
    model = _build_tiny_model()
    input_ids = torch.tensor([[1, 4, 0], [1, 5, 8]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    try:
        greedy_generate(
            model,
            input_ids,
            GenerationConfig(max_new_tokens=1),
            attention_mask=attention_mask,
        )
    except ValueError as error:
        assert "left padding" in str(error)
    else:
        raise AssertionError("right padding should be rejected")
