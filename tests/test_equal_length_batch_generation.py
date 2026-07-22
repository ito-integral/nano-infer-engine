import torch

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.greedy import greedy_generate
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig


class _ScriptedModel:
    def __init__(self) -> None:
        self.step = 0

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        next_tokens = (
            torch.tensor([2, 3], device=input_ids.device)
            if self.step == 0
            else torch.tensor([4, 2], device=input_ids.device)
        )
        self.step += 1
        logits = torch.zeros((*input_ids.shape, 8), device=input_ids.device)
        logits[:, -1].scatter_(1, next_tokens[:, None], 1.0)
        return logits


def test_equal_length_batch_matches_individual_generation() -> None:
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
    input_ids = torch.tensor(
        [
            [1, 4, 7, 10],
            [1, 5, 8, 11],
        ]
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

    cached_batch = greedy_generate(model, input_ids, cached_config)
    no_cache_batch = greedy_generate(model, input_ids, no_cache_config)
    individual_sequences = torch.cat(
        [
            greedy_generate(model, row[None, :], cached_config).sequences
            for row in input_ids
        ],
        dim=0,
    )

    assert torch.equal(cached_batch.sequences, no_cache_batch.sequences)
    assert torch.equal(cached_batch.sequences, individual_sequences)
    assert torch.equal(cached_batch.generated_tokens, torch.tensor([4, 4]))
    assert not cached_batch.stopped_by_eos.any()


def test_equal_length_batch_tracks_eos_per_sequence() -> None:
    input_ids = torch.tensor([[1, 4], [1, 5]])
    output = greedy_generate(
        _ScriptedModel(),
        input_ids,
        GenerationConfig(max_new_tokens=4, eos_token_id=2, use_cache=False),
    )

    assert torch.equal(output.sequences[:, -2:], torch.tensor([[2, 2], [3, 2]]))
    assert torch.equal(output.generated_tokens, torch.tensor([1, 2]))
    assert output.stopped_by_eos.all()
