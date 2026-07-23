import torch

from nano_infer_engine.cache import KVCache
from nano_infer_engine.generation.prefill import prefill
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
