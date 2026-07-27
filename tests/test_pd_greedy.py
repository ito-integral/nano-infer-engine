import pytest
import torch

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.paged_greedy import paged_greedy_generate
from nano_infer_engine.generation.pd_greedy import pd_greedy_generate
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache


def _build_model(device: str = "cpu") -> Llama3_2:
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
    ).to(device).eval()


def _build_cache(model: Llama3_2, device: str = "cpu") -> PagedKVCache:
    return PagedKVCache(
        num_blocks=4,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=device,
    )


def test_pd_greedy_matches_generation_without_kv_transfer() -> None:
    reference_model = _build_model()
    prefill_model = _build_model()
    decode_model = _build_model()
    prompt = torch.tensor([[1, 4, 7]])
    config = GenerationConfig(
        max_new_tokens=4,
        eos_token_id=None,
        use_cache=True,
    )
    reference_cache = _build_cache(reference_model)
    prefill_cache = _build_cache(prefill_model)
    decode_cache = _build_cache(decode_model)

    expected = paged_greedy_generate(
        reference_model,
        (prompt,),
        config,
        reference_cache,
        ("reference",),
    )
    actual = pd_greedy_generate(
        prefill_model,
        decode_model,
        prompt,
        config,
        prefill_cache,
        decode_cache,
        "migrated",
    )

    assert torch.equal(actual.sequences, expected.sequences[0])
    assert torch.equal(actual.generated_tokens, expected.generated_tokens)
    assert torch.equal(actual.stopped_by_eos, expected.stopped_by_eos)
    assert prefill_cache.allocator.allocated_block_count == 0
    assert decode_cache.get_sequence_length("migrated") == 6


def test_pd_greedy_releases_both_caches_when_handoff_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefill_model = _build_model()
    decode_model = _build_model()
    prefill_cache = _build_cache(prefill_model)
    decode_cache = _build_cache(decode_model)

    def fail_after_import(sequence_id: str, transfer: object) -> None:
        decode_cache.ensure_capacity(sequence_id, 1)
        raise RuntimeError("simulated handoff failure")

    monkeypatch.setattr(decode_cache, "import_sequence", fail_after_import)

    with pytest.raises(RuntimeError, match="simulated handoff failure"):
        pd_greedy_generate(
            prefill_model,
            decode_model,
            torch.tensor([[1, 4, 7]]),
            GenerationConfig(max_new_tokens=2, use_cache=True),
            prefill_cache,
            decode_cache,
            "request-a",
        )

    assert prefill_cache.allocator.allocated_block_count == 0
    assert decode_cache.allocator.allocated_block_count == 0


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires at least two CUDA devices",
)
def test_pd_greedy_runs_prefill_and_decode_on_different_gpus() -> None:
    prefill_model = _build_model("cuda:0")
    decode_model = _build_model("cuda:1")
    for parameter in prefill_model.parameters():
        parameter.data.zero_()
    for parameter in decode_model.parameters():
        parameter.data.zero_()

    prefill_cache = _build_cache(prefill_model, "cuda:0")
    decode_cache = _build_cache(decode_model, "cuda:1")
    prompt = torch.tensor([[1, 4, 7]], device="cuda:0")

    output = pd_greedy_generate(
        prefill_model,
        decode_model,
        prompt,
        GenerationConfig(
            max_new_tokens=3,
            eos_token_id=None,
            use_cache=True,
        ),
        prefill_cache,
        decode_cache,
        "request-a",
    )

    assert output.sequences.device == torch.device("cuda:1")
    assert torch.equal(
        output.sequences,
        torch.tensor([[1, 4, 7, 0, 0, 0]], device="cuda:1"),
    )
    assert prefill_cache.allocator.allocated_block_count == 0
    assert decode_cache.get_sequence_length("request-a") == 5
