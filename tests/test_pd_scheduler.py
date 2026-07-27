import pytest
import torch

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.paged_greedy import paged_greedy_generate
from nano_infer_engine.generation.pd_scheduler import (
    PDContinuousBatchingScheduler,
)
from nano_infer_engine.generation.request import RequestStatus
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


def _build_cache(
    model: Llama3_2,
    device: str = "cpu",
    num_blocks: int = 8,
) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=2,
        num_layers=len(model.decoders),
        kv_head_num=model.config.kv_head_num,
        head_dim=model.config.head_dim,
        dtype=model.embed.weight.dtype,
        device=device,
    )


def _build_scheduler(
    config: GenerationConfig,
    *,
    max_batch_size: int = 2,
) -> PDContinuousBatchingScheduler:
    prefill_model = _build_model()
    decode_model = _build_model()
    return PDContinuousBatchingScheduler(
        prefill_model,
        decode_model,
        config,
        _build_cache(prefill_model),
        _build_cache(decode_model),
        max_batch_size,
    )


def test_pd_scheduler_matches_generation_without_device_split() -> None:
    config = GenerationConfig(
        max_new_tokens=3,
        eos_token_id=None,
        use_cache=True,
    )
    prompts = (
        torch.tensor([[1, 4]]),
        torch.tensor([[1, 5, 8, 11]]),
        torch.tensor([[1, 6, 9]]),
    )
    sequence_ids = ("request-a", "request-b", "request-c")
    reference_model = _build_model()
    expected = paged_greedy_generate(
        reference_model,
        prompts,
        config,
        _build_cache(reference_model, num_blocks=10),
        sequence_ids,
    )
    scheduler = _build_scheduler(config)
    requests = [
        scheduler.add_request(sequence_id, prompt)
        for sequence_id, prompt in zip(sequence_ids, prompts)
    ]

    scheduler.run_until_idle()

    for request, expected_sequence in zip(requests, expected.sequences):
        assert request.status is RequestStatus.COMPLETED
        assert torch.equal(request.sequence, expected_sequence)
    assert scheduler.prefill_cache.allocator.allocated_block_count == 0
    assert scheduler.decode_cache.allocator.allocated_block_count == 0


def test_pd_scheduler_refills_decode_batch_after_a_decode_step() -> None:
    scheduler = _build_scheduler(
        GenerationConfig(
            max_new_tokens=2,
            eos_token_id=None,
            use_cache=True,
        )
    )
    requests = [
        scheduler.add_request(
            f"request-{index}",
            torch.tensor([[1, index + 3]]),
        )
        for index in range(3)
    ]

    first_output = scheduler.step()
    assert tuple(event.sequence_id for event in first_output.token_events) == (
        "request-0",
        "request-1",
    )
    assert scheduler.active_count == 2
    assert scheduler.pending_count == 1

    second_output = scheduler.step()
    assert tuple(event.sequence_id for event in second_output.token_events) == (
        "request-0",
        "request-1",
    )
    assert requests[0].status is RequestStatus.COMPLETED
    assert requests[1].status is RequestStatus.COMPLETED
    assert requests[2].status is RequestStatus.ACTIVE
    assert requests[2].generated_tokens == 0
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 0


def test_pd_scheduler_prefill_failure_does_not_block_later_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nano_infer_engine.generation.pd_scheduler as pd_scheduler_module

    scheduler = _build_scheduler(
        GenerationConfig(
            max_new_tokens=1,
            eos_token_id=None,
            use_cache=True,
        )
    )
    original_paged_prefill = pd_scheduler_module.paged_prefill

    def fail_one_request(model, prompts, paged_cache, sequence_ids):
        if sequence_ids == ("request-b",):
            raise RuntimeError("simulated prefill failure")
        return original_paged_prefill(
            model,
            prompts,
            paged_cache,
            sequence_ids,
        )

    monkeypatch.setattr(
        pd_scheduler_module,
        "paged_prefill",
        fail_one_request,
    )
    requests = [
        scheduler.add_request(sequence_id, torch.tensor([[1, token_id]]))
        for sequence_id, token_id in (
            ("request-a", 3),
            ("request-b", 4),
            ("request-c", 5),
        )
    ]

    scheduler.run_until_idle()

    assert requests[0].status is RequestStatus.COMPLETED
    assert requests[1].status is RequestStatus.FAILED
    assert isinstance(requests[1].error, RuntimeError)
    assert requests[2].status is RequestStatus.COMPLETED
    assert scheduler.prefill_cache.allocator.allocated_block_count == 0
    assert scheduler.decode_cache.allocator.allocated_block_count == 0


def test_pd_scheduler_close_cancels_pending_and_active_requests() -> None:
    scheduler = _build_scheduler(
        GenerationConfig(
            max_new_tokens=3,
            eos_token_id=None,
            use_cache=True,
        ),
        max_batch_size=1,
    )
    active = scheduler.add_request("request-a", torch.tensor([[1, 3]]))
    pending = scheduler.add_request("request-b", torch.tensor([[1, 4]]))
    scheduler.step()

    cancelled = scheduler.close()

    assert cancelled == (pending, active)
    assert active.status is RequestStatus.CANCELLED
    assert pending.status is RequestStatus.CANCELLED
    assert scheduler.prefill_cache.allocator.allocated_block_count == 0
    assert scheduler.decode_cache.allocator.allocated_block_count == 0
    assert scheduler.close() == ()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires at least two CUDA devices",
)
def test_pd_scheduler_pipelines_requests_across_two_gpus() -> None:
    prefill_model = _build_model("cuda:0")
    decode_model = _build_model("cuda:1")
    for parameter in prefill_model.parameters():
        parameter.data.zero_()
    for parameter in decode_model.parameters():
        parameter.data.zero_()

    scheduler = PDContinuousBatchingScheduler(
        prefill_model,
        decode_model,
        GenerationConfig(
            max_new_tokens=2,
            eos_token_id=None,
            use_cache=True,
        ),
        _build_cache(prefill_model, "cuda:0"),
        _build_cache(decode_model, "cuda:1"),
        max_batch_size=2,
    )
    requests = [
        scheduler.add_request(
            f"request-{index}",
            torch.tensor([[1, index + 3]], device="cuda:0"),
        )
        for index in range(3)
    ]

    scheduler.run_until_idle()

    for index, request in enumerate(requests):
        assert request.sequence.device == torch.device("cuda:1")
        assert torch.equal(
            request.sequence,
            torch.tensor(
                [[1, index + 3, 0, 0]],
                device="cuda:1",
            ),
        )
    assert scheduler.prefill_cache.allocator.allocated_block_count == 0
    assert scheduler.decode_cache.allocator.allocated_block_count == 0
