import pytest
import torch

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.events import TokenEvent
from nano_infer_engine.generation.greedy import greedy_generate
from nano_infer_engine.generation.paged_greedy import paged_greedy_generate
from nano_infer_engine.generation.request import PagedRequest, RequestStatus
from nano_infer_engine.generation.scheduler import ContinuousBatchingScheduler
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache


def test_paged_request_initializes_generation_state() -> None:
    prompt = torch.tensor([[1, 4]])

    request = PagedRequest(sequence_id="request-a", prompt=prompt)

    assert request.sequence is prompt
    assert request.generated_tokens == 0
    assert not request.finished
    assert request.status is RequestStatus.PENDING
    assert request.error is None


class _ScriptedPagedModel:
    def __init__(self, token_schedules: dict[str, tuple[int, ...]]) -> None:
        self.token_schedules = token_schedules
        self.next_schedule_indices = {sequence_id: 0 for sequence_id in token_schedules}
        self.decode_batch_sizes: list[int] = []
        self.decode_sequence_ids: list[tuple[str, ...]] = []
        self.prefill_sequence_ids: list[str] = []
        self.prefill_batch_sizes: list[int] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        sequence_id: str = "default",
        sequence_ids: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        if sequence_ids is None:
            current_sequence_ids = (sequence_id,)
            token_count = input_ids.shape[1]
            self.prefill_sequence_ids.append(sequence_id)
            self.prefill_batch_sizes.append(1)
        else:
            current_sequence_ids = sequence_ids
            is_prefill = all(
                kv_cache.get_sequence_length(current_sequence_id) == 0
                for current_sequence_id in current_sequence_ids
            )
            token_count = input_ids.shape[1] if is_prefill else 1
            if is_prefill:
                self.prefill_sequence_ids.extend(sequence_ids)
                self.prefill_batch_sizes.append(len(sequence_ids))
            else:
                self.decode_batch_sizes.append(len(sequence_ids))
                self.decode_sequence_ids.append(sequence_ids)

        logits = torch.zeros(
            len(current_sequence_ids),
            input_ids.shape[1],
            8,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for batch_index, current_sequence_id in enumerate(current_sequence_ids):
            current_length = kv_cache.get_sequence_length(current_sequence_id)
            kv_cache.ensure_capacity(
                current_sequence_id,
                current_length + token_count,
            )
            kv_cache.advance(current_sequence_id, token_count)

            schedule_index = self.next_schedule_indices[current_sequence_id]
            next_token = self.token_schedules[current_sequence_id][schedule_index]
            self.next_schedule_indices[current_sequence_id] += 1
            logits[batch_index, -1, next_token] = 1.0

        return logits


class _FailingPrefillModel(_ScriptedPagedModel):
    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        sequence_id: str = "default",
        sequence_ids: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        current_sequence_ids = (sequence_id,) if sequence_ids is None else sequence_ids
        if (
            "request-failed" in current_sequence_ids
            and all(
                kv_cache.get_sequence_length(current_sequence_id) == 0
                for current_sequence_id in current_sequence_ids
            )
        ):
            kv_cache.ensure_capacity("request-failed", input_ids.shape[1])
            kv_cache.advance("request-failed", input_ids.shape[1])
            raise RuntimeError("prefill failed")
        return super().__call__(
            input_ids,
            kv_cache=kv_cache,
            sequence_id=sequence_id,
            sequence_ids=sequence_ids,
        )


class _FailingDecodeModel(_ScriptedPagedModel):
    def __init__(
        self,
        token_schedules: dict[str, tuple[int, ...]],
        failing_sequence_ids: tuple[str, ...],
    ) -> None:
        super().__init__(token_schedules)
        self.failing_sequence_ids = failing_sequence_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        sequence_id: str = "default",
        sequence_ids: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        is_decode = sequence_ids is not None and any(
            kv_cache.get_sequence_length(current_sequence_id) > 0
            for current_sequence_id in sequence_ids
        )
        logits = super().__call__(
            input_ids,
            kv_cache=kv_cache,
            sequence_id=sequence_id,
            sequence_ids=sequence_ids,
        )
        if is_decode and sequence_ids == self.failing_sequence_ids:
            raise RuntimeError("decode failed")
        return logits


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


def test_paged_greedy_retires_eos_requests_and_releases_blocks() -> None:
    eos_token_id = 2
    model = _ScriptedPagedModel(
        {
            "request-a": (eos_token_id,),
            "request-b": (3, 4, eos_token_id),
            "request-c": (5, eos_token_id),
        }
    )
    cache = PagedKVCache(
        num_blocks=12,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )

    output = paged_greedy_generate(
        model,
        (
            torch.tensor([[6]]),
            torch.tensor([[6]]),
            torch.tensor([[6]]),
        ),
        GenerationConfig(
            max_new_tokens=4,
            eos_token_id=eos_token_id,
            use_cache=True,
        ),
        cache,
        ("request-a", "request-b", "request-c"),
    )

    assert tuple(sequence.shape[1] for sequence in output.sequences) == (2, 4, 3)
    assert torch.equal(output.generated_tokens, torch.tensor([1, 3, 2]))
    assert output.stopped_by_eos.all()
    assert model.decode_batch_sizes == [2, 1]
    assert cache.allocator.allocated_block_count == 0


def test_paged_greedy_admits_pending_requests_after_eos() -> None:
    eos_token_id = 2
    model = _ScriptedPagedModel(
        {
            "request-a": (eos_token_id,),
            "request-b": (3, 4, eos_token_id),
            "request-c": (5, eos_token_id),
        }
    )
    cache = PagedKVCache(
        num_blocks=8,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )

    output = paged_greedy_generate(
        model,
        (
            torch.tensor([[6]]),
            torch.tensor([[6]]),
            torch.tensor([[6]]),
        ),
        GenerationConfig(
            max_new_tokens=4,
            eos_token_id=eos_token_id,
            use_cache=True,
        ),
        cache,
        ("request-a", "request-b", "request-c"),
        max_batch_size=2,
    )

    assert model.prefill_sequence_ids == [
        "request-a",
        "request-b",
        "request-c",
    ]
    assert model.prefill_batch_sizes == [2, 1]
    assert model.decode_sequence_ids == [
        ("request-b",),
        ("request-b", "request-c"),
    ]
    assert torch.equal(output.generated_tokens, torch.tensor([1, 3, 2]))
    assert output.stopped_by_eos.all()
    assert cache.allocator.allocated_block_count == 0


def test_paged_greedy_admits_pending_requests_after_token_limit() -> None:
    model = _ScriptedPagedModel(
        {
            "request-a": (3, 4),
            "request-b": (5, 6),
        }
    )
    cache = PagedKVCache(
        num_blocks=2,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )

    output = paged_greedy_generate(
        model,
        (torch.tensor([[1]]), torch.tensor([[1]])),
        GenerationConfig(max_new_tokens=2, use_cache=True),
        cache,
        ("request-a", "request-b"),
        max_batch_size=1,
    )

    assert model.prefill_sequence_ids == ["request-a", "request-b"]
    assert model.decode_sequence_ids == [
        ("request-a",),
        ("request-b",),
    ]
    assert torch.equal(output.generated_tokens, torch.tensor([2, 2]))
    assert not output.stopped_by_eos.any()
    assert cache.allocator.allocated_block_count == 0


def test_scheduler_accepts_requests_between_decode_steps() -> None:
    eos_token_id = 2
    model = _ScriptedPagedModel(
        {
            "request-a": (eos_token_id,),
            "request-b": (3, eos_token_id),
            "request-c": (4, eos_token_id),
        }
    )
    cache = PagedKVCache(
        num_blocks=6,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(
            max_new_tokens=3,
            eos_token_id=eos_token_id,
            use_cache=True,
        ),
        cache,
        max_batch_size=2,
    )

    scheduler.add_request("request-a", torch.tensor([[6]]))
    scheduler.add_request("request-b", torch.tensor([[6]]))

    assert scheduler.pending_count == 2
    assert scheduler.active_count == 0
    step_output = scheduler.step()
    assert step_output.token_events == (
        TokenEvent("request-a", eos_token_id),
        TokenEvent("request-b", 3),
    )
    assert tuple(
        request.sequence_id for request in step_output.terminal_requests
    ) == ("request-a",)

    scheduler.add_request("request-c", torch.tensor([[6]]))

    assert scheduler.pending_count == 1
    assert scheduler.active_count == 1
    assert tuple(
        request.sequence_id
        for request in scheduler.step().terminal_requests
    ) == ("request-b",)
    assert tuple(
        request.sequence_id
        for request in scheduler.step().terminal_requests
    ) == ("request-c",)
    assert not scheduler.has_work
    assert model.prefill_sequence_ids == [
        "request-a",
        "request-b",
        "request-c",
    ]
    assert cache.allocator.allocated_block_count == 0


def test_scheduler_emits_each_generated_token_once() -> None:
    eos_token_id = 2
    model = _ScriptedPagedModel(
        {"request-a": (3, 4, eos_token_id)}
    )
    cache = PagedKVCache(
        num_blocks=3,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(
            max_new_tokens=3,
            eos_token_id=eos_token_id,
            use_cache=True,
        ),
        cache,
        max_batch_size=1,
    )
    request = scheduler.add_request("request-a", torch.tensor([[1]]))
    token_events: list[TokenEvent] = []
    terminal_requests: list[PagedRequest] = []

    while scheduler.has_work:
        step_output = scheduler.step()
        token_events.extend(step_output.token_events)
        terminal_requests.extend(step_output.terminal_requests)

    assert token_events == [
        TokenEvent("request-a", 3),
        TokenEvent("request-a", 4),
        TokenEvent("request-a", eos_token_id),
    ]
    assert terminal_requests == [request]
    assert torch.equal(request.sequence, torch.tensor([[1, 3, 4, 2]]))


def test_scheduler_cleans_up_failed_prefill_and_continues() -> None:
    eos_token_id = 2
    model = _FailingPrefillModel(
        {
            "request-failed": (3,),
            "request-ok": (eos_token_id,),
        }
    )
    cache = PagedKVCache(
        num_blocks=4,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(
            max_new_tokens=2,
            eos_token_id=eos_token_id,
            use_cache=True,
        ),
        cache,
        max_batch_size=2,
    )
    failed_request = scheduler.add_request(
        "request-failed",
        torch.tensor([[1]]),
    )
    successful_request = scheduler.add_request(
        "request-ok",
        torch.tensor([[1]]),
    )

    step_output = scheduler.step()

    assert step_output.token_events == (
        TokenEvent("request-ok", eos_token_id),
    )
    assert step_output.terminal_requests == (
        failed_request,
        successful_request,
    )
    assert failed_request.status is RequestStatus.FAILED
    assert isinstance(failed_request.error, RuntimeError)
    assert successful_request.status is RequestStatus.COMPLETED
    assert not scheduler.has_work
    assert cache.allocator.allocated_block_count == 0


def test_scheduler_advances_chunked_prefill_once_per_step() -> None:
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
            max_seq_len=8,
            tie_word_embeddings=False,
        )
    ).eval()
    cache = PagedKVCache(
        num_blocks=8,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(
            max_new_tokens=1,
            use_cache=True,
            prefill_chunk_size=2,
        ),
        cache,
        max_batch_size=1,
    )
    request = scheduler.add_request(
        "request-a", torch.tensor([[1, 2, 3, 4, 5]])
    )
    next_request = scheduler.add_request("request-b", torch.tensor([[6]]))

    assert scheduler.step().token_events == ()
    assert request.status is RequestStatus.PREFILLING
    assert request.prefill_offset == 2
    assert cache.get_sequence_length(request.sequence_id) == 2

    assert scheduler.step().token_events == ()
    assert request.status is RequestStatus.PREFILLING
    assert request.prefill_offset == 4
    assert cache.get_sequence_length(request.sequence_id) == 4

    output = scheduler.step()
    assert output.token_events == ()
    assert output.terminal_requests == ()
    assert request.status is RequestStatus.ACTIVE
    assert request.prefill_offset == 5

    output = scheduler.step()
    assert len(output.token_events) == 1
    assert output.terminal_requests == (request,)
    assert request.status is RequestStatus.COMPLETED
    assert next_request.status is RequestStatus.PENDING
    assert scheduler.pending_count == 1

    output = scheduler.step()
    assert output.token_events == ()
    assert next_request.status is RequestStatus.ACTIVE
    assert scheduler.pending_count == 0

    output = scheduler.step()
    assert output.terminal_requests == (next_request,)
    assert not scheduler.has_work


def test_scheduler_applies_global_prefill_budget_fairly() -> None:
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
            max_seq_len=8,
            tie_word_embeddings=False,
        )
    ).eval()
    cache = PagedKVCache(
        num_blocks=24,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(
            max_new_tokens=1,
            use_cache=True,
            prefill_chunk_size=2,
            max_prefill_tokens_per_step=3,
        ),
        cache,
        max_batch_size=3,
    )
    requests = [
        scheduler.add_request(
            f"request-{index}", torch.tensor([[1, 2, 3, 4, 5, 6]])
        )
        for index in range(3)
    ]

    scheduler.step()
    assert tuple(request.prefill_offset for request in requests) == (2, 1, 0)
    assert sum(
        cache.get_sequence_length(request.sequence_id) for request in requests
    ) == 3

    scheduler.step()
    assert tuple(request.prefill_offset for request in requests) == (3, 1, 2)
    assert sum(
        cache.get_sequence_length(request.sequence_id) for request in requests
    ) == 6

    scheduler.step()
    assert tuple(request.prefill_offset for request in requests) == (3, 3, 3)
    assert all(request.prefill_offset > 0 for request in requests)


def test_scheduler_unifies_prefill_and_decode_in_one_ragged_forward() -> None:
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
            max_seq_len=8,
            tie_word_embeddings=False,
        )
    ).eval()
    cache = PagedKVCache(
        num_blocks=12,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(
            max_new_tokens=2,
            use_cache=True,
            prefill_chunk_size=2,
        ),
        cache,
        max_batch_size=2,
    )
    request_a = scheduler.add_request("request-a", torch.tensor([[1]]))
    request_b = scheduler.add_request(
        "request-b", torch.tensor([[2, 3, 4, 5]])
    )
    calls: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    original_forward_ragged = model.forward_ragged

    def recording_forward_ragged(
        input_ids,
        *,
        kv_cache,
        sequence_ids,
        query_start_loc,
    ):
        calls.append((sequence_ids, tuple(query_start_loc.tolist())))
        return original_forward_ragged(
            input_ids,
            kv_cache=kv_cache,
            sequence_ids=sequence_ids,
            query_start_loc=query_start_loc,
        )

    model.forward_ragged = recording_forward_ragged  # type: ignore[method-assign]

    scheduler.step()
    assert request_a.status is RequestStatus.ACTIVE
    assert request_b.status is RequestStatus.PREFILLING
    calls.clear()

    output = scheduler.step()

    assert len(calls) == 1
    assert calls == [(('request-b', 'request-a'), (0, 2, 3))]
    assert output.token_events[0].sequence_id == "request-a"
    assert request_b.status is RequestStatus.ACTIVE


def test_scheduler_cancels_pending_and_active_requests() -> None:
    model = _ScriptedPagedModel(
        {
            "request-active": (3, 4, 5),
            "request-pending": (6,),
        }
    )
    cache = PagedKVCache(
        num_blocks=3,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(max_new_tokens=3, use_cache=True),
        cache,
        max_batch_size=1,
    )
    active_request = scheduler.add_request(
        "request-active",
        torch.tensor([[1]]),
    )
    pending_request = scheduler.add_request(
        "request-pending",
        torch.tensor([[1]]),
    )
    scheduler.step()

    assert scheduler.cancel_request("request-pending")
    assert scheduler.cancel_request("request-active")

    assert pending_request.status is RequestStatus.CANCELLED
    assert active_request.status is RequestStatus.CANCELLED
    assert not scheduler.cancel_request("request-active")
    assert not scheduler.has_work
    assert scheduler.reserved_blocks == 0
    assert cache.allocator.allocated_block_count == 0


def test_scheduler_cleans_up_failed_decode_and_continues() -> None:
    eos_token_id = 2
    model = _FailingDecodeModel(
        {
            "request-a": (3, 4),
            "request-b": (5, 6),
            "request-c": (eos_token_id,),
        },
        failing_sequence_ids=("request-a", "request-b"),
    )
    cache = PagedKVCache(
        num_blocks=6,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(
            max_new_tokens=3,
            eos_token_id=eos_token_id,
            use_cache=True,
        ),
        cache,
        max_batch_size=2,
    )
    request_a = scheduler.add_request("request-a", torch.tensor([[1]]))
    request_b = scheduler.add_request("request-b", torch.tensor([[1]]))
    request_c = scheduler.add_request("request-c", torch.tensor([[1]]))

    step_output = scheduler.step()
    failed_requests = step_output.terminal_requests

    assert step_output.token_events == (
        TokenEvent("request-a", 3),
        TokenEvent("request-b", 5),
    )
    assert failed_requests == (request_a, request_b)
    assert all(
        request.status is RequestStatus.FAILED
        for request in failed_requests
    )
    assert all(
        isinstance(request.error, RuntimeError)
        for request in failed_requests
    )
    assert scheduler.active_count == 1
    assert scheduler.pending_count == 0
    assert scheduler.reserved_blocks == request_c.required_blocks

    step_output = scheduler.step()
    assert step_output.token_events == (
        TokenEvent("request-c", eos_token_id),
    )
    assert step_output.terminal_requests == (request_c,)
    assert request_c.status is RequestStatus.COMPLETED
    assert not scheduler.has_work
    assert scheduler.reserved_blocks == 0
    assert cache.allocator.allocated_block_count == 0


def test_scheduler_close_cancels_work_and_is_idempotent() -> None:
    model = _ScriptedPagedModel(
        {
            "request-active": (3, 4, 5),
            "request-pending": (6,),
        }
    )
    cache = PagedKVCache(
        num_blocks=3,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(max_new_tokens=3, use_cache=True),
        cache,
        max_batch_size=1,
    )
    active_request = scheduler.add_request(
        "request-active",
        torch.tensor([[1]]),
    )
    pending_request = scheduler.add_request(
        "request-pending",
        torch.tensor([[1]]),
    )
    scheduler.step()

    assert scheduler.close() == (pending_request, active_request)
    assert active_request.status is RequestStatus.CANCELLED
    assert pending_request.status is RequestStatus.CANCELLED
    assert scheduler.is_closed
    assert not scheduler.has_work
    assert scheduler.reserved_blocks == 0
    assert cache.allocator.allocated_block_count == 0
    assert scheduler.close() == ()

    with pytest.raises(RuntimeError, match="scheduler is closed"):
        scheduler.step()
    with pytest.raises(RuntimeError, match="scheduler is closed"):
        scheduler.add_request("request-new", torch.tensor([[1]]))


def test_scheduler_close_releases_retained_completed_cache() -> None:
    model = _ScriptedPagedModel({"request-a": (3,)})
    cache = PagedKVCache(
        num_blocks=1,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    scheduler = ContinuousBatchingScheduler(
        model,
        GenerationConfig(max_new_tokens=1, use_cache=True),
        cache,
        max_batch_size=1,
        release_on_token_limit=False,
    )
    request = scheduler.add_request("request-a", torch.tensor([[1]]))

    step_output = scheduler.step()
    assert step_output.token_events == (TokenEvent("request-a", 3),)
    assert step_output.terminal_requests == (request,)
    assert request.status is RequestStatus.COMPLETED
    assert cache.allocator.allocated_block_count == 1

    assert scheduler.close() == ()
    assert request.status is RequestStatus.COMPLETED
    assert cache.allocator.allocated_block_count == 0
