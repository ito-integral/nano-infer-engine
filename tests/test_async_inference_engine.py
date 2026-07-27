import asyncio

import torch

from nano_infer_engine.generation.async_engine import (
    AsyncInferenceEngine,
    AsyncPDInferenceEngine,
)
from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.events import RequestResult, TokenEvent
from nano_infer_engine.generation.request import RequestStatus
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache


class _AsyncScriptedModel:
    def __init__(
        self,
        token_schedules: dict[str, tuple[int, ...]],
        *,
        failing_prefill_id: str | None = None,
    ) -> None:
        self.token_schedules = token_schedules
        self.schedule_indices = {
            sequence_id: 0 for sequence_id in token_schedules
        }
        self.failing_prefill_id = failing_prefill_id

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
        else:
            current_sequence_ids = sequence_ids
            token_count = 1

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
            if (
                sequence_ids is None
                and current_sequence_id == self.failing_prefill_id
            ):
                raise RuntimeError("prefill failed")

            schedule_index = self.schedule_indices[current_sequence_id]
            next_token = self.token_schedules[current_sequence_id][schedule_index]
            self.schedule_indices[current_sequence_id] += 1
            logits[batch_index, -1, next_token] = 1.0
        return logits


def _build_engine(
    model: _AsyncScriptedModel,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = 2,
) -> tuple[AsyncInferenceEngine, PagedKVCache]:
    cache = PagedKVCache(
        num_blocks=max_new_tokens * 2,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    engine = AsyncInferenceEngine(
        model,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            use_cache=True,
        ),
        cache,
        max_batch_size=2,
    )
    return engine, cache


def _build_zero_model() -> Llama3_2:
    model = Llama3_2(
        LlamaConfig(
            vocab_size=8,
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
    for parameter in model.parameters():
        parameter.data.zero_()
    return model


def _build_pd_engine() -> tuple[
    AsyncPDInferenceEngine,
    PagedKVCache,
    PagedKVCache,
]:
    prefill_model = _build_zero_model()
    decode_model = _build_zero_model()

    def build_cache() -> PagedKVCache:
        return PagedKVCache(
            num_blocks=8,
            block_size=1,
            num_layers=1,
            kv_head_num=1,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
        )

    prefill_cache = build_cache()
    decode_cache = build_cache()
    engine = AsyncPDInferenceEngine(
        prefill_model,
        decode_model,
        GenerationConfig(
            max_new_tokens=4,
            eos_token_id=None,
            use_cache=True,
        ),
        prefill_cache,
        decode_cache,
        max_batch_size=1,
    )
    return engine, prefill_cache, decode_cache


async def _collect_tokens(handle) -> tuple[list[TokenEvent], RequestResult]:
    events = [event async for event in handle]
    return events, await handle.result()


def test_async_engine_streams_events_to_each_request() -> None:
    async def scenario() -> None:
        eos_token_id = 2
        engine, cache = _build_engine(
            _AsyncScriptedModel(
                {
                    "request-a": (3, eos_token_id),
                    "request-b": (4, 5, eos_token_id),
                }
            ),
            max_new_tokens=3,
            eos_token_id=eos_token_id,
        )
        handle_a = await engine.submit(
            torch.tensor([[1]]),
            sequence_id="request-a",
        )
        handle_b = await engine.submit(
            torch.tensor([[1]]),
            sequence_id="request-b",
        )

        (events_a, result_a), (events_b, result_b) = await asyncio.gather(
            _collect_tokens(handle_a),
            _collect_tokens(handle_b),
        )

        assert events_a == [
            TokenEvent("request-a", 3),
            TokenEvent("request-a", eos_token_id),
        ]
        assert events_b == [
            TokenEvent("request-b", 4),
            TokenEvent("request-b", 5),
            TokenEvent("request-b", eos_token_id),
        ]
        assert result_a.status is RequestStatus.COMPLETED
        assert result_b.status is RequestStatus.COMPLETED
        assert result_a.stopped_by_eos
        assert result_b.stopped_by_eos

        await engine.close()
        assert engine.is_closed
        assert cache.allocator.allocated_block_count == 0

    asyncio.run(scenario())


def test_async_pd_engine_accepts_request_while_decode_is_active() -> None:
    async def scenario() -> None:
        engine, prefill_cache, decode_cache = _build_pd_engine()
        handle_a = await engine.submit(
            torch.tensor([[1, 3]]),
            sequence_id="request-a",
        )

        first_event = await anext(handle_a)
        assert first_event == TokenEvent("request-a", 0)
        assert engine.scheduler.active_count == 1

        handle_b = await engine.submit(
            torch.tensor([[1, 4]]),
            sequence_id="request-b",
        )
        assert engine.scheduler.pending_count == 1

        remaining_a, result_a = await _collect_tokens(handle_a)
        events_b, result_b = await _collect_tokens(handle_b)

        assert remaining_a == [TokenEvent("request-a", 0)] * 3
        assert events_b == [TokenEvent("request-b", 0)] * 4
        assert result_a.status is RequestStatus.COMPLETED
        assert result_b.status is RequestStatus.COMPLETED
        assert result_a.sequence.device == torch.device("cpu")
        assert result_b.sequence.device == torch.device("cpu")

        await engine.close()
        assert prefill_cache.allocator.allocated_block_count == 0
        assert decode_cache.allocator.allocated_block_count == 0

    asyncio.run(scenario())


def test_async_engine_routes_prefill_failure_to_its_handle() -> None:
    async def scenario() -> None:
        eos_token_id = 2
        engine, cache = _build_engine(
            _AsyncScriptedModel(
                {
                    "request-failed": (3,),
                    "request-ok": (eos_token_id,),
                },
                failing_prefill_id="request-failed",
            ),
            max_new_tokens=2,
            eos_token_id=eos_token_id,
        )
        failed_handle = await engine.submit(
            torch.tensor([[1]]),
            sequence_id="request-failed",
        )
        successful_handle = await engine.submit(
            torch.tensor([[1]]),
            sequence_id="request-ok",
        )

        failed_events, failed_result = await _collect_tokens(failed_handle)
        successful_events, successful_result = await _collect_tokens(
            successful_handle
        )

        assert failed_events == []
        assert failed_result.status is RequestStatus.FAILED
        assert isinstance(failed_result.error, RuntimeError)
        assert successful_events == [
            TokenEvent("request-ok", eos_token_id)
        ]
        assert successful_result.status is RequestStatus.COMPLETED

        await engine.close()
        assert cache.allocator.allocated_block_count == 0

    asyncio.run(scenario())


def test_async_engine_wakes_after_becoming_idle() -> None:
    async def scenario() -> None:
        eos_token_id = 2
        engine, cache = _build_engine(
            _AsyncScriptedModel(
                {
                    "request-a": (eos_token_id,),
                    "request-b": (eos_token_id,),
                }
            ),
            max_new_tokens=1,
            eos_token_id=eos_token_id,
        )

        handle_a = await engine.submit(
            torch.tensor([[1]]),
            sequence_id="request-a",
        )
        events_a, result_a = await _collect_tokens(handle_a)
        assert events_a == [TokenEvent("request-a", eos_token_id)]
        assert result_a.status is RequestStatus.COMPLETED
        assert engine.is_running

        handle_b = await engine.submit(
            torch.tensor([[1]]),
            sequence_id="request-b",
        )
        events_b, result_b = await _collect_tokens(handle_b)
        assert events_b == [TokenEvent("request-b", eos_token_id)]
        assert result_b.status is RequestStatus.COMPLETED

        await engine.close()
        assert cache.allocator.allocated_block_count == 0

    asyncio.run(scenario())


def test_async_engine_cancels_request_and_closes_idempotently() -> None:
    async def scenario() -> None:
        model = _AsyncScriptedModel(
            {"request-a": tuple(3 for _ in range(128))}
        )
        engine, cache = _build_engine(
            model,
            max_new_tokens=128,
            eos_token_id=None,
        )
        handle = await engine.submit(
            torch.tensor([[1]]),
            sequence_id="request-a",
        )

        first_event = await anext(handle)
        assert first_event == TokenEvent("request-a", 3)
        assert await handle.cancel()

        remaining_events = [event async for event in handle]
        result = await handle.result()
        assert all(event.sequence_id == "request-a" for event in remaining_events)
        assert result.status is RequestStatus.CANCELLED

        await engine.close()
        await engine.close()
        assert cache.allocator.allocated_block_count == 0

    asyncio.run(scenario())
