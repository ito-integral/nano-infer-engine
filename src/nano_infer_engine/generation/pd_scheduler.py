from collections import deque

import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig
from .events import SchedulerStepOutput
from .paged_prefill import _validate_paged_prefill_inputs, paged_prefill
from .pd_greedy import _release_if_allocated, _validate_model_cache
from .request import PagedRequest, RequestStatus
from .scheduler import ContinuousBatchingScheduler


class PDContinuousBatchingScheduler:
    """Pipeline prefill on one device and continuous decode on another."""

    def __init__(
        self,
        prefill_model,
        decode_model,
        config: GenerationConfig,
        prefill_cache: PagedKVCache,
        decode_cache: PagedKVCache,
        max_batch_size: int,
    ) -> None:
        if not config.use_cache:
            raise ValueError("P/D scheduling requires config.use_cache=True")
        if prefill_cache is decode_cache:
            raise ValueError("prefill_cache and decode_cache must be different caches")

        _validate_model_cache(prefill_model, prefill_cache, "prefill")
        _validate_model_cache(decode_model, decode_cache, "decode")

        self.prefill_model = prefill_model
        self.decode_model = decode_model
        self.config = config
        self.prefill_cache = prefill_cache
        self.decode_cache = decode_cache
        self.decode_scheduler = ContinuousBatchingScheduler(
            decode_model,
            config,
            decode_cache,
            max_batch_size,
            release_on_token_limit=True,
        )
        self.max_batch_size = self.decode_scheduler.max_batch_size
        self.block_budget = self.decode_scheduler.block_budget

        self.pending_requests: deque[PagedRequest] = deque()
        self.completed_requests: list[PagedRequest] = []
        self._requests: dict[str, PagedRequest] = {}
        self._closed = False

    @property
    def has_work(self) -> bool:
        return bool(self.pending_requests or self.decode_scheduler.has_work)

    @property
    def pending_count(self) -> int:
        return len(self.pending_requests)

    @property
    def active_count(self) -> int:
        return self.decode_scheduler.active_count

    @property
    def is_closed(self) -> bool:
        return self._closed

    def get_request(self, sequence_id: str) -> PagedRequest:
        try:
            return self._requests[sequence_id]
        except KeyError:
            raise KeyError(sequence_id) from None

    def add_request(self, sequence_id: str, prompt: torch.Tensor) -> PagedRequest:
        """Submit a prompt to the P/D pipeline's FIFO queue."""
        if self._closed:
            raise RuntimeError("scheduler is closed")
        if sequence_id in self._requests:
            raise ValueError(f"sequence ID already submitted: {sequence_id}")

        _validate_paged_prefill_inputs(
            (prompt,),
            self.prefill_cache,
            (sequence_id,),
        )
        try:
            self.decode_cache.get_block_table(sequence_id)
        except KeyError:
            pass
        else:
            raise ValueError(f"sequence ID already exists: {sequence_id}")

        required_blocks = (
            prompt.shape[1]
            + self.config.max_new_tokens
            - 1
            + self.decode_cache.block_size
            - 1
        ) // self.decode_cache.block_size
        if required_blocks > self.block_budget:
            raise ValueError(
                f"request cannot fit in decode cache: {sequence_id}"
            )
        required_prefill_blocks = (
            prompt.shape[1] + self.prefill_cache.block_size - 1
        ) // self.prefill_cache.block_size
        if required_prefill_blocks > self.prefill_cache.allocator.num_blocks:
            raise ValueError(
                f"request cannot fit in prefill cache: {sequence_id}"
            )

        request = PagedRequest(
            sequence_id=sequence_id,
            prompt=prompt,
            required_blocks=required_blocks,
        )
        self.pending_requests.append(request)
        self._requests[sequence_id] = request
        return request

    def _handoff_request(self, request: PagedRequest) -> None:
        try:
            logits = paged_prefill(
                self.prefill_model,
                (request.prompt,),
                self.prefill_cache,
                (request.sequence_id,),
            )
            transfer = self.prefill_cache.export_sequence(request.sequence_id)
            self.decode_cache.import_sequence(request.sequence_id, transfer)
            del transfer

            decode_prompt = request.prompt.to(self.decode_cache.keys.device)
            last_logits = logits[0].to(self.decode_cache.keys.device)
            request.prompt = decode_prompt
            request.sequence = decode_prompt
            self.decode_scheduler.add_prefilled_request(request, last_logits)
        except Exception:
            _release_if_allocated(self.decode_cache, request.sequence_id)
            raise
        finally:
            _release_if_allocated(self.prefill_cache, request.sequence_id)

    def _admit_pending_requests(self) -> list[PagedRequest]:
        failed_requests: list[PagedRequest] = []
        while (
            self.pending_requests
            and self.active_count < self.max_batch_size
        ):
            request = self.pending_requests[0]
            if (
                self.decode_scheduler.reserved_blocks
                + request.required_blocks
                > self.block_budget
            ):
                break

            self.pending_requests.popleft()
            try:
                self._handoff_request(request)
            except Exception as error:
                request.status = RequestStatus.FAILED
                request.error = error
                failed_requests.append(request)

        return failed_requests

    def cancel_request(self, sequence_id: str) -> bool:
        """Cancel a pending or decoding request and release its KV blocks."""
        request = self.get_request(sequence_id)
        if request.status is RequestStatus.PENDING:
            self.pending_requests = deque(
                pending
                for pending in self.pending_requests
                if pending is not request
            )
            request.status = RequestStatus.CANCELLED
            self.completed_requests.append(request)
            return True
        if request.status is RequestStatus.ACTIVE:
            cancelled = self.decode_scheduler.cancel_request(sequence_id)
            if cancelled:
                self.completed_requests.append(request)
            return cancelled
        return False

    def close(self) -> tuple[PagedRequest, ...]:
        """Cancel all unfinished requests and release both caches."""
        if self._closed:
            return ()
        self._closed = True

        cancelled = []
        for request in self.pending_requests:
            request.status = RequestStatus.CANCELLED
            cancelled.append(request)
        self.pending_requests.clear()
        cancelled.extend(self.decode_scheduler.close())

        for request in self._requests.values():
            _release_if_allocated(self.prefill_cache, request.sequence_id)
            _release_if_allocated(self.decode_cache, request.sequence_id)

        self.completed_requests.extend(cancelled)
        return tuple(cancelled)

    @torch.inference_mode()
    def step(self) -> SchedulerStepOutput:
        """Advance decode, then fill free slots through the prefill pipeline."""
        if self._closed:
            raise RuntimeError("scheduler is closed")

        token_events = ()
        completed_now: list[PagedRequest] = []
        decoded_existing_batch = self.active_count > 0
        if decoded_existing_batch:
            decode_output = self.decode_scheduler.step()
            token_events = decode_output.token_events
            completed_now.extend(decode_output.terminal_requests)

        completed_now.extend(self._admit_pending_requests())

        # Bootstrap an empty decode device without requiring an extra step.
        if not decoded_existing_batch and self.active_count > 0:
            decode_output = self.decode_scheduler.step()
            token_events = decode_output.token_events
            completed_now.extend(decode_output.terminal_requests)
            completed_now.extend(self._admit_pending_requests())

        self.completed_requests.extend(completed_now)
        return SchedulerStepOutput(token_events, tuple(completed_now))

    def run_until_idle(self) -> tuple[PagedRequest, ...]:
        """Run pipeline steps until every submitted request is terminal."""
        completed = []
        while self.has_work:
            completed.extend(self.step().terminal_requests)
        return tuple(completed)
