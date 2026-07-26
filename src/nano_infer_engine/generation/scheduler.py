from collections import deque

import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig
from .events import SchedulerStepOutput, TokenEvent
from .paged_prefill import _validate_paged_prefill_inputs, paged_prefill
from .request import PagedRequest, RequestStatus


class ContinuousBatchingScheduler:
    """Schedule dynamically submitted requests over a shared paged KV cache."""

    def __init__(
        self,
        model,
        config: GenerationConfig,
        paged_cache: PagedKVCache,
        max_batch_size: int,
        *,
        release_on_token_limit: bool = True,
    ) -> None:
        if not config.use_cache:
            raise ValueError("continuous batching requires config.use_cache=True")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(paged_cache, PagedKVCache):
            raise TypeError("paged_cache must be a PagedKVCache")

        self.model = model
        self.config = config
        self.paged_cache = paged_cache
        self.max_batch_size = max_batch_size
        self.release_on_token_limit = release_on_token_limit
        self.block_budget = paged_cache.allocator.free_block_count

        self.pending_requests: deque[PagedRequest] = deque()
        self.active_requests: list[PagedRequest] = []
        self.completed_requests: list[PagedRequest] = []
        self.reserved_blocks = 0
        self._requests: dict[str, PagedRequest] = {}
        self._closed = False

    @property
    def has_work(self) -> bool:
        """Return whether pending or active requests remain."""
        return bool(self.pending_requests or self.active_requests)

    @property
    def pending_count(self) -> int:
        return len(self.pending_requests)

    @property
    def active_count(self) -> int:
        return len(self.active_requests)

    def get_request(self, sequence_id: str) -> PagedRequest:
        """Return request state owned by this scheduler."""
        try:
            return self._requests[sequence_id]
        except KeyError:
            raise KeyError(sequence_id) from None

    @property
    def is_closed(self) -> bool:
        return self._closed

    def add_request(self, sequence_id: str, prompt: torch.Tensor) -> PagedRequest:
        """Submit a request to the FIFO pending queue."""
        if self._closed:
            raise RuntimeError("scheduler is closed")
        if sequence_id in self._requests:
            raise ValueError(f"sequence ID already submitted: {sequence_id}")

        _validate_paged_prefill_inputs(
            (prompt,),
            self.paged_cache,
            (sequence_id,),
        )
        required_blocks = (
            prompt.shape[1]
            + self.config.max_new_tokens
            - 1
            + self.paged_cache.block_size
            - 1
        ) // self.paged_cache.block_size
        if required_blocks > self.block_budget:
            raise ValueError(
                f"request cannot fit in paged cache: {sequence_id}"
            )

        request = PagedRequest(
            sequence_id=sequence_id,
            prompt=prompt,
            required_blocks=required_blocks,
        )
        self.pending_requests.append(request)
        self._requests[sequence_id] = request
        return request

    def _release_request_cache(self, sequence_id: str) -> None:
        try:
            self.paged_cache.get_block_table(sequence_id)
        except KeyError:
            return
        self.paged_cache.release(sequence_id)

    def _admit_pending_requests(self) -> list[PagedRequest]:
        failed_requests: list[PagedRequest] = []
        while (
            self.pending_requests
            and len(self.active_requests) < self.max_batch_size
        ):
            request = self.pending_requests[0]
            if (
                self.reserved_blocks + request.required_blocks
                > self.block_budget
            ):
                break

            self.pending_requests.popleft()
            try:
                logits = paged_prefill(
                    self.model,
                    (request.prompt,),
                    self.paged_cache,
                    (request.sequence_id,),
                )
                request.last_logits = logits[0]
            except Exception as error:
                self._release_request_cache(request.sequence_id)
                request.status = RequestStatus.FAILED
                request.error = error
                failed_requests.append(request)
                continue

            request.status = RequestStatus.ACTIVE
            self.active_requests.append(request)
            self.reserved_blocks += request.required_blocks
        return failed_requests

    def cancel_request(self, sequence_id: str) -> bool:
        """Cancel a pending or active request and release its cache blocks."""
        request = self._requests.get(sequence_id)
        if request is None:
            raise KeyError(sequence_id)
        if request.status not in {
            RequestStatus.PENDING,
            RequestStatus.ACTIVE,
        }:
            return False

        if request.status is RequestStatus.PENDING:
            self.pending_requests = deque(
                pending
                for pending in self.pending_requests
                if pending is not request
            )
        else:
            self.active_requests = [
                active
                for active in self.active_requests
                if active is not request
            ]
            self.reserved_blocks -= request.required_blocks
            self._release_request_cache(request.sequence_id)

        request.status = RequestStatus.CANCELLED
        self.completed_requests.append(request)
        return True

    def close(self) -> tuple[PagedRequest, ...]:
        """Cancel unfinished requests and release all scheduler-owned cache."""
        if self._closed:
            return ()

        self._closed = True
        cancelled_requests: list[PagedRequest] = []
        for request in (*self.pending_requests, *self.active_requests):
            request.status = RequestStatus.CANCELLED
            cancelled_requests.append(request)

        for request in self._requests.values():
            self._release_request_cache(request.sequence_id)

        self.pending_requests.clear()
        self.active_requests.clear()
        self.reserved_blocks = 0
        self.completed_requests.extend(cancelled_requests)
        return tuple(cancelled_requests)

    @torch.inference_mode()
    def step(self) -> SchedulerStepOutput:
        """Run one iteration and return its tokens and terminal requests."""
        if self._closed:
            raise RuntimeError("scheduler is closed")
        completed_now = self._admit_pending_requests()
        if not self.active_requests:
            if self.pending_requests:
                raise RuntimeError("pending requests cannot be admitted")
            self.completed_requests.extend(completed_now)
            return SchedulerStepOutput((), tuple(completed_now))

        request_logits: list[torch.Tensor] = []
        for request in self.active_requests:
            if request.last_logits is None:
                raise RuntimeError("active request is missing logits")
            request_logits.append(request.last_logits)
        last_logits = torch.stack(request_logits)
        next_tokens = last_logits.argmax(dim=-1, keepdim=True)
        next_token_ids = next_tokens.squeeze(-1).tolist()
        token_events = tuple(
            TokenEvent(
                sequence_id=request.sequence_id,
                token_id=next_token_ids[local_index],
            )
            for local_index, request in enumerate(self.active_requests)
        )

        for local_index, request in enumerate(self.active_requests):
            request.generated_tokens += 1
            request.sequence = torch.cat(
                (
                    request.sequence,
                    next_tokens[local_index : local_index + 1],
                ),
                dim=1,
            )

        if self.config.eos_token_id is not None:
            finished_now = next_tokens.squeeze(-1).eq(
                self.config.eos_token_id
            )
        else:
            finished_now = torch.zeros(
                len(self.active_requests),
                dtype=torch.bool,
                device=next_tokens.device,
            )

        survivor_indices: list[int] = []
        survivors: list[PagedRequest] = []
        for local_index, request in enumerate(self.active_requests):
            stopped_by_eos = bool(finished_now[local_index])
            reached_token_limit = (
                request.generated_tokens >= self.config.max_new_tokens
            )
            if stopped_by_eos or reached_token_limit:
                request.finished = stopped_by_eos
                request.status = RequestStatus.COMPLETED
                self.reserved_blocks -= request.required_blocks
                if stopped_by_eos or self.release_on_token_limit:
                    self._release_request_cache(request.sequence_id)
                completed_now.append(request)
                continue

            survivor_indices.append(local_index)
            survivors.append(request)

        self.active_requests = survivors
        if survivors:
            try:
                logits = self.model(
                    next_tokens[survivor_indices],
                    kv_cache=self.paged_cache,
                    sequence_ids=tuple(
                        request.sequence_id for request in survivors
                    ),
                )
                for local_index, request in enumerate(survivors):
                    request.last_logits = logits[local_index, -1]
            except Exception as error:
                for request in survivors:
                    request.status = RequestStatus.FAILED
                    request.error = error
                    self.reserved_blocks -= request.required_blocks
                    self._release_request_cache(request.sequence_id)
                    completed_now.append(request)
                self.active_requests = []

        completed_now.extend(self._admit_pending_requests())
        self.completed_requests.extend(completed_now)
        return SchedulerStepOutput(token_events, tuple(completed_now))

    def run_until_idle(self) -> tuple[PagedRequest, ...]:
        """Run steps until every currently submitted request completes."""
        completed: list[PagedRequest] = []
        while self.has_work:
            completed.extend(self.step().terminal_requests)
        return tuple(completed)
