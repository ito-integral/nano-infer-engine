from collections import deque

import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig
from .events import SchedulerStepOutput, TokenEvent
from .paged_prefill import (
    _validate_paged_prefill_inputs,
    paged_prefill,
    paged_prefill_chunks,
)
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
        self.prefilling_requests: list[PagedRequest] = []
        self.active_requests: list[PagedRequest] = []
        self.completed_requests: list[PagedRequest] = []
        self.reserved_blocks = 0
        self._requests: dict[str, PagedRequest] = {}
        self._closed = False

    @property
    def has_work(self) -> bool:
        """Return whether pending or active requests remain."""
        return bool(
            self.pending_requests or self.prefilling_requests or self.active_requests
        )

    @property
    def pending_count(self) -> int:
        return len(self.pending_requests)

    @property
    def active_count(self) -> int:
        return len(self.active_requests)

    @property
    def prefilling_count(self) -> int:
        return len(self.prefilling_requests)

    @property
    def free_block_count(self) -> int:
        return self.paged_cache.allocator.free_block_count

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
        model_config = getattr(self.model, "config", None)
        max_model_len = getattr(model_config, "max_seq_len", None)
        if max_model_len is not None and (
            prompt.shape[1] + self.config.max_new_tokens > max_model_len
        ):
            raise ValueError(
                f"request exceeds max model length: {sequence_id}"
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

    def add_prefilled_request(
        self,
        request: PagedRequest,
        last_logits: torch.Tensor,
    ) -> None:
        """Admit a request whose KV has already been imported into this cache."""
        if self._closed:
            raise RuntimeError("scheduler is closed")
        if not isinstance(request, PagedRequest):
            raise TypeError("request must be a PagedRequest")
        if request.sequence_id in self._requests:
            raise ValueError(
                f"sequence ID already submitted: {request.sequence_id}"
            )
        if request.status is not RequestStatus.PENDING:
            raise ValueError("prefilled request must have pending status")
        if (
            isinstance(request.required_blocks, bool)
            or not isinstance(request.required_blocks, int)
            or request.required_blocks <= 0
        ):
            raise ValueError("prefilled request must reserve at least one block")
        if len(self.active_requests) >= self.max_batch_size:
            raise ValueError("decode batch is full")
        if request.prompt.device != self.paged_cache.keys.device:
            raise ValueError("prefilled prompt must be on the decode cache device")
        if not isinstance(last_logits, torch.Tensor):
            raise TypeError("last_logits must be a torch.Tensor")
        if last_logits.ndim != 1:
            raise ValueError("last_logits must be a 1D tensor")
        if last_logits.device != self.paged_cache.keys.device:
            raise ValueError("last_logits must be on the decode cache device")
        if (
            self.reserved_blocks + request.required_blocks
            > self.block_budget
        ):
            raise ValueError("not enough reserved decode cache blocks")

        try:
            cached_length = self.paged_cache.get_sequence_length(
                request.sequence_id
            )
        except KeyError:
            raise ValueError("prefilled request is missing decode KV") from None
        if cached_length != request.prompt.shape[1]:
            raise ValueError("decode KV length must match the prompt length")

        request.last_logits = last_logits
        request.status = RequestStatus.ACTIVE
        self.active_requests.append(request)
        self.reserved_blocks += request.required_blocks
        self._requests[request.sequence_id] = request

    def _release_request_cache(self, sequence_id: str) -> None:
        try:
            self.paged_cache.get_block_table(sequence_id)
        except KeyError:
            return
        self.paged_cache.release(sequence_id)

    def _admit_pending_requests(self) -> list[PagedRequest]:
        if self.config.prefill_chunk_size is not None:
            self._admit_chunked_requests()
            return []

        failed_requests: list[PagedRequest] = []
        admitted_requests: list[PagedRequest] = []
        while (
            self.pending_requests
            and len(self.active_requests) + len(admitted_requests)
            < self.max_batch_size
        ):
            request = self.pending_requests[0]
            if (
                self.reserved_blocks
                + sum(admitted.required_blocks for admitted in admitted_requests)
                + request.required_blocks
                > self.block_budget
            ):
                break

            self.pending_requests.popleft()
            admitted_requests.append(request)

        if not admitted_requests:
            return failed_requests

        try:
            logits = paged_prefill(
                self.model,
                tuple(request.prompt for request in admitted_requests),
                self.paged_cache,
                tuple(request.sequence_id for request in admitted_requests),
            )
        except Exception:
            # Retry requests separately so one bad prompt does not fail the
            # other requests that shared its batched prefill attempt.
            for request in admitted_requests:
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

        for request, request_logits in zip(admitted_requests, logits):
            request.last_logits = request_logits
            request.status = RequestStatus.ACTIVE
            self.active_requests.append(request)
            self.reserved_blocks += request.required_blocks
        return failed_requests

    def _admit_chunked_requests(self) -> None:
        occupied_slots = len(self.active_requests) + len(self.prefilling_requests)
        while self.pending_requests and occupied_slots < self.max_batch_size:
            request = self.pending_requests[0]
            if self.reserved_blocks + request.required_blocks > self.block_budget:
                break
            self.pending_requests.popleft()
            request.status = RequestStatus.PREFILLING
            self.prefilling_requests.append(request)
            self.reserved_blocks += request.required_blocks
            occupied_slots += 1

    def _take_prefill_chunks(
        self,
    ) -> tuple[list[PagedRequest], tuple[torch.Tensor, ...]]:
        chunk_size = self.config.prefill_chunk_size
        assert chunk_size is not None
        token_budget = self.config.max_prefill_tokens_per_step
        if token_budget is None:
            token_budget = sum(
                min(
                    chunk_size,
                    request.prompt.shape[1] - request.prefill_offset,
                )
                for request in self.prefilling_requests
            )

        scheduled_requests: list[PagedRequest] = []
        chunks_list: list[torch.Tensor] = []
        while self.prefilling_requests and token_budget > 0:
            request = self.prefilling_requests.pop(0)
            remaining_tokens = request.prompt.shape[1] - request.prefill_offset
            scheduled_tokens = min(chunk_size, remaining_tokens, token_budget)
            chunks_list.append(
                request.prompt[
                    :,
                    request.prefill_offset : request.prefill_offset
                    + scheduled_tokens,
                ]
            )
            scheduled_requests.append(request)
            token_budget -= scheduled_tokens

        return scheduled_requests, tuple(chunks_list)

    def _finish_prefill_chunks(
        self,
        scheduled_requests: list[PagedRequest],
        chunks: tuple[torch.Tensor, ...],
        logits: torch.Tensor,
    ) -> None:
        for request, request_logits, chunk in zip(scheduled_requests, logits, chunks):
            request.prefill_offset += chunk.shape[1]
            request.last_logits = request_logits
            if request.prefill_offset == request.prompt.shape[1]:
                request.status = RequestStatus.ACTIVE
                self.active_requests.append(request)
            else:
                # Append incomplete requests after the unscheduled requests so
                # the next step starts with work that missed this round.
                self.prefilling_requests.append(request)

    def cancel_request(self, sequence_id: str) -> bool:
        """Cancel a pending or active request and release its cache blocks."""
        request = self._requests.get(sequence_id)
        if request is None:
            raise KeyError(sequence_id)
        if request.status not in {
            RequestStatus.PENDING,
            RequestStatus.PREFILLING,
            RequestStatus.ACTIVE,
        }:
            return False

        if request.status is RequestStatus.PENDING:
            self.pending_requests = deque(
                pending
                for pending in self.pending_requests
                if pending is not request
            )
        elif request.status is RequestStatus.PREFILLING:
            self.prefilling_requests = [
                prefilling
                for prefilling in self.prefilling_requests
                if prefilling is not request
            ]
            self.reserved_blocks -= request.required_blocks
            self._release_request_cache(request.sequence_id)
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
        for request in (
            *self.pending_requests,
            *self.prefilling_requests,
            *self.active_requests,
        ):
            request.status = RequestStatus.CANCELLED
            cancelled_requests.append(request)

        for request in self._requests.values():
            self._release_request_cache(request.sequence_id)

        self.pending_requests.clear()
        self.prefilling_requests.clear()
        self.active_requests.clear()
        self.reserved_blocks = 0
        self.completed_requests.extend(cancelled_requests)
        return tuple(cancelled_requests)

    def _step_unified_ragged(self) -> SchedulerStepOutput:
        """Run chunked prefill and decode tokens in one flattened forward."""
        self._admit_chunked_requests()
        if (
            not self.active_requests
            and not self.prefilling_requests
            and self.pending_requests
        ):
            raise RuntimeError("pending requests cannot be admitted")
        completed_now: list[PagedRequest] = []

        decode_requests = list(self.active_requests)
        request_logits: list[torch.Tensor] = []
        for request in decode_requests:
            if request.last_logits is None:
                raise RuntimeError("active request is missing logits")
            request_logits.append(request.last_logits)

        if request_logits:
            next_tokens = torch.stack(request_logits).argmax(dim=-1, keepdim=True)
            next_token_ids = next_tokens.squeeze(-1).tolist()
        else:
            next_tokens = torch.empty(
                (0, 1),
                dtype=torch.long,
                device=self.paged_cache.keys.device,
            )
            next_token_ids = []

        token_events = tuple(
            TokenEvent(request.sequence_id, next_token_ids[index])
            for index, request in enumerate(decode_requests)
        )
        for index, request in enumerate(decode_requests):
            request.generated_tokens += 1
            request.sequence = torch.cat(
                (request.sequence, next_tokens[index : index + 1]), dim=1
            )

        if self.config.eos_token_id is None:
            finished_now = torch.zeros(
                len(decode_requests),
                dtype=torch.bool,
                device=next_tokens.device,
            )
        else:
            finished_now = next_tokens.squeeze(-1).eq(self.config.eos_token_id)

        survivors: list[PagedRequest] = []
        survivor_tokens: list[torch.Tensor] = []
        for index, request in enumerate(decode_requests):
            stopped_by_eos = bool(finished_now[index])
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
            else:
                survivors.append(request)
                survivor_tokens.append(next_tokens[index : index + 1])
        self.active_requests = survivors

        prefill_requests, prefill_chunks = self._take_prefill_chunks()
        ragged_requests = [*prefill_requests, *survivors]
        ragged_chunks = (
            *prefill_chunks,
            *(token for token in survivor_tokens),
        )
        if ragged_requests:
            try:
                logits = paged_prefill_chunks(
                    self.model,
                    tuple(ragged_chunks),
                    self.paged_cache,
                    tuple(request.sequence_id for request in ragged_requests),
                )
            except Exception as error:
                for request in ragged_requests:
                    request.status = RequestStatus.FAILED
                    request.error = error
                    self.reserved_blocks -= request.required_blocks
                    self._release_request_cache(request.sequence_id)
                    completed_now.append(request)
                self.active_requests = []
            else:
                prefill_count = len(prefill_requests)
                self._finish_prefill_chunks(
                    prefill_requests,
                    prefill_chunks,
                    logits[:prefill_count],
                )
                for request, request_logits in zip(
                    survivors, logits[prefill_count:]
                ):
                    request.last_logits = request_logits

        self.completed_requests.extend(completed_now)
        return SchedulerStepOutput(token_events, tuple(completed_now))

    @torch.inference_mode()
    def step(self) -> SchedulerStepOutput:
        """Run one iteration and return its tokens and terminal requests."""
        if self._closed:
            raise RuntimeError("scheduler is closed")
        if self.config.prefill_chunk_size is not None:
            return self._step_unified_ragged()
        completed_now = self._admit_pending_requests()
        if not self.active_requests:
            if self.pending_requests and not self.prefilling_requests:
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
