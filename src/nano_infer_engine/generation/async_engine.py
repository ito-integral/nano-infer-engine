import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig
from .events import RequestResult, SchedulerStepOutput, TokenEvent
from .request import PagedRequest
from .scheduler import ContinuousBatchingScheduler


class _EndOfStream:
    pass


_END_OF_STREAM = _EndOfStream()


class AsyncRequestHandle(AsyncIterator[TokenEvent]):
    """Stream tokens and await the terminal result of one request."""

    def __init__(
        self,
        sequence_id: str,
        engine: "AsyncInferenceEngine",
    ) -> None:
        self.sequence_id = sequence_id
        self._engine = engine
        self._queue: asyncio.Queue[TokenEvent | _EndOfStream] = (
            asyncio.Queue()
        )
        self._result_future: asyncio.Future[RequestResult] = (
            # asyncio.get_running_loop() 找取号机
            # asyncio.Future() 创建一个未来对象（拿到取号机的号码）
            asyncio.get_running_loop().create_future()
        )

    def __aiter__(self) -> "AsyncRequestHandle":
        return self

    async def __anext__(self) -> TokenEvent:
        event = await self._queue.get()
        if event is _END_OF_STREAM:
            raise StopAsyncIteration
        return event

    async def result(self) -> RequestResult:
        """Wait until the request completes, fails, or is cancelled."""
        return await asyncio.shield(self._result_future)

    async def cancel(self) -> bool:
        """Cancel this request through its owning engine."""
        return await self._engine.cancel(self.sequence_id)

    def _put_token(self, event: TokenEvent) -> None:
        # 直接同步
        self._queue.put_nowait(event)

    def _finish(self, result: RequestResult) -> None:
        if self._result_future.done():
            return
        self._result_future.set_result(result)
        self._queue.put_nowait(_END_OF_STREAM)

    def _finish_with_error(self, error: Exception) -> None:
        if self._result_future.done():
            return
        self._result_future.set_exception(error)
        self._queue.put_nowait(_END_OF_STREAM)


class AsyncInferenceEngine:
    """Run a continuous-batching scheduler behind asynchronous request APIs."""

    def __init__(
        self,
        model,
        config: GenerationConfig,
        paged_cache: PagedKVCache,
        max_batch_size: int,
    ) -> None:
        self.scheduler = ContinuousBatchingScheduler(
            model,
            config,
            paged_cache,
            max_batch_size,
        )
        self._handles: dict[str, AsyncRequestHandle] = {}
        self._scheduler_lock = asyncio.Lock()
        self._work_available = asyncio.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._use_executor = paged_cache.keys.device.type == "cuda"
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def start(self) -> None:
        """Start the background scheduling task."""
        if self._closed:
            raise RuntimeError("async inference engine is closed")
        if self.is_running:
            return

        if self._use_executor:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="nano-infer-engine",
            )
        self._worker_task = asyncio.create_task(self._run(), name="inference-engine")

    async def submit(
        self,
        prompt: torch.Tensor,
        sequence_id: str | None = None,
    ) -> AsyncRequestHandle:
        """Submit a prompt and return its asynchronous result handle."""
        if self._closed:
            raise RuntimeError("async inference engine is closed")
        if not self.is_running:
            await self.start()

        current_sequence_id = sequence_id or str(uuid4())
        handle = AsyncRequestHandle(current_sequence_id, self)
        async with self._scheduler_lock:
            self.scheduler.add_request(current_sequence_id, prompt)
            self._handles[current_sequence_id] = handle
            self._work_available.set()
        return handle

    async def cancel(self, sequence_id: str) -> bool:
        """Cancel a request and finish its asynchronous handle."""
        async with self._scheduler_lock:
            cancelled = self.scheduler.cancel_request(sequence_id)
            if not cancelled:
                return False
            request = self.scheduler.get_request(sequence_id)
        self._finish_request(request)
        return True

    async def close(self) -> None:
        """Stop scheduling, cancel unfinished work, and release resources."""
        if not self._closed:
            self._closed = True
            async with self._scheduler_lock:
                cancelled_requests = self.scheduler.close()
            for request in cancelled_requests:
                self._finish_request(request)
            # 唤起操作
            self._work_available.set()

        if (
            self._worker_task is not None
            and self._worker_task is not asyncio.current_task()
        ):
            # 等待”这个任务彻底结束
            await self._worker_task
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def _run(self) -> None:
        try:
            while not self._closed:
                await self._work_available.wait()
                if self._closed:
                    break

                while True:
                    async with self._scheduler_lock:
                        if not self.scheduler.has_work:
                            self._work_available.clear()
                            break
                        output = await self._run_scheduler_step()
                        self._dispatch(output)
                    if not self._use_executor:
                        await asyncio.sleep(0)
        except Exception as error:
            await self._handle_worker_failure(error)

    async def _run_scheduler_step(self) -> SchedulerStepOutput:
        if not self._use_executor:
            # 同步操作
            return self.scheduler.step()
        if self._executor is None:
            raise RuntimeError("inference executor is not started")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.scheduler.step)

    def _dispatch(self, output: SchedulerStepOutput) -> None:
        for event in output.token_events:
            handle = self._handles.get(event.sequence_id)
            if handle is not None:
                handle._put_token(event)
        for request in output.terminal_requests:
            self._finish_request(request)

    def _finish_request(self, request: PagedRequest) -> None:
        handle = self._handles.pop(request.sequence_id, None)
        if handle is None:
            return
        handle._finish(
            RequestResult(
                sequence_id=request.sequence_id,
                status=request.status,
                sequence=request.sequence.detach().clone(),
                generated_tokens=request.generated_tokens,
                stopped_by_eos=request.finished,
                error=request.error,
            )
        )

    async def _handle_worker_failure(self, error: Exception) -> None:
        async with self._scheduler_lock:
            self.scheduler.close()
        for handle in tuple(self._handles.values()):
            handle._finish_with_error(error)
        self._handles.clear()
        self._closed = True
