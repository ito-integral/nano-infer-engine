from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from nano_infer_engine.generation.async_engine import AsyncInferenceEngine
from nano_infer_engine.generation.request import RequestStatus


@dataclass(frozen=True)
class InferenceRuntime:
    """Process-wide objects shared by every HTTP request."""

    engine: AsyncInferenceEngine
    tokenizer: Any
    device: torch.device


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    sequence_id: str | None = Field(default=None, min_length=1)


class GenerateResponse(BaseModel):
    sequence_id: str
    text: str
    token_ids: list[int]
    generated_tokens: int
    stopped_by_eos: bool
    status: str


class HealthResponse(BaseModel):
    status: str
    engine_running: bool
    engine_closed: bool
    pending_requests: int
    active_requests: int
    free_blocks: int


def create_app(
    runtime_factory: Callable[[], InferenceRuntime],
) -> FastAPI:
    """Create an application with one runtime for its entire lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory()
        app.state.runtime = runtime
        await runtime.engine.start()
        try:
            yield
        finally:
            await runtime.engine.close()

    app = FastAPI(
        title="Nano Infer Engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(http_request: Request) -> HealthResponse:
        runtime: InferenceRuntime = http_request.app.state.runtime
        scheduler = runtime.engine.scheduler
        engine_available = (
            runtime.engine.is_running and not runtime.engine.is_closed
        )
        return HealthResponse(
            status="ok" if engine_available else "unavailable",
            engine_running=runtime.engine.is_running,
            engine_closed=runtime.engine.is_closed,
            pending_requests=scheduler.pending_count,
            active_requests=scheduler.active_count,
            free_blocks=scheduler.paged_cache.allocator.free_block_count,
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(
        body: GenerateRequest,
        http_request: Request,
    ) -> GenerateResponse:
        runtime: InferenceRuntime = http_request.app.state.runtime
        formatted_prompt = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": body.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = runtime.tokenizer(
            formatted_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(runtime.device)

        try:
            handle = await runtime.engine.submit(
                input_ids,
                sequence_id=body.sequence_id,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        try:
            token_ids = [event.token_id async for event in handle]
            result = await handle.result()
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if result.status is RequestStatus.FAILED:
            detail = str(result.error) if result.error is not None else "generation failed"
            raise HTTPException(status_code=500, detail=detail)
        if result.status is RequestStatus.CANCELLED:
            raise HTTPException(status_code=409, detail="generation was cancelled")

        text = runtime.tokenizer.decode(token_ids, skip_special_tokens=True)
        return GenerateResponse(
            sequence_id=result.sequence_id,
            text=text,
            token_ids=token_ids,
            generated_tokens=result.generated_tokens,
            stopped_by_eos=result.stopped_by_eos,
            status=result.status.value,
        )

    return app
