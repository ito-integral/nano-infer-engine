import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import time
from typing import Any, Literal
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nano_infer_engine.generation.async_engine import (
    AsyncInferenceEngine,
    AsyncPDInferenceEngine,
)
from nano_infer_engine.generation.request import RequestStatus


@dataclass(frozen=True)
class InferenceRuntime:
    """Process-wide objects shared by every HTTP request."""

    engine: AsyncInferenceEngine | AsyncPDInferenceEngine
    tokenizer: Any
    device: torch.device
    served_model_name: str


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    sequence_id: str | None = Field(default=None, min_length=1)


class GenerateResponse(BaseModel):
    sequence_id: str
    model: str
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


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "nano-infer-engine"


class OpenAIModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionStreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = 0.0
    stream: bool = False
    stream_options: ChatCompletionStreamOptions | None = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


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

    async def run_generation(
        runtime: InferenceRuntime,
        input_ids: torch.Tensor,
        sequence_id: str | None,
        max_tokens: int | None = None,
    ):
        try:
            handle = await runtime.engine.submit(
                input_ids,
                sequence_id=sequence_id,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        token_ids = []
        stopped_by_request_limit = False
        try:
            async for event in handle:
                token_ids.append(event.token_id)
                if max_tokens is not None and len(token_ids) >= max_tokens:
                    stopped_by_request_limit = await handle.cancel()
                    break
            result = await handle.result()
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return token_ids, result, stopped_by_request_limit

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
            free_blocks=scheduler.free_block_count,
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

        token_ids, result, _ = await run_generation(
            runtime,
            input_ids,
            body.sequence_id,
        )
        if result.status is RequestStatus.FAILED:
            detail = str(result.error) if result.error is not None else "generation failed"
            raise HTTPException(status_code=500, detail=detail)
        if result.status is RequestStatus.CANCELLED:
            raise HTTPException(status_code=409, detail="generation was cancelled")

        text = runtime.tokenizer.decode(token_ids, skip_special_tokens=True)
        return GenerateResponse(
            sequence_id=result.sequence_id,
            model=runtime.served_model_name,
            text=text,
            token_ids=token_ids,
            generated_tokens=result.generated_tokens,
            stopped_by_eos=result.stopped_by_eos,
            status=result.status.value,
        )

    @app.get("/v1/models", response_model=OpenAIModelList)
    async def list_models(http_request: Request) -> OpenAIModelList:
        runtime: InferenceRuntime = http_request.app.state.runtime
        return OpenAIModelList(
            data=[
                OpenAIModel(
                    id=runtime.served_model_name,
                    created=int(time()),
                )
            ]
        )

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
    )
    async def create_chat_completion(
        body: ChatCompletionRequest,
        http_request: Request,
    ) -> ChatCompletionResponse | StreamingResponse:
        runtime: InferenceRuntime = http_request.app.state.runtime
        if body.model != runtime.served_model_name:
            raise HTTPException(status_code=404, detail="model not found")
        if body.temperature != 0:
            raise HTTPException(
                status_code=400,
                detail="only greedy decoding with temperature=0 is supported",
            )

        engine_max_tokens = runtime.engine.scheduler.config.max_new_tokens
        if body.max_tokens is not None and body.max_tokens > engine_max_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    "max_tokens cannot exceed the engine limit "
                    f"({engine_max_tokens})"
                ),
            )

        messages = [message.model_dump() for message in body.messages]
        formatted_prompt = runtime.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = runtime.tokenizer(
            formatted_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(runtime.device)
        completion_id = f"chatcmpl-{uuid4().hex}"
        if body.stream:
            try:
                handle = await runtime.engine.submit(
                    input_ids,
                    sequence_id=completion_id,
                )
            except (TypeError, ValueError) as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error),
                ) from error
            except RuntimeError as error:
                raise HTTPException(
                    status_code=503,
                    detail=str(error),
                ) from error

            async def stream_chunks():
                token_ids: list[int] = []
                decoded_text = ""
                result = None
                created = int(time())

                def encode_chunk(payload: dict[str, Any]) -> str:
                    return f"data: {json.dumps(payload)}\n\n"

                try:
                    async for event in handle:
                        token_ids.append(event.token_id)
                        current_text = runtime.tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        )
                        if current_text.startswith(decoded_text):
                            delta = current_text[len(decoded_text) :]
                        else:
                            delta = runtime.tokenizer.decode(
                                [event.token_id],
                                skip_special_tokens=True,
                            )
                        decoded_text = current_text
                        if delta:
                            yield encode_chunk(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": runtime.served_model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": delta},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        if (
                            body.max_tokens is not None
                            and len(token_ids) >= body.max_tokens
                        ):
                            await handle.cancel()
                            break

                    result = await handle.result()
                    finish_reason = (
                        "stop" if result.stopped_by_eos else "length"
                    )
                    yield encode_chunk(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": runtime.served_model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                    )
                    if (
                        body.stream_options is not None
                        and body.stream_options.include_usage
                    ):
                        prompt_tokens = input_ids.shape[1]
                        completion_tokens = len(token_ids)
                        yield encode_chunk(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": runtime.served_model_name,
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": (
                                        prompt_tokens + completion_tokens
                                    ),
                                },
                            }
                        )
                    yield "data: [DONE]\n\n"
                finally:
                    if result is None:
                        await handle.cancel()

            return StreamingResponse(
                stream_chunks(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        token_ids, result, stopped_by_request_limit = await run_generation(
            runtime,
            input_ids,
            completion_id,
            body.max_tokens,
        )

        intentionally_cancelled = (
            stopped_by_request_limit
            and result.status is RequestStatus.CANCELLED
        )
        if result.status is RequestStatus.FAILED:
            detail = (
                str(result.error)
                if result.error is not None
                else "generation failed"
            )
            raise HTTPException(status_code=500, detail=detail)
        if result.status is RequestStatus.CANCELLED and not intentionally_cancelled:
            raise HTTPException(status_code=409, detail="generation was cancelled")

        completion_tokens = len(token_ids)
        prompt_tokens = input_ids.shape[1]
        finish_reason: Literal["stop", "length"] = (
            "stop" if result.stopped_by_eos else "length"
        )
        return ChatCompletionResponse(
            id=completion_id,
            created=int(time()),
            model=runtime.served_model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        content=runtime.tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        )
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app
