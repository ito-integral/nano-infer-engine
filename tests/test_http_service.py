import asyncio
from types import SimpleNamespace

import httpx
import torch

from nano_infer_engine.generation.async_engine import (
    AsyncInferenceEngine,
    AsyncPDInferenceEngine,
)
from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig
from nano_infer_engine.paged_cache import PagedKVCache
from nano_infer_engine.service.app import InferenceRuntime, create_app


class _ServiceTokenizer:
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return messages[0]["content"]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> SimpleNamespace:
        assert text
        assert not add_special_tokens
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[1]]))

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens
        return " ".join(
            str(token_id)
            for token_id in token_ids
            if token_id != self.eos_token_id
        )


class _ServiceModel:
    def __init__(self) -> None:
        self.schedules = {
            "request-a": (3, 2),
        }
        self.indices = {sequence_id: 0 for sequence_id in self.schedules}

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
        )
        for batch_index, current_sequence_id in enumerate(current_sequence_ids):
            current_length = kv_cache.get_sequence_length(current_sequence_id)
            kv_cache.ensure_capacity(
                current_sequence_id,
                current_length + token_count,
            )
            kv_cache.advance(current_sequence_id, token_count)
            schedule_index = self.indices[current_sequence_id]
            next_token = self.schedules[current_sequence_id][schedule_index]
            self.indices[current_sequence_id] += 1
            logits[batch_index, -1, next_token] = 1.0
        return logits


def _build_runtime() -> tuple[InferenceRuntime, PagedKVCache]:
    cache = PagedKVCache(
        num_blocks=4,
        block_size=1,
        num_layers=1,
        kv_head_num=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    engine = AsyncInferenceEngine(
        _ServiceModel(),
        GenerationConfig(
            max_new_tokens=2,
            eos_token_id=2,
            use_cache=True,
        ),
        cache,
        max_batch_size=2,
    )
    return (
        InferenceRuntime(
            engine=engine,
            tokenizer=_ServiceTokenizer(),
            device=torch.device("cpu"),
        ),
        cache,
    )


def _build_pd_runtime() -> tuple[
    InferenceRuntime,
    PagedKVCache,
    PagedKVCache,
]:
    def build_model() -> Llama3_2:
        model = Llama3_2(
            LlamaConfig(
                vocab_size=8,
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
        for parameter in model.parameters():
            parameter.data.zero_()
        return model

    def build_cache() -> PagedKVCache:
        return PagedKVCache(
            num_blocks=4,
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
        build_model(),
        build_model(),
        GenerationConfig(
            max_new_tokens=2,
            eos_token_id=None,
            use_cache=True,
        ),
        prefill_cache,
        decode_cache,
        max_batch_size=1,
    )
    return (
        InferenceRuntime(
            engine=engine,
            tokenizer=_ServiceTokenizer(),
            device=torch.device("cpu"),
        ),
        prefill_cache,
        decode_cache,
    )


def test_http_service_uses_one_runtime_for_its_lifespan() -> None:
    async def scenario() -> None:
        runtime, cache = _build_runtime()
        factory_calls = 0

        def runtime_factory() -> InferenceRuntime:
            nonlocal factory_calls
            factory_calls += 1
            return runtime

        app = create_app(runtime_factory)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                health_response = await client.get("/health")
                assert health_response.status_code == 200
                assert health_response.json() == {
                    "status": "ok",
                    "engine_running": True,
                    "engine_closed": False,
                    "pending_requests": 0,
                    "active_requests": 0,
                    "free_blocks": 4,
                }

                generation_response = await client.post(
                    "/generate",
                    json={
                        "prompt": "hello",
                        "sequence_id": "request-a",
                    },
                )
                assert generation_response.status_code == 200
                assert generation_response.json() == {
                    "sequence_id": "request-a",
                    "text": "3",
                    "token_ids": [3, 2],
                    "generated_tokens": 2,
                    "stopped_by_eos": True,
                    "status": "completed",
                }
                invalid_response = await client.post(
                    "/generate",
                    json={"prompt": ""},
                )
                assert invalid_response.status_code == 422
                assert factory_calls == 1

        assert runtime.engine.is_closed
        assert cache.allocator.allocated_block_count == 0

    asyncio.run(scenario())


def test_http_service_accepts_concurrent_requests_with_pd_runtime() -> None:
    async def scenario() -> None:
        runtime, prefill_cache, decode_cache = _build_pd_runtime()
        app = create_app(lambda: runtime)
        transport = httpx.ASGITransport(app=app)

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response_a, response_b = await asyncio.gather(
                    client.post(
                        "/generate",
                        json={
                            "prompt": "hello",
                            "sequence_id": "request-a",
                        },
                    ),
                    client.post(
                        "/generate",
                        json={
                            "prompt": "world",
                            "sequence_id": "request-b",
                        },
                    ),
                )
                assert response_a.status_code == 200
                assert response_b.status_code == 200
                assert response_a.json()["token_ids"] == [0, 0]
                assert response_b.json()["token_ids"] == [0, 0]

                health = (await client.get("/health")).json()
                assert health["pending_requests"] == 0
                assert health["active_requests"] == 0
                assert health["free_blocks"] == 4

        assert runtime.engine.is_closed
        assert prefill_cache.allocator.allocated_block_count == 0
        assert decode_cache.allocator.allocated_block_count == 0

    asyncio.run(scenario())
