from collections import deque

import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig, RaggedGenerationOutput
from .paged_prefill import _validate_paged_prefill_inputs, paged_prefill
from .request import PagedRequest


@torch.inference_mode()
def paged_greedy_generate(
    model,
    prompts: tuple[torch.Tensor, ...],
    config: GenerationConfig,
    paged_cache: PagedKVCache,
    sequence_ids: tuple[str, ...],
    *,
    max_batch_size: int | None = None,
) -> RaggedGenerationOutput:
    """Generate from a FIFO queue using a shared paged KV cache."""
    if not config.use_cache:
        raise ValueError("paged greedy generation requires config.use_cache=True")
    if max_batch_size is not None and (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or max_batch_size <= 0
    ):
        raise ValueError("max_batch_size must be a positive integer")

    _validate_paged_prefill_inputs(prompts, paged_cache, sequence_ids)
    continuous_batching = max_batch_size is not None
    batch_limit = max_batch_size or len(prompts)
    block_budget = paged_cache.allocator.free_block_count

    requests = []
    for prompt, sequence_id in zip(prompts, sequence_ids):
        required_blocks = (
            prompt.shape[1]
            + config.max_new_tokens
            - 1
            + paged_cache.block_size
            - 1
        ) // paged_cache.block_size
        if required_blocks > block_budget:
            raise ValueError(
                f"request cannot fit in paged cache: {sequence_id}"
            )
        requests.append(
            PagedRequest(
                sequence_id=sequence_id,
                prompt=prompt,
                required_blocks=required_blocks,
            )
        )

    if not continuous_batching and sum(
        request.required_blocks for request in requests
    ) > block_budget:
        raise ValueError("not enough free blocks for paged generation")

    pending_requests = deque(requests)
    active_requests: list[PagedRequest] = []
    reserved_blocks = 0
    device = prompts[0].device

    def admit_pending_requests() -> None:
        nonlocal reserved_blocks

        while pending_requests and len(active_requests) < batch_limit:
            request = pending_requests[0]
            if reserved_blocks + request.required_blocks > block_budget:
                break

            pending_requests.popleft()
            logits = paged_prefill(
                model,
                (request.prompt,),
                paged_cache,
                (request.sequence_id,),
            )
            request.last_logits = logits[0]
            active_requests.append(request)
            reserved_blocks += request.required_blocks

    admit_pending_requests()

    while active_requests:
        last_logits = torch.stack(
            [request.last_logits for request in active_requests]
        )
        next_tokens = last_logits.argmax(dim=-1, keepdim=True)

        for local_index, request in enumerate(active_requests):
            request.generated_tokens += 1
            request.sequence = torch.cat(
                (
                    request.sequence,
                    next_tokens[local_index : local_index + 1],
                ),
                dim=1,
            )

        if config.eos_token_id is not None:
            finished_now = next_tokens.squeeze(-1).eq(config.eos_token_id)
        else:
            finished_now = torch.zeros(
                len(active_requests),
                dtype=torch.bool,
                device=device,
            )

        survivor_local_indices: list[int] = []
        survivor_requests: list[PagedRequest] = []
        for local_index, request in enumerate(active_requests):
            stopped_by_eos = bool(finished_now[local_index])
            reached_token_limit = (
                request.generated_tokens >= config.max_new_tokens
            )
            if stopped_by_eos or reached_token_limit:
                request.finished = stopped_by_eos
                reserved_blocks -= request.required_blocks
                if stopped_by_eos or continuous_batching:
                    paged_cache.release(request.sequence_id)
                continue

            survivor_local_indices.append(local_index)
            survivor_requests.append(request)

        active_requests = survivor_requests
        if active_requests:
            active_next_tokens = next_tokens[survivor_local_indices]
            active_sequence_ids = tuple(
                request.sequence_id for request in active_requests
            )
            logits = model(
                active_next_tokens,
                kv_cache=paged_cache,
                sequence_ids=active_sequence_ids,
            )
            for local_index, request in enumerate(active_requests):
                request.last_logits = logits[local_index, -1]

        admit_pending_requests()

    if pending_requests:
        raise RuntimeError("pending requests cannot be admitted")

    return RaggedGenerationOutput(
        sequences=tuple(request.sequence for request in requests),
        generated_tokens=torch.tensor(
            [request.generated_tokens for request in requests],
            dtype=torch.long,
            device=device,
        ),
        stopped_by_eos=torch.tensor(
            [request.finished for request in requests],
            dtype=torch.bool,
            device=device,
        ),
    )
