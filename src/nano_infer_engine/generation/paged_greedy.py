import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig, RaggedGenerationOutput
from .paged_prefill import paged_prefill
from .request import PagedRequest


@torch.inference_mode()
def paged_greedy_generate(
    model,
    prompts: tuple[torch.Tensor, ...],
    config: GenerationConfig,
    paged_cache: PagedKVCache,
    sequence_ids: tuple[str, ...],
) -> RaggedGenerationOutput:
    """Generate from variable-length prompts using a shared paged KV cache."""
    if not config.use_cache:
        raise ValueError("paged greedy generation requires config.use_cache=True")

    # Check worst-case cache usage before mutating state. The final generated
    # token is returned without being decoded again, so it needs no cache slot.
    if (
        isinstance(prompts, tuple)
        and prompts
        and all(
            isinstance(prompt, torch.Tensor)
            and prompt.ndim == 2
            and prompt.shape[0] == 1
            for prompt in prompts
        )
    ):
        required_blocks = sum(
            (prompt.shape[1] + config.max_new_tokens - 1 + paged_cache.block_size - 1)
            // paged_cache.block_size
            for prompt in prompts
        )
        if required_blocks > paged_cache.allocator.free_block_count:
            raise ValueError("not enough free blocks for paged generation")

    last_logits = paged_prefill(
        model,
        prompts,
        paged_cache,
        sequence_ids,
    )
    requests = [
        PagedRequest(sequence_id=sequence_id, prompt=prompt)
        for prompt, sequence_id in zip(prompts, sequence_ids)
    ]
    active_requests = list(requests)
    device = prompts[0].device

    for step in range(config.max_new_tokens):
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

        survivor_local_indices = []
        survivor_requests = []
        for local_index, request in enumerate(active_requests):
            if bool(finished_now[local_index]):
                request.finished = True
                paged_cache.release(request.sequence_id)
            else:
                survivor_local_indices.append(local_index)
                survivor_requests.append(request)

        if not survivor_requests:
            break

        if step + 1 < config.max_new_tokens:
            active_requests = survivor_requests
            active_next_tokens = next_tokens[survivor_local_indices]
            active_sequence_ids = tuple(
                request.sequence_id for request in active_requests
            )
            logits = model(
                active_next_tokens,
                kv_cache=paged_cache,
                sequence_ids=active_sequence_ids,
            )
            last_logits = logits[:, -1]

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
