import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig, RaggedGenerationOutput
from .paged_prefill import paged_prefill


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
    sequences = list(prompts)
    request_count = len(prompts)
    device = prompts[0].device

    finished = torch.zeros(request_count, dtype=torch.bool, device=device)
    generated_tokens = torch.zeros(request_count, dtype=torch.long, device=device)
    active_indices = list(range(request_count))

    for step in range(config.max_new_tokens):
        next_tokens = last_logits.argmax(dim=-1, keepdim=True)

        for local_index, original_index in enumerate(active_indices):
            generated_tokens[original_index] += 1
            sequences[original_index] = torch.cat(
                (
                    sequences[original_index],
                    next_tokens[local_index : local_index + 1],
                ),
                dim=1,
            )

        if config.eos_token_id is not None:
            finished_now = next_tokens.squeeze(-1).eq(config.eos_token_id)
        else:
            finished_now = torch.zeros(
                len(active_indices),
                dtype=torch.bool,
                device=device,
            )

        survivor_local_indices = []
        for local_index, original_index in enumerate(active_indices):
            if bool(finished_now[local_index]):
                finished[original_index] = True
                paged_cache.release(sequence_ids[original_index])
            else:
                survivor_local_indices.append(local_index)

        if not survivor_local_indices:
            break

        if step + 1 < config.max_new_tokens:
            active_indices = [
                active_indices[local_index] for local_index in survivor_local_indices
            ]
            active_next_tokens = next_tokens[survivor_local_indices]
            active_sequence_ids = tuple(
                sequence_ids[original_index] for original_index in active_indices
            )
            logits = model(
                active_next_tokens,
                kv_cache=paged_cache,
                sequence_ids=active_sequence_ids,
            )
            last_logits = logits[:, -1]

    return RaggedGenerationOutput(
        sequences=tuple(sequences),
        generated_tokens=generated_tokens,
        stopped_by_eos=finished,
    )
