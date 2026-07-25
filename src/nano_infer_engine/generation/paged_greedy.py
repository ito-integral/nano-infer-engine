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

    for step in range(config.max_new_tokens):
        next_tokens = last_logits.argmax(dim=-1, keepdim=True)

        if config.eos_token_id is not None:
            next_tokens = torch.where(
                finished[:, None],
                torch.full_like(next_tokens, config.eos_token_id),
                next_tokens,
            )

        generated_tokens += (~finished).long()
        sequences = [
            torch.cat((sequence, next_tokens[index : index + 1]), dim=1)
            for index, sequence in enumerate(sequences)
        ]

        if config.eos_token_id is not None:
            finished |= next_tokens.squeeze(-1).eq(config.eos_token_id)
        if finished.all():
            break

        if step + 1 < config.max_new_tokens:
            logits = model(
                next_tokens,
                kv_cache=paged_cache,
                sequence_ids=sequence_ids,
            )
            last_logits = logits[:, -1]

    return RaggedGenerationOutput(
        sequences=tuple(sequences),
        generated_tokens=generated_tokens,
        stopped_by_eos=finished,
    )
