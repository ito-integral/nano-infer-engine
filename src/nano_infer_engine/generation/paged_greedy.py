import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig, RaggedGenerationOutput
from .paged_prefill import _validate_paged_prefill_inputs
from .request import RequestStatus
from .scheduler import ContinuousBatchingScheduler


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
    batch_limit = max_batch_size or len(prompts)
    block_budget = paged_cache.allocator.free_block_count

    required_blocks = []
    for prompt, sequence_id in zip(prompts, sequence_ids):
        request_blocks = (
            prompt.shape[1]
            + config.max_new_tokens
            - 1
            + paged_cache.block_size
            - 1
        ) // paged_cache.block_size
        if request_blocks > block_budget:
            raise ValueError(
                f"request cannot fit in paged cache: {sequence_id}"
            )
        required_blocks.append(request_blocks)

    if max_batch_size is None and sum(required_blocks) > block_budget:
        raise ValueError("not enough free blocks for paged generation")

    scheduler = ContinuousBatchingScheduler(
        model,
        config,
        paged_cache,
        batch_limit,
        release_on_token_limit=max_batch_size is not None,
    )
    requests = [
        scheduler.add_request(sequence_id, prompt)
        for prompt, sequence_id in zip(prompts, sequence_ids)
    ]
    scheduler.run_until_idle()
    failed_request = next(
        (
            request
            for request in requests
            if request.status is RequestStatus.FAILED
        ),
        None,
    )
    if failed_request is not None:
        raise RuntimeError(
            f"paged prefill failed: {failed_request.sequence_id}"
        ) from failed_request.error
    device = prompts[0].device

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
