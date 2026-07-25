import torch

from nano_infer_engine.paged_cache import PagedKVCache


@torch.inference_mode()
def paged_prefill(
    model,
    prompts: tuple[torch.Tensor, ...],
    paged_cache: PagedKVCache,
    sequence_ids: tuple[str, ...],
) -> torch.Tensor:
    """Prefill variable-length prompts without storing padding tokens."""
    if not isinstance(paged_cache, PagedKVCache):
        raise TypeError("paged_cache must be a PagedKVCache")
    if not isinstance(prompts, tuple):
        raise TypeError("prompts must be a tuple")
    if not isinstance(sequence_ids, tuple):
        raise TypeError("sequence_ids must be a tuple")
    if not prompts:
        raise ValueError("prompts must not be empty")
    if len(prompts) != len(sequence_ids):
        raise ValueError("prompts and sequence_ids must have the same length")

    if any(
        not isinstance(sequence_id, str) or not sequence_id
        for sequence_id in sequence_ids
    ):
        raise ValueError("sequence IDs must be non-empty strings")
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("sequence IDs must be unique")

    for prompt in prompts:
        if not isinstance(prompt, torch.Tensor):
            raise TypeError("each prompt must be a torch.Tensor")
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("each prompt must have shape (1, prompt_length)")
        if prompt.shape[1] <= 0:
            raise ValueError("each prompt must contain at least one token")
        if prompt.device != paged_cache.keys.device:
            raise ValueError("each prompt must be on the paged cache device")

    for sequence_id in sequence_ids:
        try:
            paged_cache.get_block_table(sequence_id)
        except KeyError:
            continue
        raise ValueError(f"sequence ID already exists: {sequence_id}")

    required_blocks = sum(
        (prompt.shape[1] + paged_cache.block_size - 1) // paged_cache.block_size
        for prompt in prompts
    )
    if required_blocks > paged_cache.allocator.free_block_count:
        raise ValueError("not enough free blocks")

    logits_list: list[torch.Tensor] = []

    for prompt, sequence_id in zip(prompts, sequence_ids):
        logits = model(
            prompt,
            kv_cache=paged_cache,
            sequence_id=sequence_id,
        )
        logits_list.append(logits[:, -1])

    return torch.cat(logits_list, dim=0)
