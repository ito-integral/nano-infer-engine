import torch

from nano_infer_engine.paged_cache import PagedKVCache


def _validate_paged_prefill_inputs(
    prompts: tuple[torch.Tensor, ...],
    paged_cache: PagedKVCache,
    sequence_ids: tuple[str, ...],
) -> None:
    """Validate prefill inputs without mutating the paged cache."""
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


@torch.inference_mode()
def paged_prefill(
    model,
    prompts: tuple[torch.Tensor, ...],
    paged_cache: PagedKVCache,
    sequence_ids: tuple[str, ...],
) -> torch.Tensor:
    """Prefill variable-length prompts without storing padding tokens."""
    _validate_paged_prefill_inputs(prompts, paged_cache, sequence_ids)

    required_blocks = sum(
        (prompt.shape[1] + paged_cache.block_size - 1) // paged_cache.block_size
        for prompt in prompts
    )
    if required_blocks > paged_cache.allocator.free_block_count:
        raise ValueError("not enough free blocks")

    logits_by_index: list[torch.Tensor | None] = [None] * len(prompts)
    groups: dict[int, list[int]] = {}
    for prompt_index, prompt in enumerate(prompts):
        groups.setdefault(prompt.shape[1], []).append(prompt_index)

    try:
        for prompt_indices in groups.values():
            batched_prompts = torch.cat(
                [prompts[prompt_index] for prompt_index in prompt_indices],
                dim=0,
            )
            batched_sequence_ids = tuple(
                sequence_ids[prompt_index] for prompt_index in prompt_indices
            )
            if len(batched_sequence_ids) == 1:
                logits = model(
                    batched_prompts,
                    kv_cache=paged_cache,
                    sequence_id=batched_sequence_ids[0],
                )
            else:
                logits = model(
                    batched_prompts,
                    kv_cache=paged_cache,
                    sequence_ids=batched_sequence_ids,
                )
            last_logits = logits[:, -1]
            for batch_index, prompt_index in enumerate(prompt_indices):
                logits_by_index[prompt_index] = last_logits[batch_index]
    except Exception:
        # A batched model call can mutate several cache entries before failing.
        # Restore the all-or-nothing behavior expected by scheduler admission.
        for sequence_id in sequence_ids:
            try:
                paged_cache.release(sequence_id)
            except KeyError:
                pass
        raise

    if any(logits is None for logits in logits_by_index):
        raise RuntimeError("prefill did not produce logits for every prompt")
    return torch.stack([logits for logits in logits_by_index if logits is not None])


@torch.inference_mode()
def paged_prefill_chunks(
    model,
    chunks: tuple[torch.Tensor, ...],
    paged_cache: PagedKVCache,
    sequence_ids: tuple[str, ...],
) -> torch.Tensor:
    """Append flattened variable-length chunks and return their last logits."""
    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(chunks) != len(sequence_ids):
        raise ValueError("chunks and sequence_ids must have the same length")
    for chunk in chunks:
        if not isinstance(chunk, torch.Tensor):
            raise TypeError("each chunk must be a torch.Tensor")
        if chunk.ndim != 2 or chunk.shape[0] != 1 or chunk.shape[1] <= 0:
            raise ValueError("each chunk must have shape (1, chunk_length)")
        if chunk.device != paged_cache.keys.device:
            raise ValueError("each chunk must be on the paged cache device")
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("sequence IDs must be unique")

    flat_input_ids = torch.cat([chunk[0] for chunk in chunks])
    query_lengths = torch.tensor(
        [chunk.shape[1] for chunk in chunks],
        dtype=torch.long,
        device=flat_input_ids.device,
    )
    query_start_loc = torch.zeros(
        len(chunks) + 1,
        dtype=torch.long,
        device=flat_input_ids.device,
    )
    query_start_loc[1:] = query_lengths.cumsum(dim=0)
    try:
        forward_ragged = model.forward_ragged
    except AttributeError:
        raise TypeError("model must implement forward_ragged") from None
    logits = forward_ragged(
        flat_input_ids,
        kv_cache=paged_cache,
        sequence_ids=sequence_ids,
        query_start_loc=query_start_loc,
    )
    last_token_indices = query_start_loc[1:] - 1
    return logits[last_token_indices]
