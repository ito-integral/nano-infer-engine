import torch

from nano_infer_engine.paged_cache import PagedKVCache

from .config import GenerationConfig, GenerationOutput
from .paged_prefill import _validate_paged_prefill_inputs, paged_prefill


def _validate_model_cache(model, cache: PagedKVCache, name: str) -> None:
    if model.embed.weight.device != cache.keys.device:
        raise ValueError(f"{name} model and cache must be on the same device")
    if model.embed.weight.dtype != cache.keys.dtype:
        raise ValueError(f"{name} model and cache must have the same dtype")
    if len(model.decoders) != cache.num_layers:
        raise ValueError(f"{name} model and cache must have the same layer count")
    if model.config.kv_head_num != cache.kv_head_num:
        raise ValueError(f"{name} model and cache must have the same KV head count")
    if model.config.head_dim != cache.head_dim:
        raise ValueError(f"{name} model and cache must have the same head dimension")


def _release_if_allocated(cache: PagedKVCache, sequence_id: str) -> None:
    try:
        cache.get_block_table(sequence_id)
    except KeyError:
        return
    cache.release(sequence_id)


@torch.inference_mode()
def pd_greedy_generate(
    prefill_model,
    decode_model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    prefill_cache: PagedKVCache,
    decode_cache: PagedKVCache,
    sequence_id: str = "default",
) -> GenerationOutput:
    """Synchronously prefill, transfer KV, and decode one request.

    The prefill cache is released after handoff. The successful decode cache
    remains allocated so its caller can inspect, continue, or release it.
    """
    if not config.use_cache:
        raise ValueError("P/D generation requires config.use_cache=True")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape (1, prompt_length)")
    if input_ids.shape[1] <= 0:
        raise ValueError("input_ids must contain at least one token")
    if not isinstance(prefill_cache, PagedKVCache):
        raise TypeError("prefill_cache must be a PagedKVCache")
    if not isinstance(decode_cache, PagedKVCache):
        raise TypeError("decode_cache must be a PagedKVCache")
    if prefill_cache is decode_cache:
        raise ValueError("prefill_cache and decode_cache must be different caches")

    _validate_model_cache(prefill_model, prefill_cache, "prefill")
    _validate_model_cache(decode_model, decode_cache, "decode")
    _validate_paged_prefill_inputs(
        (input_ids,),
        prefill_cache,
        (sequence_id,),
    )
    try:
        decode_cache.get_block_table(sequence_id)
    except KeyError:
        pass
    else:
        raise ValueError(f"sequence ID already exists: {sequence_id}")

    cached_token_capacity = input_ids.shape[1] + config.max_new_tokens - 1
    required_decode_blocks = (
        cached_token_capacity + decode_cache.block_size - 1
    ) // decode_cache.block_size
    if required_decode_blocks > decode_cache.allocator.free_block_count:
        raise ValueError("not enough free blocks in decode cache")

    try:
        last_logits = paged_prefill(
            prefill_model,
            (input_ids,),
            prefill_cache,
            (sequence_id,),
        )
        transfer = prefill_cache.export_sequence(sequence_id)
        decode_cache.import_sequence(sequence_id, transfer)
        del transfer
        last_logits = last_logits.to(decode_cache.keys.device)
    except Exception:
        _release_if_allocated(decode_cache, sequence_id)
        raise
    finally:
        _release_if_allocated(prefill_cache, sequence_id)

    sequences = input_ids.to(decode_cache.keys.device)
    generated_tokens = torch.zeros(
        1,
        dtype=torch.long,
        device=decode_cache.keys.device,
    )
    stopped_by_eos = torch.zeros(
        1,
        dtype=torch.bool,
        device=decode_cache.keys.device,
    )

    try:
        for step in range(config.max_new_tokens):
            next_token = last_logits.argmax(dim=-1, keepdim=True)
            sequences = torch.cat((sequences, next_token), dim=1)
            generated_tokens += 1

            if config.eos_token_id is not None:
                stopped_by_eos |= next_token.squeeze(-1).eq(
                    config.eos_token_id
                )
                if stopped_by_eos.all():
                    break

            if step + 1 < config.max_new_tokens:
                logits = decode_model(
                    next_token,
                    kv_cache=decode_cache,
                    sequence_id=sequence_id,
                )
                last_logits = logits[:, -1]
    except Exception:
        _release_if_allocated(decode_cache, sequence_id)
        raise

    return GenerationOutput(
        sequences=sequences,
        generated_tokens=generated_tokens,
        stopped_by_eos=stopped_by_eos,
    )
