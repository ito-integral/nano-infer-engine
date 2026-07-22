from dataclasses import dataclass

import torch

from .config import GenerationConfig


@dataclass
class GenerationOutput:
    sequences: torch.Tensor
    generated_tokens: int
    stopped_by_eos: bool


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> GenerationOutput:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be a 2D tensor of shape (batch_size, seq_len)")

    if input_ids.shape[0] != 1:
        raise ValueError("greedy_generate currently supports batch_size=1")

    sequences = input_ids
    stopped_by_eos = False

    if config.use_cache:
        cache_capacity = input_ids.shape[1] + config.max_new_tokens
        logits, k_caches, v_caches = model(
            sequences,
            use_cache=True,
            cache_position=0,
            cache_capacity=cache_capacity,
        )
        cache_position = input_ids.shape[1]

    for step in range(config.max_new_tokens):
        if not config.use_cache:
            logits = model(sequences)

        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

        sequences = torch.cat((sequences, next_token), dim=-1)

        if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
            stopped_by_eos = True
            break

        if config.use_cache and step + 1 < config.max_new_tokens:
            logits, k_caches, v_caches = model(
                next_token,
                k_caches=k_caches,
                v_caches=v_caches,
                use_cache=True,
                cache_position=cache_position,
            )
            cache_position += 1

    return GenerationOutput(
        sequences=sequences,
        generated_tokens=sequences.shape[1] - input_ids.shape[1],
        stopped_by_eos=stopped_by_eos,
    )
