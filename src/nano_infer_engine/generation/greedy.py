from dataclasses import dataclass

import torch

from nano_infer_engine.cache import KVCache

from .config import GenerationConfig


@dataclass
class GenerationOutput:
    sequences: torch.Tensor
    generated_tokens: torch.Tensor
    stopped_by_eos: torch.Tensor


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
) -> GenerationOutput:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be a 2D tensor of shape (batch_size, seq_len)")

    sequences = input_ids

    if config.use_cache:
        kv_cache = KVCache(
            num_layers=len(model.decoders),
            batch_size=input_ids.shape[0],
            capacity=input_ids.shape[1] + config.max_new_tokens,
            kv_head_num=model.config.kv_head_num,
            head_dim=model.config.head_dim,
            dtype=model.embed.weight.dtype,
            device=input_ids.device,
        )
        logits = model(sequences, kv_cache=kv_cache)

    finished = torch.zeros(
        input_ids.shape[0],
        dtype=torch.bool,
        device=input_ids.device,
    )

    generated_tokens = torch.zeros(
        input_ids.shape[0],
        dtype=torch.long,
        device=input_ids.device,
    )

    for step in range(config.max_new_tokens):
        if not config.use_cache:
            logits = model(sequences)

        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

        # 如果已经是eos的后续还是拼接eos
        if config.eos_token_id is not None:
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, config.eos_token_id),
                next_token,
            )

        generated_tokens += (~finished).long()
        sequences = torch.cat((sequences, next_token), dim=-1)

        if config.eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(config.eos_token_id)

        if finished.all():
            break

        if config.use_cache and step + 1 < config.max_new_tokens:
            logits = model(next_token, kv_cache=kv_cache)

    return GenerationOutput(
        sequences=sequences,
        generated_tokens=generated_tokens,
        stopped_by_eos=finished,
    )
