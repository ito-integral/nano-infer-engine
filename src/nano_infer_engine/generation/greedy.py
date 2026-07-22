from dataclasses import dataclass

import torch

from nano_infer_engine.cache import KVCache

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

    for step in range(config.max_new_tokens):
        if not config.use_cache:
            logits = model(sequences)

        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

        sequences = torch.cat((sequences, next_token), dim=-1)

        if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
            stopped_by_eos = True
            break

        if config.use_cache and step + 1 < config.max_new_tokens:
            logits = model(next_token, kv_cache=kv_cache)

    return GenerationOutput(
        sequences=sequences,
        generated_tokens=sequences.shape[1] - input_ids.shape[1],
        stopped_by_eos=stopped_by_eos,
    )
