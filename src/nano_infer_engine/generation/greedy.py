import torch

from nano_infer_engine.cache import KVCache

from .config import GenerationConfig, GenerationOutput
from .prefill import prefill

@torch.inference_mode()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    attention_mask: torch.Tensor | None = None,
) -> GenerationOutput:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be a 2D tensor of shape (batch_size, seq_len)")
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one sequence and one token")

    if attention_mask is None:
        # [batch_size, prompt_seq_len]
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if attention_mask.device != input_ids.device:
        raise ValueError("attention_mask must be on the same device as input_ids")

    attention_mask = attention_mask.bool()
    if not attention_mask.any(dim=1).all():
        raise ValueError("each sequence must contain at least one non-padding token")
    if (attention_mask[:, 1:] < attention_mask[:, :-1]).any():
        raise ValueError("variable-length batches require left padding")

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
        logits = prefill(
            model,
            sequences,
            attention_mask,
            kv_cache,
        )

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
            logits = model(sequences, attention_mask=attention_mask)

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

        if step + 1 < config.max_new_tokens:
            # Append one valid position for the token generated in this step.
            # [batch_size, total_seq_len] -> [batch_size, total_seq_len + 1]
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=torch.bool,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )

            if config.use_cache:
                logits = model(
                    next_token,
                    kv_cache=kv_cache,
                    attention_mask=attention_mask,
                )

    return GenerationOutput(
        sequences=sequences,
        generated_tokens=generated_tokens,
        stopped_by_eos=finished,
    )
