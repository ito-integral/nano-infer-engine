import torch


def prefill(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kv_cache,
) -> torch.Tensor:
    """Run a one-shot prompt prefill and return its logits."""
    return model(
        input_ids,
        kv_cache=kv_cache,
        attention_mask=attention_mask,
    )
