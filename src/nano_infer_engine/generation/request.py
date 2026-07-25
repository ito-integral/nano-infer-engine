from dataclasses import dataclass, field

import torch


@dataclass
class PagedRequest:
    """Mutable generation state for one request in a paged batch."""

    sequence_id: str
    prompt: torch.Tensor
    sequence: torch.Tensor = field(init=False)
    generated_tokens: int = 0
    finished: bool = False
    required_blocks: int = 0
    last_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.sequence = self.prompt
