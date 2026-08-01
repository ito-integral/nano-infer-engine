from dataclasses import dataclass, field
from enum import Enum

import torch


class RequestStatus(str, Enum):
    """Lifecycle state of a paged generation request."""

    PENDING = "pending"
    PREFILLING = "prefilling"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    prefill_offset: int = 0
    status: RequestStatus = RequestStatus.PENDING
    error: Exception | None = None

    def __post_init__(self) -> None:
        self.sequence = self.prompt
