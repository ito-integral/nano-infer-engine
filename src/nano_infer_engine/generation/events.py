from dataclasses import dataclass

import torch

from .request import PagedRequest, RequestStatus


@dataclass(frozen=True)
class TokenEvent:
    """One token generated for a request during a scheduler step."""

    sequence_id: str
    token_id: int


@dataclass(frozen=True)
class SchedulerStepOutput:
    """Tokens and terminal requests produced by one scheduler step."""

    token_events: tuple[TokenEvent, ...]
    terminal_requests: tuple[PagedRequest, ...]


@dataclass(frozen=True)
class RequestResult:
    """Immutable terminal result exposed by the asynchronous engine."""

    sequence_id: str
    status: RequestStatus
    sequence: torch.Tensor
    generated_tokens: int
    stopped_by_eos: bool
    error: Exception | None
