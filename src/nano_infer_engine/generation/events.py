from dataclasses import dataclass

from .request import PagedRequest


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
