from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 20
    eos_token_id: int | None = None
    use_cache: bool = True
    prefill_chunk_size: int | None = None
    max_prefill_tokens_per_step: int | None = None

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.prefill_chunk_size is not None:
            if not isinstance(self.prefill_chunk_size, int) or isinstance(
                self.prefill_chunk_size, bool
            ):
                raise TypeError("prefill_chunk_size must be an integer or None")
            if self.prefill_chunk_size <= 0:
                raise ValueError("prefill_chunk_size must be positive")
        if self.max_prefill_tokens_per_step is not None:
            if not isinstance(self.max_prefill_tokens_per_step, int) or isinstance(
                self.max_prefill_tokens_per_step, bool
            ):
                raise TypeError(
                    "max_prefill_tokens_per_step must be an integer or None"
                )
            if self.max_prefill_tokens_per_step <= 0:
                raise ValueError("max_prefill_tokens_per_step must be positive")
            if self.prefill_chunk_size is None:
                raise ValueError(
                    "max_prefill_tokens_per_step requires prefill_chunk_size"
                )


@dataclass
class GenerationOutput:
    sequences: torch.Tensor
    generated_tokens: torch.Tensor
    stopped_by_eos: torch.Tensor


@dataclass
class RaggedGenerationOutput:
    sequences: tuple[torch.Tensor, ...]
    generated_tokens: torch.Tensor
    stopped_by_eos: torch.Tensor
