from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 20
    eos_token_id: int | None = None
    use_cache: bool = True

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
