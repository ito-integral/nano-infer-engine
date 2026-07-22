import torch
from torch import nn

from nano_infer_engine.layers.rope import Rope, RopeScale
from nano_infer_engine.layers.decoder import Llama3Decoder
from nano_infer_engine.layers.rms_norm import RmsNorm
from nano_infer_engine.cache import KVCache


from dataclasses import dataclass, field


@dataclass
class RopeScalingConfig:
    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass
class LlamaConfig:
    # Vocabulary
    vocab_size: int = 128256

    # Transformer
    hidden_size: int = 2048
    mlp_inner_size: int = 8192
    num_layers: int = 16

    # GQA
    q_head_num: int = 32
    kv_head_num: int = 8

    # RMSNorm
    norm_eps: float = 1e-5

    # RoPE
    rope_type: str = "llama3"
    rope_theta: float = 500000.0
    max_seq_len: int = 131072
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    # Model architecture
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True

    # Initialization
    initializer_range: float = 0.02

    # Token IDs
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    def __post_init__(self):
        assert self.hidden_size % self.q_head_num == 0
        assert self.q_head_num % self.kv_head_num == 0
        assert self.head_dim % 2 == 0

    @property
    def head_dim(self):
        return self.hidden_size // self.q_head_num

    @property
    def group_num(self):
        return self.q_head_num // self.kv_head_num


def build_rope(config):
    if config.rope_type == "default":
        return Rope(
            dim=config.head_dim,
            theta_base=config.rope_theta,
        )

    if config.rope_type == "llama3":
        return RopeScale(
            dim=config.head_dim,
            theta_base=config.rope_theta,
            scaling_factor=config.rope_scaling.factor,
            low_freq_factor=config.rope_scaling.low_freq_factor,
            high_freq_factor=config.rope_scaling.high_freq_factor,
            original_max_position_embeddings=(
                config.rope_scaling.original_max_position_embeddings
            ),
        )

    raise ValueError(f"Unsupported rope type: {config.rope_type}")


class Llama3_2(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = build_rope(config)

        self.decoders = nn.ModuleList(
            [
                Llama3Decoder(
                    config.norm_eps,
                    config.q_head_num,
                    config.kv_head_num,
                    config.hidden_size,
                    config.mlp_inner_size,
                    self.rope,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_rms = RmsNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
            assert self.lm_head.weight is self.embed.weight

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        # x.shape = batch_size, seq_len

        seq_len = x.shape[1]
        if kv_cache is not None:
            kv_cache.validate_append(seq_len)
            kv_position = kv_cache.position
        else:
            kv_position = 0

        total_seq_len = kv_position + seq_len
        if total_seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {total_seq_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        x = self.embed(x)

        for layer_idx, decoder in enumerate(self.decoders):
            if kv_cache is not None:
                k_cache, v_cache = kv_cache.get(layer_idx)
            else:
                k_cache, v_cache = None, None

            x = decoder(x, k_cache, v_cache, kv_position)

        x = self.final_rms(x)
        x = self.lm_head(x)

        if kv_cache is not None:
            kv_cache.advance(seq_len)

        return x
