import torch
from torch import nn

from nano_infer_engine.layers.rope import Rope, RopeScale
from nano_infer_engine.layers.decoder import Llama3Decoder
from nano_infer_engine.layers.rms_norm import RmsNorm


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
        x,
        k_caches=None,
        v_caches=None,
        use_cache=False,
        cache_position=0,
        cache_capacity=None,
    ):
        # x.shape = batch_size, seq_len
        if (k_caches is None) != (v_caches is None):
            raise ValueError(
                "k_caches and v_caches must both be provided or both be None"
            )

        if k_caches is None and use_cache:
            if cache_capacity is None:
                raise ValueError("cache_capacity is required when creating a KV cache")
            if cache_capacity < cache_position + x.shape[1]:
                raise ValueError("cache_capacity is too small")
            cache_shape = (
                x.shape[0],
                cache_capacity,
                self.config.kv_head_num,
                self.config.head_dim,
            )
            k_caches = [
                torch.empty(
                    cache_shape,
                    dtype=self.embed.weight.dtype,
                    device=x.device,
                )
                for _ in self.decoders
            ]
            v_caches = [
                torch.empty(
                    cache_shape,
                    dtype=self.embed.weight.dtype,
                    device=x.device,
                )
                for _ in self.decoders
            ]
        elif k_caches is None:
            k_caches = [None] * len(self.decoders)
            v_caches = [None] * len(self.decoders)
        else:
            use_cache = True
            if len(k_caches) != len(self.decoders):
                raise ValueError("k_caches length must equal the number of layers")
            if len(v_caches) != len(self.decoders):
                raise ValueError("v_caches length must equal the number of layers")

        seq_len = x.shape[1]
        total_seq_len = cache_position + seq_len
        if total_seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {total_seq_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        x = self.embed(x)
        new_k_caches = []
        new_v_caches = []
        for decoder, k_cache, v_cache in zip(
            self.decoders,
            k_caches,
            v_caches,
        ):
            x, new_k_cache, new_v_cache = decoder(
                x,
                k_cache,
                v_cache,
                cache_position,
            )
            new_k_caches.append(new_k_cache)
            new_v_caches.append(new_v_cache)

        x = self.final_rms(x)
        x = self.lm_head(x)

        if use_cache:
            return x, new_k_caches, new_v_caches
        return x
