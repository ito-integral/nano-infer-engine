import torch
from torch import nn

from nano_infer_engine.layers.rope import Rope, RopeScale
from nano_infer_engine.layers.decoder import Llama3Decoder
from nano_infer_engine.layers.rms_norm import RmsNorm
from nano_infer_engine.cache import KVCache
from nano_infer_engine.paged_cache import PagedKVCache


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
        kv_cache: KVCache | PagedKVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        sequence_id: str = "default",
        sequence_ids: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        # x.shape = batch_size, seq_len

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        use_paged_attention = isinstance(kv_cache, PagedKVCache) and seq_len == 1
        paged_sequence_ids: tuple[str, ...] | None = None
        paged_sequence_lengths: tuple[int, ...] | None = None
        if isinstance(kv_cache, KVCache):
            kv_cache.validate_append(seq_len)
            kv_position = kv_cache.position
        elif isinstance(kv_cache, PagedKVCache):
            if sequence_ids is None:
                if batch_size != 1:
                    raise ValueError(
                        "sequence_ids is required for batched paged inference"
                    )
                paged_sequence_ids = (sequence_id,)
            else:
                if not isinstance(sequence_ids, tuple):
                    raise TypeError("sequence_ids must be a tuple")
                if len(sequence_ids) != batch_size:
                    raise ValueError("sequence_ids must match the batch size")
                if any(
                    not isinstance(current_sequence_id, str) or not current_sequence_id
                    for current_sequence_id in sequence_ids
                ):
                    raise ValueError("sequence IDs must be non-empty strings")
                if len(set(sequence_ids)) != len(sequence_ids):
                    raise ValueError("sequence IDs must be unique within a batch")
                paged_sequence_ids = sequence_ids

            paged_sequence_lengths = tuple(
                kv_cache.get_sequence_length(current_sequence_id)
                for current_sequence_id in paged_sequence_ids
            )
            if not use_paged_attention and len(set(paged_sequence_lengths)) != 1:
                raise ValueError(
                    "batched paged prefill currently requires equal sequence lengths"
                )
            kv_position = paged_sequence_lengths[0]
        else:
            kv_position = 0

        if use_paged_attention:
            assert paged_sequence_lengths is not None
            total_seq_len = max(paged_sequence_lengths) + 1
        else:
            total_seq_len = kv_position + seq_len
        if total_seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {total_seq_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        if isinstance(kv_cache, PagedKVCache):
            assert paged_sequence_ids is not None
            assert paged_sequence_lengths is not None
            for batch_index, current_sequence_id in enumerate(paged_sequence_ids):
                required_tokens = (
                    paged_sequence_lengths[batch_index] + 1
                    if use_paged_attention
                    else total_seq_len
                )
                kv_cache.ensure_capacity(current_sequence_id, required_tokens)

        if use_paged_attention:
            if attention_mask is not None:
                if attention_mask.device != x.device:
                    raise ValueError("attention_mask must be on the same device as x")
                if not bool(attention_mask.all()):
                    raise ValueError("paged inference does not support padding masks")
                attention_mask = attention_mask.bool()

            assert paged_sequence_lengths is not None
            paged_position_ids = torch.tensor(
                paged_sequence_lengths,
                dtype=torch.long,
                device=x.device,
            )[:, None]
            if position_ids is None:
                position_ids = paged_position_ids
            else:
                if position_ids.shape != (batch_size, 1):
                    raise ValueError(
                        "position_ids must have shape "
                        f"{(batch_size, 1)}, got {tuple(position_ids.shape)}"
                    )
                if position_ids.device != x.device:
                    raise ValueError("position_ids must be on the same device as x")
                if not torch.equal(position_ids, paged_position_ids):
                    raise ValueError(
                        "position_ids must match paged cache sequence lengths"
                    )
        else:
            # The mask covers both cached keys and current input tokens.
            expected_mask_shape = (batch_size, total_seq_len)
            if attention_mask is None:
                attention_mask = torch.ones(
                    expected_mask_shape,
                    dtype=torch.bool,
                    device=x.device,
                )
            else:
                if attention_mask.shape != expected_mask_shape:
                    raise ValueError(
                        "attention_mask must have shape "
                        f"{expected_mask_shape}, got {tuple(attention_mask.shape)}"
                    )
                if attention_mask.device != x.device:
                    raise ValueError("attention_mask must be on the same device as x")
                attention_mask = attention_mask.bool()

            if isinstance(kv_cache, PagedKVCache) and not bool(attention_mask.all()):
                raise ValueError("paged inference does not support padding masks")

        if position_ids is None:
            assert attention_mask is not None
            # full_position_ids.shape = [batch_size, total_seq_len]
            full_position_ids = attention_mask.long().cumsum(dim=1) - 1
            full_position_ids = full_position_ids.masked_fill(~attention_mask, 0)
            # Only current Q/K tokens need RoPE positions.
            # position_ids.shape = [batch_size, seq_len]
            position_ids = full_position_ids[:, -seq_len:]
        else:
            expected_position_shape = (batch_size, seq_len)
            if position_ids.shape != expected_position_shape:
                raise ValueError(
                    "position_ids must have shape "
                    f"{expected_position_shape}, got {tuple(position_ids.shape)}"
                )
            if position_ids.device != x.device:
                raise ValueError("position_ids must be on the same device as x")

        x = self.embed(x)

        for layer_idx, decoder in enumerate(self.decoders):
            if isinstance(kv_cache, KVCache):
                k_cache, v_cache = kv_cache.get(layer_idx)
            elif isinstance(kv_cache, PagedKVCache) and not use_paged_attention:
                cache_shape = (
                    batch_size,
                    total_seq_len,
                    kv_cache.kv_head_num,
                    kv_cache.head_dim,
                )
                k_cache = torch.empty(
                    cache_shape,
                    dtype=kv_cache.keys.dtype,
                    device=kv_cache.keys.device,
                )
                v_cache = torch.empty_like(k_cache)

                if kv_position > 0:
                    assert paged_sequence_ids is not None
                    for batch_index, current_sequence_id in enumerate(
                        paged_sequence_ids
                    ):
                        previous_keys, previous_values = kv_cache.gather(
                            layer_idx,
                            current_sequence_id,
                            kv_position,
                        )
                        k_cache[batch_index, :kv_position].copy_(previous_keys)
                        v_cache[batch_index, :kv_position].copy_(previous_values)
            else:
                k_cache, v_cache = None, None

            x = decoder(
                x,
                k_cache,
                v_cache,
                kv_position,
                attention_mask,
                position_ids,
                paged_cache=kv_cache if use_paged_attention else None,
                layer_index=layer_idx if use_paged_attention else None,
                sequence_id=sequence_id,
                sequence_ids=paged_sequence_ids if use_paged_attention else None,
            )

            if isinstance(kv_cache, PagedKVCache) and not use_paged_attention:
                assert paged_sequence_ids is not None
                for batch_index, current_sequence_id in enumerate(paged_sequence_ids):
                    kv_cache.write(
                        layer_idx,
                        current_sequence_id,
                        kv_position,
                        k_cache[batch_index, kv_position:total_seq_len],
                        v_cache[batch_index, kv_position:total_seq_len],
                    )

        x = self.final_rms(x)
        x = self.lm_head(x)

        if isinstance(kv_cache, KVCache):
            kv_cache.advance(seq_len)
        elif isinstance(kv_cache, PagedKVCache):
            assert paged_sequence_ids is not None
            for current_sequence_id in paged_sequence_ids:
                kv_cache.advance(current_sequence_id, seq_len)

        return x

    def forward_ragged(
        self,
        input_ids: torch.Tensor,
        *,
        kv_cache: PagedKVCache,
        sequence_ids: tuple[str, ...],
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Run a flattened variable-query prefill over a paged KV cache."""
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a non-empty 1D tensor")
        if not isinstance(kv_cache, PagedKVCache):
            raise TypeError("kv_cache must be a PagedKVCache")
        if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("sequence_ids must be non-empty and unique")
        if query_start_loc.ndim != 1 or query_start_loc.numel() != len(sequence_ids) + 1:
            raise ValueError("query_start_loc must have one boundary per sequence")
        if query_start_loc.device != input_ids.device:
            raise ValueError("query_start_loc must be on the input device")
        boundaries = query_start_loc.tolist()
        if boundaries[0] != 0 or boundaries[-1] != input_ids.numel():
            raise ValueError("query_start_loc must span all input tokens")
        if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
            raise ValueError("each ragged query must contain at least one token")

        context_lengths = tuple(
            kv_cache.get_sequence_length(sequence_id) for sequence_id in sequence_ids
        )
        query_lengths = tuple(
            end - start for start, end in zip(boundaries, boundaries[1:])
        )
        for sequence_id, context_length, query_length in zip(
            sequence_ids, context_lengths, query_lengths
        ):
            sequence_length = context_length + query_length
            if sequence_length > self.config.max_seq_len:
                raise ValueError(
                    f"Sequence length {sequence_length} exceeds "
                    f"max_seq_len={self.config.max_seq_len}"
                )
            kv_cache.ensure_capacity(sequence_id, sequence_length)

        x = self.embed(input_ids)
        for layer_index, decoder in enumerate(self.decoders):
            x = decoder.forward_ragged(
                x,
                paged_cache=kv_cache,
                layer_index=layer_index,
                sequence_ids=sequence_ids,
                query_start_loc=query_start_loc,
                context_lengths=context_lengths,
            )
        x = self.lm_head(self.final_rms(x))
        for sequence_id, query_length in zip(sequence_ids, query_lengths):
            kv_cache.advance(sequence_id, query_length)
        return x
