import torch
from torch import nn
import math


def _apply_rotary_emb(x, sin, cos):
    """Apply Llama-style RoPE by pairing the two halves of the head dimension."""
    x_first, x_second = x.chunk(2, dim=-1)

    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]

    output_first = x_first * cos - x_second * sin
    output_second = x_second * cos + x_first * sin
    return torch.cat((output_first, output_second), dim=-1)


class Rope(nn.Module):
    def __init__(self, dim, theta_base=500000):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0
        self.theta_base = theta_base

    def _build_rope_sin_cos(self, seq_len, device, dtype):
        # Calculate positions and frequencies in FP32 for stable long-context RoPE.
        i = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1 / self.theta_base ** (i / self.dim)

        position = torch.arange(0, seq_len, dtype=torch.float32, device=device)
        angles = torch.einsum("i,j->ij", position, inv_freq)
        # seq, dim/2
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    def forward(self, x):
        # x.shape = [B, seq_len, H, head_dim]
        bs, seq_len, head_num, head_dim = x.shape

        assert head_dim == self.dim

        sin_, cos_ = self._build_rope_sin_cos(seq_len, x.device, x.dtype)

        return _apply_rotary_emb(x, sin_, cos_)


class RopeScale(nn.Module):
    def __init__(
        self,
        dim,
        theta_base=500000.0,
        scaling_factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()

        self.dim = dim
        assert self.dim % 2 == 0

        self.theta_base = theta_base
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings

    def _build_rope_sin_cos(self, seq_len, device, dtype):
        # RoPE 的频率和角度使用 FP32 计算
        i = torch.arange(
            0,
            self.dim,
            2,
            dtype=torch.float32,
            device=device,
        )

        inv_freq = 1 / self.theta_base ** (i / self.dim)

        # Llama 3.2 的频率分段缩放
        wavelen = 2 * math.pi / inv_freq

        low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor

        high_freq_wavelen = (
            self.original_max_position_embeddings / self.high_freq_factor
        )

        # 低频部分：波长很长，频率除以 scaling_factor
        scaled_inv_freq = inv_freq / self.scaling_factor

        # 中频部分：在原始频率和缩放频率之间平滑插值
        smooth_factor = (
            self.original_max_position_embeddings / wavelen - self.low_freq_factor
        ) / (self.high_freq_factor - self.low_freq_factor)

        smooth_inv_freq = (
            1 - smooth_factor
        ) * scaled_inv_freq + smooth_factor * inv_freq

        is_low_freq = wavelen > low_freq_wavelen
        is_high_freq = wavelen < high_freq_wavelen
        is_medium_freq = ~(is_low_freq | is_high_freq)

        inv_freq = torch.where(
            is_low_freq,
            scaled_inv_freq,
            inv_freq,
        )

        inv_freq = torch.where(
            is_medium_freq,
            smooth_inv_freq,
            inv_freq,
        )

        position = torch.arange(
            0,
            seq_len,
            dtype=torch.float32,
            device=device,
        )

        angles = torch.einsum(
            "i,j->ij",
            position,
            inv_freq,
        )

        sin_ = torch.sin(angles).to(dtype)
        cos_ = torch.cos(angles).to(dtype)

        return sin_, cos_

    def forward(self, x):
        # x.shape = [B, seq_len, H, head_dim]
        bs, seq_len, head_num, head_dim = x.shape

        assert head_dim == self.dim

        sin_, cos_ = self._build_rope_sin_cos(
            seq_len,
            x.device,
            x.dtype,
        )

        return _apply_rotary_emb(x, sin_, cos_)


def _apply_rope_at_position(rope, x, position):
    """Apply RoPE to one vector at a specific position."""
    assert x.ndim == 1
    rope_input = torch.zeros(
        1, position + 1, 1, x.shape[0], dtype=x.dtype, device=x.device
    )
    rope_input[0, position, 0] = x
    return rope(rope_input)[0, position, 0]


if __name__ == "__main__":
    torch.manual_seed(123)

    dim = 128
    rope = Rope(dim)
    q = torch.randn(dim)
    k = torch.randn(dim)

    pos_m = 3
    pos_n = 11

    q_pos0 = _apply_rope_at_position(rope, q, 0)
    print("position 0 unchanged:", torch.allclose(q_pos0, q, rtol=1e-6, atol=1e-6))

    q_m = _apply_rope_at_position(rope, q, pos_m)
    print(
        "norm unchanged:",
        torch.allclose(q_m.norm(), q.norm(), rtol=1e-6, atol=1e-6),
    )

    q_m_then_n = _apply_rope_at_position(rope, q_m, pos_n)
    q_m_plus_n = _apply_rope_at_position(rope, q, pos_m + pos_n)
    print(
        "rotate m then n equals rotate m+n:",
        torch.allclose(q_m_then_n, q_m_plus_n, rtol=1e-5, atol=1e-6),
    )

    q_m = _apply_rope_at_position(rope, q, pos_m)
    k_n = _apply_rope_at_position(rope, k, pos_n)
    k_n_minus_m = _apply_rope_at_position(rope, k, pos_n - pos_m)

    left = torch.dot(q_m, k_n)
    right = torch.dot(q, k_n_minus_m)
    print(
        "(R_m q)^T (R_n k) ~= q^T R_{n-m} k:",
        torch.allclose(left, right, rtol=1e-5, atol=1e-6),
    )
    print("dot diff:", (left - right).abs().item())
