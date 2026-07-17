import torch
from torch import nn


class Rope(nn.Module):
    def __init__(self, dim, theta_base=500000):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0
        self.theta_base = theta_base

    def _build_rope_sin_cos(self, seq_len, device, dtype):
        i = torch.arange(0, self.dim, 2, dtype=dtype, device=device)
        inv_freq = 1 / self.theta_base ** (i / self.dim)

        position = torch.arange(0, seq_len, dtype=dtype, device=device)
        angles = torch.einsum("i,j->ij", position, inv_freq)
        # seq, dim/2
        return torch.sin(angles), torch.cos(angles)

    def forward(self, x):
        # x.shape = [B, seq_len, H, head_dim]
        bs, seq_len, head_num, head_dim = x.shape

        assert head_dim == self.dim

        sin_, cos_ = self._build_rope_sin_cos(seq_len, x.device, x.dtype)

        x_even = x[..., ::2]
        x_ord = x[..., 1::2]

        sin_ = sin_[None, :, None, :]
        cos_ = cos_[None, :, None, :]

        y_even = x_even * cos_ - x_ord * sin_
        y_ord = x_even * sin_ + x_ord * cos_

        x[..., ::2] = y_even
        x[..., 1::2] = y_ord

        return x


def _apply_rope_at_position(rope, x, position):
    """Apply RoPE to one vector at a specific position."""
    assert x.ndim == 1
    rope_input = torch.zeros(1, position + 1, 1, x.shape[0], dtype=x.dtype, device=x.device)
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
