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
