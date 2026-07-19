from torch import nn

# 无状态的算子(即没有学习参数)
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, inner_size, hidden_size):
        super().__init__()
        self.act_fn = F.silu
        self.hidden_size = hidden_size
        self.inner_size = inner_size

        self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, inner_size, bias=False)

    def forward(self, x):
        bs, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size
        output = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(output)
        return output
