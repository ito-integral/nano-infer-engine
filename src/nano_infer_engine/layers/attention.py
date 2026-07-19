from .rope import Rope

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, hidden_size, rope=None):
        super().__init__()
        assert hidden_size % head_num == 0
        self.head_dim = hidden_size // head_num
        self.head_num = head_num
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if rope is None:
            self.rope = Rope(self.head_dim)
        else:
            self.rope = rope

    def forward(self, x):
        bs, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        # x = x.view(bs, seq_len, self.head_num, hidden_size)
        # q.shape =(bs, seq_len, hidden_size)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bs, seq_len, self.head_num, self.head_dim)
        k = k.view(bs, seq_len, self.head_num, self.head_dim)
        v = v.view(bs, seq_len, self.head_num, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)  # bs, head_num, seq_len, head_dim
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # bs, head_num, seq_len, seq_len
        qk_scores = q @ k.transpose(-1, -2) / (self.head_dim**0.5)

        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        mask = mask.triu(diagonal=1)

        qk_scores = qk_scores.masked_fill(mask, float("-inf"))

        scores = torch.softmax(qk_scores, dim=-1, dtype=torch.float32)
        scores = scores.to(v.dtype)
        scores = scores @ v  # bs, head_num, seq_len, head_dim

        scores = scores.transpose(1, 2)
        # bs, seq_len, hidden_size
        scores = scores.flatten(start_dim=-2, end_dim=-1)
        output = self.o_proj(scores)
        assert output.shape == x.shape
        return output


class GroupedQueryAttention(nn.Module):
    def __init__(self, q_head_num, kv_head_num, hidden_size, rope=None):
        super().__init__()
        assert hidden_size % q_head_num == 0
        assert q_head_num % kv_head_num == 0

        self.head_dim = hidden_size // q_head_num
        self.q_head_num = q_head_num
        self.hidden_size = hidden_size
        self.kv_head_num = kv_head_num
        self.group_num = q_head_num // kv_head_num

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.kv_head_num * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.kv_head_num * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if rope is None:
            self.rope = Rope(self.head_dim)
        else:
            self.rope = rope

    def forward(self, x):
        bs, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        # x = x.view(bs, seq_len, self.head_num, hidden_size)
        # q.shape =(bs, seq_len, hidden_size)
        q = self.q_proj(
            x
        )  # (bs, seq_len, hidden_size) <=>(bs, seq_len, q_head_num*head_dim)
        k = self.k_proj(x)  # (bs, seq_len, kv_head_num * head_dim)
        v = self.v_proj(x)  # (bs, seq_len, kv_head_num * head_dim)

        q = q.view(bs, seq_len, self.q_head_num, self.head_dim)
        k = k.view(bs, seq_len, self.kv_head_num, self.head_dim)
        v = v.view(bs, seq_len, self.kv_head_num, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)  # bs, q_head_num, seq_len, head_dim
        k = k.transpose(1, 2)  # bs, kv_head_num, seq_len, head_dim
        v = v.transpose(1, 2)  # bs, kv_head_num, seq_len, head_dim

        k = k.repeat_interleave(
            self.group_num, dim=1
        )  # bs, q_head_num, seq_len, head_dim
        v = v.repeat_interleave(
            self.group_num, dim=1
        )  # bs, q_head_num, seq_len, head_dim

        # bs, head_num, seq_len, seq_len
        qk_scores = q @ k.transpose(-1, -2) / (self.head_dim**0.5)

        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        mask = mask.triu(diagonal=1)

        qk_scores = qk_scores.masked_fill(mask, float("-inf"))

        scores = torch.softmax(qk_scores, dim=-1, dtype=torch.float32)
        scores = scores.to(v.dtype)
        scores = scores @ v  # bs, head_num, seq_len, head_dim

        scores = scores.transpose(1, 2)
        # bs, seq_len, hidden_size
        scores = scores.flatten(start_dim=-2, end_dim=-1)
        output = self.o_proj(scores)

        assert output.shape == x.shape
        return output
