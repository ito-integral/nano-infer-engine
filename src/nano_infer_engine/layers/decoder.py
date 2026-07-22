from torch import nn
from .attention import GroupedQueryAttention
from .mlp import SwiGLUMLP
from .rms_norm import RmsNorm


class Llama3Decoder(nn.Module):
    def __init__(
        self, norm_eps, q_head_num, kv_head_num, hidden_size, mlp_inner_size, rope=None
    ):
        super().__init__()
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.inner_size = mlp_inner_size

        self.attn = GroupedQueryAttention(q_head_num, kv_head_num, hidden_size, rope)
        self.ffn = SwiGLUMLP(mlp_inner_size, hidden_size)
        self.pre_norm = RmsNorm(hidden_size, eps=norm_eps)
        self.post_norm = RmsNorm(hidden_size, eps=norm_eps)

    def forward(self, x, k_cache=None, v_cache=None, cache_position=0):
        # x.shape: batch_size , seq_len, hidden_size
        attn_output = self.attn(
            self.pre_norm(x),
            k_cache,
            v_cache,
            cache_position,
        )
        h = attn_output + x
        o = self.ffn(self.post_norm(h)) + h

        return o
