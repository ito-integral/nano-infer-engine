from .rope import Rope
from .paged_attention import batched_paged_attention_reference

import torch
from torch import nn

from nano_infer_engine.paged_cache import PagedKVCache


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

    def forward(
        self,
        x,
        k_cache=None,
        v_cache=None,
        cache_position=0,
        attention_mask=None,
        position_ids=None,
    ):
        """
        k_cache/v_cache.shape = bs, cached_len, head_num, head_dim
        """
        bs, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        if (k_cache is None) != (v_cache is None):
            raise ValueError(
                "k_cache and v_cache must both be provided or both be None"
            )
        kv_seq_len = cache_position

        # x = x.view(bs, seq_len, self.head_num, hidden_size)
        # q.shape =(bs, seq_len, hidden_size)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bs, seq_len, self.head_num, self.head_dim)
        k = k.view(bs, seq_len, self.head_num, self.head_dim)
        v = v.view(bs, seq_len, self.head_num, self.head_dim)

        q = self.rope(q, position_offset=kv_seq_len, position_ids=position_ids)
        k = self.rope(k, position_offset=kv_seq_len, position_ids=position_ids)

        if k_cache is not None and v_cache is not None:
            new_kv_seq_len = kv_seq_len + seq_len
            if new_kv_seq_len > k_cache.shape[1]:
                raise ValueError("KV cache capacity exceeded")
            k_cache[:, kv_seq_len:new_kv_seq_len].copy_(k)
            v_cache[:, kv_seq_len:new_kv_seq_len].copy_(v)
            k = k_cache[:, :new_kv_seq_len]
            v = v_cache[:, :new_kv_seq_len]
        else:
            new_kv_seq_len = seq_len

        q = q.transpose(1, 2)  # bs, head_num, seq_len, head_dim
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # qk_scores.shape = [bs, head_num, seq_len, new_kv_seq_len]
        qk_scores = q @ k.transpose(-1, -2) * (self.head_dim**-0.5)

        query_positions = torch.arange(
            kv_seq_len,
            kv_seq_len + seq_len,
            device=x.device,
        )[:, None]
        key_positions = torch.arange(
            new_kv_seq_len,
            device=x.device,
        )[None, :]
        mask = key_positions > query_positions

        qk_scores = qk_scores.masked_fill(mask, float("-inf"))
        if attention_mask is not None:
            if attention_mask.shape != (bs, new_kv_seq_len):
                raise ValueError(
                    "attention_mask must match the batch and KV sequence length"
                )
            # [bs, new_kv_seq_len] -> [bs, 1, 1, new_kv_seq_len]
            # Broadcast over attention heads and all query positions.
            key_padding_mask = ~attention_mask.bool()[:, None, None, :]
            qk_scores = qk_scores.masked_fill(key_padding_mask, float("-inf"))
            # [bs, seq_len] -> [bs, 1, seq_len, 1]
            # Broadcast over attention heads and all key positions.
            query_padding_mask = ~attention_mask.bool()[
                :, None, kv_seq_len:new_kv_seq_len, None
            ]
            qk_scores = qk_scores.masked_fill(query_padding_mask, 0.0)

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

    def forward(
        self,
        x,
        k_cache=None,
        v_cache=None,
        cache_position=0,
        attention_mask=None,
        position_ids=None,
        *,
        paged_cache: PagedKVCache | None = None,
        layer_index: int | None = None,
        sequence_id: str = "default",
        sequence_ids: tuple[str, ...] | None = None,
    ):
        """
        kv_cache.shape = bs, cached_len, kv_head_num, head_dim
        """
        bs, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        if (k_cache is None) != (v_cache is None):
            raise ValueError(
                "k_cache and v_cache must both be provided or both be None"
            )
        if paged_cache is not None and (k_cache is not None or v_cache is not None):
            raise ValueError("contiguous cache and paged cache cannot be used together")

        kv_seq_len = cache_position

        paged_sequence_ids: tuple[str, ...] | None = None
        paged_sequence_lengths: tuple[int, ...] | None = None
        if paged_cache is not None:
            if seq_len != 1:
                raise ValueError(
                    "paged attention reference currently requires seq_len=1"
                )
            if layer_index is None:
                raise ValueError("layer_index is required for paged attention")
            if attention_mask is not None and not bool(attention_mask.all()):
                raise ValueError(
                    "paged attention reference does not support padding masks"
                )

            if sequence_ids is None:
                if bs != 1:
                    raise ValueError(
                        "sequence_ids is required for batched paged attention"
                    )
                paged_sequence_ids = (sequence_id,)
            else:
                if not isinstance(sequence_ids, tuple):
                    raise TypeError("sequence_ids must be a tuple")
                if len(sequence_ids) != bs:
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
                paged_cache.get_sequence_length(current_sequence_id)
                for current_sequence_id in paged_sequence_ids
            )
            paged_position_ids = torch.tensor(
                paged_sequence_lengths,
                dtype=torch.long,
                device=x.device,
            )[:, None]
            if position_ids is not None and not torch.equal(
                position_ids, paged_position_ids
            ):
                raise ValueError("position_ids must match paged cache sequence lengths")
            position_ids = paged_position_ids

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

        q = self.rope(q, position_offset=kv_seq_len, position_ids=position_ids)
        k = self.rope(k, position_offset=kv_seq_len, position_ids=position_ids)

        if paged_cache is not None:
            assert paged_sequence_ids is not None
            assert paged_sequence_lengths is not None
            assert layer_index is not None

            for batch_index, current_sequence_id in enumerate(paged_sequence_ids):
                paged_cache.write(
                    layer_index,
                    current_sequence_id,
                    paged_sequence_lengths[batch_index],
                    k[batch_index],
                    v[batch_index],
                )

            new_sequence_lengths = tuple(
                sequence_length + 1 for sequence_length in paged_sequence_lengths
            )
            block_tables = tuple(
                paged_cache.get_block_table(current_sequence_id)
                for current_sequence_id in paged_sequence_ids
            )
            context = batched_paged_attention_reference(
                query=q[:, 0],
                key_cache=paged_cache.keys,
                value_cache=paged_cache.values,
                block_tables=block_tables,
                sequence_lengths=new_sequence_lengths,
                layer_index=layer_index,
            )
            context = context.reshape(bs, 1, self.hidden_size)
            output = self.o_proj(context)
            assert output.shape == x.shape
            return output

        if k_cache is not None and v_cache is not None:
            new_kv_seq_len = kv_seq_len + seq_len
            if new_kv_seq_len > k_cache.shape[1]:
                raise ValueError("KV cache capacity exceeded")
            k_cache[:, kv_seq_len:new_kv_seq_len].copy_(k)
            v_cache[:, kv_seq_len:new_kv_seq_len].copy_(v)

            k = k_cache[:, :new_kv_seq_len]
            v = v_cache[:, :new_kv_seq_len]
        else:
            new_kv_seq_len = seq_len

        q = q.transpose(1, 2)  # bs, q_head_num, seq_len, head_dim
        k = k.transpose(1, 2)  # bs, kv_head_num, seq_len, head_dim
        v = v.transpose(1, 2)  # bs, kv_head_num, seq_len, head_dim

        k = k.repeat_interleave(
            self.group_num, dim=1
        )  # bs, q_head_num, seq_len, head_dim
        v = v.repeat_interleave(
            self.group_num, dim=1
        )  # bs, q_head_num, seq_len, head_dim

        # qk_scores.shape = [bs, q_head_num, seq_len, new_kv_seq_len]
        qk_scores = q @ k.transpose(-1, -2) * (self.head_dim ** (-0.5))

        query_positions = torch.arange(
            kv_seq_len,
            kv_seq_len + seq_len,
            device=x.device,
        )[:, None]
        key_positions = torch.arange(
            new_kv_seq_len,
            device=x.device,
        )[None, :]

        mask = key_positions > query_positions

        # mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        # mask = mask.triu(diagonal=1)

        qk_scores = qk_scores.masked_fill(mask, float("-inf"))
        if attention_mask is not None:
            if attention_mask.shape != (bs, new_kv_seq_len):
                raise ValueError(
                    "attention_mask must match the batch and KV sequence length"
                )
            # [bs, new_kv_seq_len] -> [bs, 1, 1, new_kv_seq_len]
            # Broadcast over attention heads and all query positions.
            key_padding_mask = ~attention_mask.bool()[:, None, None, :]
            qk_scores = qk_scores.masked_fill(key_padding_mask, float("-inf"))
            # [bs, seq_len] -> [bs, 1, seq_len, 1]
            # Broadcast over attention heads and all key positions.
            query_padding_mask = ~attention_mask.bool()[
                :, None, kv_seq_len:new_kv_seq_len, None
            ]
            qk_scores = qk_scores.masked_fill(query_padding_mask, 0.0)

        scores = torch.softmax(qk_scores, dim=-1, dtype=torch.float32)
        scores = scores.to(v.dtype)
        scores = scores @ v  # bs, head_num, seq_len, head_dim

        scores = scores.transpose(1, 2)
        # bs, seq_len, hidden_size
        scores = scores.flatten(start_dim=-2, end_dim=-1)
        output = self.o_proj(scores)

        assert output.shape == x.shape
        return output
