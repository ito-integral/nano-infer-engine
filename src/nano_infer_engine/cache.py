import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        capacity: int,
        kv_head_num: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "batch_size": batch_size,
            "capacity": capacity,
            "kv_head_num": kv_head_num,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        # 预分配kv cache
        shape = (batch_size, capacity, kv_head_num, head_dim)

        self.keys = [
            torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.values = [
            torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]

        self.position = 0
        self.capacity = capacity

    def get(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys[layer_index], self.values[layer_index]

    def validate_append(self, token_count: int) -> None:
        """验证是否超长"""
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an integer")
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.position + token_count > self.capacity:
            raise ValueError("KV cache capacity exceeded")

    def advance(self, token_count: int) -> None:
        self.validate_append(token_count)
        self.position += token_count

    def reset(self) -> None:
        """这样做的好处是避免反复申请和释放大块 GPU 显存"""
        self.position = 0
