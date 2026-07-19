from torch import nn
import torch


class RmsNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        factory_kwargs = {"device": device, "dtype": dtype}

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x / torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            x = self.weight.float() * x
        return x.to(input_dtype)


if __name__ == "__main__":
    torch.manual_seed(123)

    normalized_shape = 128
    eps = 1e-5

    x = torch.randn(2, 3, normalized_shape)

    my_norm = RmsNorm(normalized_shape, eps=eps, elementwise_affine=True)
    official_norm = nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=True)
    compiled_norm = torch.compile(my_norm)

    with torch.no_grad():
        official_norm.weight.copy_(my_norm.weight)

    y_my = my_norm(x)
    y_official = official_norm(x)
    y_my_compiled = compiled_norm(x)

    print("max abs diff:", (y_my - y_official).abs().max().item())
    print("allclose:", torch.allclose(y_my, y_official, rtol=1e-5, atol=1e-6))
    print(
        "compiled_allclose:",
        torch.allclose(y_my_compiled, y_official, rtol=1e-5, atol=1e-6),
    )
