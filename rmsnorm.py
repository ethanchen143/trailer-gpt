import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # weight is a learnable parameter, gives us individual scaling for each dimension
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x * self.weight) / rms

# if __name__ == "__main__":
#     x = torch.randn(2,3,8)
#     print(x)
#     rn = RMSNorm(8)
#     y = rn(x)
#     print(y)