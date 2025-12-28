import torch.nn as nn

# FFN with SwiGLU activation and dropout
class FFN(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        a = self.w1(x)
        b = self.activation(self.w2(x))
        return self.dropout(self.w3(a * b))