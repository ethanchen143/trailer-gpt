import torch.nn as nn
from rmsnorm import RMSNorm
from ffn import FFN
from attention import CausalSelfAttention

# Use RMSNorm instead of LayerNorm
# Use SwiGLU instead of GELU
# Casual Self Attention uses GQA (Grouped Query Attention)
# Use RoPE instead of PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_head, dropout, rope, max_pos, sliding_window, attention_sink, n_kv_head)
        self.ln2 = RMSNorm(n_embd)
        self.ffn = FFN(n_embd, mult=4, dropout=dropout)
    
    def forward(self, x, kv_cache=None, start_pos: int = 0):
        # pre-norm, more stable training
        a, kv_cache = self.attention(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + a
        f = self.ffn(self.ln2(x))
        x = x + f
        return x, kv_cache