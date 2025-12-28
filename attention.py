import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPECache, apply_rope_single
from kvcache import KVCache

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head

        self.wq = nn.Linear(n_embd, self.n_head * self.d_head, bias=False)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        # B = batch size
        # T = sequence length
        # C = n_embd
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)    # (B,n_head,T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,n_kv_head,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,n_kv_head,T,D)

        # init rope cache
        self.rope_cache = RoPECache(self.d_head, self.max_pos, device=x.device)
        pos = torch.arange(start_pos, start_pos + T, device=x.device)
        cos, sin = self.rope_cache.get(pos)
        q = apply_rope_single(q, cos, sin)   # (B,n_head,T,D)
        k = apply_rope_single(k, cos, sin)   # (B,n_kv_head,T,D)

        # Concatenate new tokens k and v with past cache (cache is stored in n_kv_head heads)
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,n_kv_head, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Sliding-window + attention-sink (crop along seq length)
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # --- GQA expand: repeat K/V heads to match Q heads before attention ---
        if self.n_kv_head != self.n_head:
            k_full = k_all.repeat_interleave(self.group_size, dim=1)  # (B,n_head,Tk,D)
            v_full = v_all.repeat_interleave(self.group_size, dim=1)  # (B,n_head,Tk,D)
        else:
            k_full, v_full = k_all, v_all

        # Scaled dot-product attention (PyTorch scales internally)
        # softmax(QK^T / sqrt(D)) * V
        y = F.scaled_dot_product_attention(q, k_full, v_full,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=kv_cache is None)          # (B,n_head,T,D)
        
        y = y.transpose(1, 2).contiguous().view(B,T,C)
        y = self.proj(y)

        # Update KV cache (store compact n_head heads, not expanded)
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B,n_khead, Tpast+T, D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)
        return y, new_cache