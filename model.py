import torch
import torch.nn as nn
import torch.nn.functional as F
from block import TransformerBlock
from utils import top_k_top_p_filtering

class GPT(nn.Module):
    def __init__(self, vocab_size: int = 32000, block_size: int = 256,
                 n_layer: int=6, n_head: int=8, n_embd: int=512, dropout: float=0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout, rope, max_pos, sliding_window, attention_sink, n_kv_head)
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_cache_list=None, start_pos: int = 0):
        B, T = idx.shape
        # clip by context length
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
        x = self.tok_emb(idx) # (B,T,C: model embedding dimension)
        x = self.dropout(x)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache = block(x, kv_cache=cache, start_pos=start_pos)
            new_caches.append(cache)
        x = self.layer_norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches
    
    @torch.no_grad()
    def generate(self, 
                 prompt: torch.Tensor, 
                 max_new_tokens=200, 
                 temperature=0.1, 
                 top_k=50, 
                 top_p=None,
                 eos_id=1,
                 sliding_window: int | None = None, 
                 attention_sink: int = 0):
        self.eval()
        idx = prompt
        kvs = [None] * len(self.blocks)

        for _ in range(max_new_tokens):
            # feed full prompt once; then only the last token
            idx_cond = idx[:, -self.block_size:] if kvs[0] is None else idx[:, -1:]

            # absolute start position from cache length (0 on first step)
            start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

            logits, _, kvs = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None:
                if (next_id == eos_id).all():
                    break

        return idx