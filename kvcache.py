import torch

class KVCache:
    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v

    @property
    def T(self):
        return self.k.size(2)