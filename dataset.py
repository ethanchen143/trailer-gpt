import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizer import BPETokenizer

class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer: BPETokenizer, block_size: int = 256):
        super().__init__()
        self.block_size = block_size
        text = Path(path).read_text(encoding='utf-8')
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return max(0, self.ids.numel() - self.block_size - 1)

    def __getitem__(self, i: int):
        x = self.ids[i:i+self.block_size]
        y = self.ids[i+1:i+self.block_size+1]
        return x, y

def make_loader(path: str, tokenizer: BPETokenizer, block_size: int, batch_size: int):
    dataset = TextDataset(path, tokenizer, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# if __name__ == "__main__":
#     tok = BPETokenizer()
#     tok.train("data/train.txt")
#     train_loader = make_loader("data/train.txt", tok, block_size=256, batch_size=32)
#     for x, y in train_loader:
#         print(x)
#         print(y)
#         break