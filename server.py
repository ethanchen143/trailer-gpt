import glob
from pathlib import Path
from typing import Optional

import torch

from tokenizer import BPETokenizer
from model import GPT

# Default fallbacks; dims will be inferred from checkpoint
DEFAULT_CONFIG = dict(
    block_size=512,
    dropout=0.1,
    rope=True,
    max_pos=4096,
    sliding_window=512,
    attention_sink=0,
)

RUNS_DIR = Path("runs")
TOKENIZER_DIR = RUNS_DIR / "tokenizer"


class _ModelCache:
    def __init__(self):
        self.tokenizer: Optional[BPETokenizer] = None
        self.model: Optional[GPT] = None
        self.ckpt_path: Optional[Path] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_tokenizer(self, tok_dir: Path = TOKENIZER_DIR):
        if self.tokenizer is None:
            tok = BPETokenizer()
            tok.load(tok_dir)
            self.tokenizer = tok
        return self.tokenizer

    def list_checkpoints(self, runs_dir: Path = RUNS_DIR):
        # include both full ckpts and raw model weights
        ckpts = list(runs_dir.glob("ckpt_step*.pt")) + list(runs_dir.glob("model_step*.pt"))
        ckpts.sort()
        return ckpts

    def _load_state(self, ckpt_path: Path):
        state = torch.load(ckpt_path, map_location=self.device)

        # handle torch.compile saved weights (_orig_mod.* keys)
        def _strip_prefix(sd, prefix="_orig_mod."):
            if not any(k.startswith(prefix) for k in sd.keys()):
                return sd
            return {k[len(prefix):]: v for k, v in sd.items()}

        # full checkpoint dict
        if isinstance(state, dict) and "model" in state:
            sd = _strip_prefix(state["model"])
        else:
            sd = _strip_prefix(state)
        return sd

    def _infer_config(self, sd: dict, tok: BPETokenizer):
        # infer dims from state dict
        vocab_size_sd, n_embd = sd["tok_emb.weight"].shape
        if tok.vocab_size != vocab_size_sd:
            # align tokenizer vocab if mismatch
            tok.vocab_size = vocab_size_sd
        # count layers
        layer_ids = []
        for k in sd.keys():
            if k.startswith("blocks.") and ".ln1.weight" in k:
                try:
                    idx = int(k.split(".")[1])
                    layer_ids.append(idx)
                except ValueError:
                    pass
        n_layer = (max(layer_ids) + 1) if layer_ids else 0

        # heads (infer GQA grouping)
        # wq shape: (n_head * d_head, n_embd) == (n_embd, n_embd) typically
        # wk shape: (n_kv_head * d_head, n_embd)
        wk = sd["blocks.0.attention.wk.weight"]
        Rkv = wk.shape[0]  # n_kv_head * d_head
        group_size = max(1, n_embd // Rkv)  # n_head / n_kv_head

        # choose n_head among divisors of n_embd that are multiples of group_size,
        # prefer head dim near 32
        candidates = []
        for n_head in range(group_size, n_embd + 1, group_size):
            if n_embd % n_head == 0:
                d_head = n_embd // n_head
                if n_head % group_size == 0:
                    candidates.append((abs(d_head - 32), n_head, d_head))
        if candidates:
            _, n_head, d_head = min(candidates)
        else:
            n_head = max(group_size, 1)
            d_head = max(1, n_embd // n_head)

        n_kv_head = max(1, n_head // group_size)

        cfg = dict(
            vocab_size=vocab_size_sd,
            block_size=DEFAULT_CONFIG["block_size"],
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=DEFAULT_CONFIG["dropout"],
            rope=DEFAULT_CONFIG["rope"],
            max_pos=DEFAULT_CONFIG["max_pos"],
            sliding_window=DEFAULT_CONFIG["sliding_window"],
            attention_sink=DEFAULT_CONFIG["attention_sink"],
            n_kv_head=n_kv_head,
        )
        return cfg

    def ensure_model(self, ckpt_path: Path):
        tok = self.load_tokenizer()
        if self.model is None or self.ckpt_path != ckpt_path:
            sd = self._load_state(ckpt_path)
            cfg = self._infer_config(sd, tok)
            model = GPT(**cfg)
            model.to(self.device)
            model.eval()
            model.load_state_dict(sd, strict=False)
            self.model = model
            self.ckpt_path = ckpt_path
        return self.model, tok, self.device


CACHE = _ModelCache()


def list_checkpoints():
    return CACHE.list_checkpoints()


@torch.no_grad()
def generate_text(
    prompt: str,
    checkpoint: Optional[str] = None,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
):
    ckpts = list_checkpoints()
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {RUNS_DIR}")
    ckpt_path = Path(checkpoint) if checkpoint else ckpts[-1]

    model, tok, device = CACHE.ensure_model(ckpt_path)

    # ensure prompt ends with newline (acts as boundary)
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"

    ids = tok.encode(prompt)
    if not ids:
        return ""
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # use newline token as default eos if available
    if eos_id is None:
        try:
            eos_id = tok.encode("\n")[0]
        except Exception:
            eos_id = None

    out_ids = model.generate(
        prompt=x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
        sliding_window=DEFAULT_CONFIG["sliding_window"],
        attention_sink=DEFAULT_CONFIG["attention_sink"],
    )[0].tolist()
    return tok.decode(out_ids)


if __name__ == "__main__":
    # Simple CLI test
    cks = list_checkpoints()
    if not cks:
        print(f"No checkpoints found in {RUNS_DIR}")
    else:
        text = generate_text("Hello from the trailer park,", str(cks[-1]), max_new_tokens=40)
        print(text)

