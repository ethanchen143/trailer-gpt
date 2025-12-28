# how to run: 
# python train.py --data ./data/train.txt --out ./runs

import argparse, time, signal
from pathlib import Path
import sys
import torch
import torch.nn as nn

from model import GPT
from tokenizer import BPETokenizer
from dataset import make_loader
from lr_scheduler import WarmupCosineLR
# gradient scaling for automatic mixed precision training
from amp_accum import AmpGrad

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out', type=str, default='/runs')

    # tokenizer / model dims
    p.add_argument('--vocab_size', type=int, default=16000)

    # context length
    p.add_argument('--block_size', type=int, default=128)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.2)

    # RoPE positional encoding
    p.add_argument('--max_pos', type=int, default=1024)

    # sliding window attention
    p.add_argument('--sliding_window', type=int, default=128)
    p.add_argument('--attention_sink', type=int, default=0)

    # GQA grouped query attention
    p.add_argument('--n_kv_head', type=int, default=2)

    # train
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--steps', type=int, default=2000, help='max optimizer steps for this run')
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--grad_accum_steps', type=int, default=8)
    p.add_argument('--save_interval', type=int, default=100, help='checkpoint interval in optimizer steps')
    p.add_argument('--resume_from', type=str, default=None, help='path to a checkpoint to resume from (ckpt_step*.pt). If omitted, will auto-pick latest in --out.')

    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train the BPE tokenizer - we're using the same data for training the tokenizer and the model
    tok = BPETokenizer(vocab_size=args.vocab_size)
    tok_dir = out_dir / 'tokenizer'
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_files_present = (tok_dir / "tokenizer.json").exists()
    if tok_files_present:
        tok.load(tok_dir)
        print(f"[tokenizer] Loaded existing tokenizer from {tok_dir}")
    else:
        tok.train(args.data)
        tok.save(tok_dir)
        print(f"[tokenizer] Trained and saved new tokenizer to {tok_dir}")
    vocab_size = tok.vocab_size

    # dataloader
    train_loader = make_loader(args.data, tok, args.block_size, args.batch_size)

    # init model, optimizer, scheduler, and amp (automatic mixed precision training)
    model = GPT(vocab_size=vocab_size, block_size=args.block_size,
                n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
                rope=True, max_pos=args.max_pos,
                sliding_window=args.sliding_window, attention_sink=args.attention_sink, n_kv_head=args.n_kv_head)
    model.to(device)
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    total_steps = min(args.steps, args.epochs * len(train_loader))
    warmup = min(args.warmup_steps, max(total_steps // 10, 1))
    scheduler = WarmupCosineLR(optim, warmup_steps=warmup, total_steps=total_steps, base_lr=args.lr)

    amp = AmpGrad(optim, accum=args.grad_accum_steps, amp=args.mixed_precision)

    # optional resume
    def latest_checkpoint(dir_path: Path):
        ckpts = sorted(dir_path.glob("ckpt_step*.pt"))
        return ckpts[-1] if ckpts else None

    ckpt_path = Path(args.resume_from) if args.resume_from else latest_checkpoint(out_dir)
    step = 0
    if ckpt_path and ckpt_path.exists():
        print(f"[resume] Loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.step_num = ckpt.get("scheduler_step", scheduler.step_num)
        amp_state = ckpt.get("amp_scaler")
        if amp_state is not None and amp.amp:
            amp.scaler.load_state_dict(amp_state)
        step = ckpt.get("step", 0)
        print(f"[resume] Resumed from step {step}")

    # training loop
    model.train()
    while step < args.steps:
        for xb, yb in train_loader:
            if step >= args.steps: 
                break
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=amp.amp):
                logits, loss, _ = model(xb, yb)
            print(step)
            amp.backward(loss)
            if amp.should_step():
                amp.step()
                amp.zero_grad()
                lr = scheduler.step()
                step += 1

                # checkpoint - save every save_interval optimizer steps
                if step > 0 and step % args.save_interval == 0:
                    ckpt = {
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "scheduler_step": scheduler.step_num,
                        "amp_scaler": amp.scaler.state_dict() if amp.amp else None,
                        "step": step,
                    }
                    torch.save(ckpt, out_dir / f"ckpt_step{step:07d}.pt")
                    torch.save(model.state_dict(), out_dir / f"model_step{step:07d}.pt")
                    print(f"[checkpoint] Saved at step {step}")
                    print(f"Loss: {loss.item():.2f}")
                    # decode a single example for quick qualitative check
                    # gen_ids = model.generate(xb[:1]).squeeze(0).cpu().tolist()
                    # tgt_ids = yb[0].cpu().tolist()
                    # print(f"Generation: {tok.decode(gen_ids)}")
                    # print(f"Original: {tok.decode(tgt_ids)}")

if __name__ == "__main__":
    main()