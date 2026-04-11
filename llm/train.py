"""
Phase-1 training loop — base language modelling on code.

Usage:
    python llm/train.py                    # uses defaults
    python llm/train.py --steps 5000 --batch-size 8
    python llm/train.py --resume           # resume from latest checkpoint

Training automatically:
  1. Collects Python source files from stdlib + site-packages
  2. Trains the BPE tokenizer (skipped if tok.json already exists)
  3. Builds the CodeGPT model
  4. Runs the training loop with AMP + gradient clipping
  5. Saves a checkpoint every `--save-every` steps
"""
import argparse
import math
import os
import sys
import time

# ── allow `python llm/train.py` from project root ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llm.config import ModelConfig
from llm.tokenizer import BPETokenizer
from llm.model import CodeGPT
from llm.dataset import collect_code_files, CodeDataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    if step >= total:
        return lr_min
    progress = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, step, loss, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step":       step,
        "loss":       loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "config":     cfg.__dict__,
    }, path)


def latest_checkpoint(cfg: ModelConfig) -> str | None:
    d = cfg.checkpoint_dir
    if not os.path.isdir(d):
        return None
    ckpts = sorted(
        [f for f in os.listdir(d) if f.startswith("step_") and f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    return os.path.join(d, ckpts[-1]) if ckpts else None


def train(
    steps: int = 80_000,
    batch_size: int = 8,
    seq_len: int | None = None,
    lr: float = 3e-4,
    lr_min: float = 1e-5,
    warmup: int = 500,
    grad_clip: float = 1.0,
    save_every: int = 500,
    resume: bool = False,
    extra_dirs: list | None = None,
    progress_callback=None,   # called with (step, loss, tokens_per_sec)
):
    cfg    = ModelConfig()
    device = get_device()
    print(f"[train] device = {device}")

    if seq_len:
        cfg.max_seq_len = seq_len

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = BPETokenizer()
    if os.path.exists(cfg.tokenizer_path):
        tokenizer.load(cfg.tokenizer_path)
    else:
        texts = collect_code_files(extra_dirs=extra_dirs)
        tokenizer.train(texts, vocab_size=cfg.vocab_size)
        tokenizer.save(cfg.tokenizer_path)

    # ── Dataset ───────────────────────────────────────────────────────────
    texts = collect_code_files(extra_dirs=extra_dirs)
    all_ids = []
    print("[train] tokenising corpus …")
    for i, text in enumerate(texts):
        all_ids.extend(tokenizer.encode(text, add_eos=True))
        if (i + 1) % 1000 == 0:
            print(f"[train]   {i+1}/{len(texts)} files")
    print(f"[train] total tokens: {len(all_ids):,}")

    dataset = CodeDataset(all_ids, cfg, fim_rate=0.5)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = CodeGPT(cfg).to(device)
    print(f"[train] model params: {model.num_params()/1e6:.1f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    start_step = 0
    if resume:
        ckpt_path = latest_checkpoint(cfg)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            print(f"[train] resumed from step {start_step}")

    # ── AMP scaler ────────────────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    step       = start_step
    data_iter  = iter(loader)
    t0         = time.time()
    tokens_run = 0

    print(f"[train] starting from step {step}, target {steps} steps …")

    while step < steps:
        # Fetch next batch (cycle through dataset)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule
        current_lr = cosine_lr(step, warmup, steps, lr, lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Forward + backward with AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_run += x.numel()

        # Logging
        if step % 50 == 0:
            elapsed = time.time() - t0
            tps     = tokens_run / elapsed
            print(f"[train] step {step:6d} | loss {loss.item():.4f} | "
                  f"lr {current_lr:.2e} | {tps:,.0f} tok/s")
            if progress_callback:
                progress_callback(step, loss.item(), tps)
            tokens_run = 0
            t0 = time.time()

        # Checkpoint
        if step % save_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"step_{step:07d}.pt")
            save_checkpoint(model, optimizer, step, loss.item(), cfg, path)
            print(f"[train] checkpoint saved → {path}")

    # Final checkpoint
    path = os.path.join(cfg.checkpoint_dir, f"step_{step:07d}.pt")
    save_checkpoint(model, optimizer, step, loss.item(), cfg, path)
    print(f"[train] training complete. final checkpoint → {path}")
    return model


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CodeGPT")
    parser.add_argument("--steps",      type=int,   default=80_000)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save-every", type=int,   default=500)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--data-dir",   type=str,   default=None,
                        help="Extra directory of code files to include")
    args = parser.parse_args()

    train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
        resume=args.resume,
        extra_dirs=[args.data_dir] if args.data_dir else None,
    )
