"""
Phase-2 instruction fine-tuning.

Extracts (docstring, function_body) pairs from the code corpus via AST,
formats them as a chat template, and fine-tunes the model at a lower LR
with loss computed only on the assistant (output) tokens.

Usage:
    python llm/finetune.py                    # uses latest checkpoint
    python llm/finetune.py --steps 2000 --lr 3e-5
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from llm.config import ModelConfig
from llm.tokenizer import BPETokenizer
from llm.model import CodeGPT
from llm.dataset import collect_code_files, extract_docstring_pairs, InstructDataset
from llm.train import get_device, cosine_lr, save_checkpoint, latest_checkpoint


def finetune(
    steps: int = 2_000,
    batch_size: int = 4,
    lr: float = 3e-5,
    lr_min: float = 1e-6,
    warmup: int = 100,
    grad_clip: float = 1.0,
    save_every: int = 500,
    extra_dirs: list | None = None,
    progress_callback=None,   # called with (step, loss, tokens_per_sec)
):
    cfg    = ModelConfig()
    device = get_device()
    print(f"[finetune] device = {device}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = BPETokenizer()
    if os.path.exists(cfg.tokenizer_path):
        tokenizer.load(cfg.tokenizer_path)
    else:
        raise RuntimeError(
            "No tokenizer found at %s — run Phase 1 training first." % cfg.tokenizer_path
        )

    # ── Dataset ────────────────────────────────────────────────────────────
    texts = collect_code_files(extra_dirs=extra_dirs)
    pairs = extract_docstring_pairs(texts)
    if not pairs:
        raise RuntimeError("No docstring pairs found — ensure Phase 1 corpus is available.")

    dataset = InstructDataset(pairs, tokenizer, cfg)
    print(f"[finetune] {len(dataset)} instruction samples")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    # ── Model: load from latest Phase-1 checkpoint ─────────────────────────
    model = CodeGPT(cfg).to(device)
    ckpt_path = latest_checkpoint(cfg)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[finetune] loaded weights from {ckpt_path}")
    else:
        print("[finetune] WARNING: no Phase-1 checkpoint found — fine-tuning from random weights")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Fine-tuning loop ───────────────────────────────────────────────────
    model.train()
    step       = 0
    data_iter  = iter(loader)
    t0         = time.time()
    tokens_run = 0

    print(f"[finetune] starting fine-tuning for {steps} steps …")

    while step < steps:
        try:
            x, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, labels = next(data_iter)

        x, labels = x.to(device), labels.to(device)

        # LR schedule
        current_lr = cosine_lr(step, warmup, steps, lr, lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Forward with masked loss (only on assistant tokens)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(x)
            # Shift: logits[..., :-1] predicts labels[..., 1:]
            # But InstructDataset already aligns x and labels as input/target
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,   # masked positions
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_run += x.numel()

        if step % 50 == 0:
            elapsed = time.time() - t0
            tps     = tokens_run / elapsed
            print(f"[finetune] step {step:5d} | loss {loss.item():.4f} | "
                  f"lr {current_lr:.2e} | {tps:,.0f} tok/s")
            if progress_callback:
                progress_callback(step, loss.item(), tps)
            tokens_run = 0
            t0 = time.time()

        if step % save_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"finetune_step_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, loss.item(), cfg, path)
            print(f"[finetune] checkpoint saved → {path}")

    # Final save with canonical name for easy loading
    final_path = os.path.join(cfg.checkpoint_dir, "finetune.pt")
    save_checkpoint(model, optimizer, step, loss.item(), cfg, final_path)
    print(f"[finetune] done. final model → {final_path}")
    return model


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeGPT (Phase 2)")
    parser.add_argument("--steps",      type=int,   default=2_000)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-5)
    parser.add_argument("--save-every", type=int,   default=500)
    parser.add_argument("--data-dir",   type=str,   default=None)
    args = parser.parse_args()

    finetune(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
        extra_dirs=[args.data_dir] if args.data_dir else None,
    )
