"""
Generation utilities for CodeGPT.

  sample()           — one-shot synchronous generation, returns full decoded string
  stream_generate()  — async generator, yields decoded tokens one at a time
  build_chat_prompt() — formats a messages list as a chat template
  build_fim_prompt()  — re-exported from fim.py for convenience
"""
import asyncio
from typing import AsyncGenerator, List, Optional

import torch

from llm.config import ModelConfig
from llm.fim import build_fim_prompt


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_chat_prompt(
    messages: List[dict],
    system: Optional[str],
    tokenizer,
    cfg: Optional[ModelConfig] = None,
) -> List[int]:
    """
    Convert a list of {"role": ..., "content": ...} messages into token ids.

    Format:
        <|system|>{system}<|user|>{user}<|assistant|>{assistant}<|user|>…<|assistant|>
    The final assistant turn is left open (no EOS) so the model continues it.
    """
    if cfg is None:
        cfg = ModelConfig()
    ids: List[int] = []

    if system:
        ids += [cfg.sys_id] + tokenizer.encode(system)

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            ids += [cfg.user_id] + tokenizer.encode(content)
        elif role == "assistant":
            ids += [cfg.asst_id] + tokenizer.encode(content) + [cfg.eos_id]
        # system messages inside the list are appended as-is
        elif role == "system":
            ids += [cfg.sys_id] + tokenizer.encode(content)

    # Open the next assistant turn
    ids += [cfg.asst_id]
    return ids


# ── Synchronous generation ────────────────────────────────────────────────────

def sample(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate a completion for a plain text prompt.
    Returns the *completion only* (not the prompt).
    """
    if device is None:
        device = next(model.parameters()).device
    cfg = model.cfg
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=cfg.eos_id,
    )
    new_ids = output_ids[0, len(ids):].tolist()
    return tokenizer.decode(new_ids)


# ── Async streaming generation ────────────────────────────────────────────────

async def stream_generate(
    model,
    tokenizer,
    input_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    yield_every: int = 1,
) -> AsyncGenerator[str, None]:
    """
    Async generator that streams decoded tokens one at a time.

    Usage:
        async for token_str in stream_generate(model, tok, ids):
            ws.send(token_str)
    """
    if device is None:
        device = next(model.parameters()).device
    cfg = model.cfg

    ctx = torch.tensor([input_ids], dtype=torch.long, device=device)
    model.eval()
    buf: List[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx_cond = ctx if ctx.size(1) <= cfg.max_seq_len \
                       else ctx[:, -cfg.max_seq_len:]

            logits, _ = model(ctx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cum_probs - torch.nn.functional.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(-1, sorted_idx, sorted_logits)

            probs   = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ctx     = torch.cat([ctx, next_id], dim=1)

            token_int = next_id.item()
            if token_int == cfg.eos_id:
                break

            buf.append(token_int)
            if len(buf) >= yield_every:
                yield tokenizer.decode(buf, skip_special=True)
                buf = []
                # Yield control back to the event loop
                await asyncio.sleep(0)

    # Flush remaining
    if buf:
        yield tokenizer.decode(buf, skip_special=True)


# ── FIM completion stream ──────────────────────────────────────────────────────

async def stream_fim(
    model,
    tokenizer,
    prefix: str,
    suffix: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a fill-in-the-middle completion (cursor insertion).
    """
    ids = build_fim_prompt(prefix, suffix, tokenizer)
    async for tok in stream_generate(
        model, tokenizer, ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    ):
        yield tok
