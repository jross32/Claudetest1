"""
LLMServer — drop-in replacement for ollama_client.py.

Loads CodeGPT + BPETokenizer from the latest checkpoint and exposes:
  is_ready()           → bool
  stream_chat(messages, system)  → AsyncGenerator[str, None]
  generate_sync(prompt, max_new) → str   (blocking, for quick tests)

The interface is intentionally compatible with the old ollama_client so
ai_coder.py and evolution.py only need a one-line import swap.
"""
import os
import threading
from typing import AsyncGenerator, List, Optional

import torch

from llm.config import ModelConfig
from llm.tokenizer import BPETokenizer
from llm.model import CodeGPT
from llm.train import get_device, latest_checkpoint
from llm.generate import build_chat_prompt, stream_generate, sample


class LLMServer:
    """
    Singleton-style server object.  Instantiate once at app startup:

        server = LLMServer()
        server.load_latest()     # non-blocking — loads in background thread
    """

    def __init__(self):
        self.cfg       = ModelConfig()
        self.device    = get_device()
        self.tokenizer = BPETokenizer()
        self.model: Optional[CodeGPT] = None
        self._ready    = False
        self._lock     = threading.Lock()
        self._status   = "unloaded"   # unloaded | loading | ready | error
        self._steps_trained = 0
        self._finetune_done = False

    # ── Status ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    @property
    def status(self) -> str:
        return self._status

    @property
    def steps_trained(self) -> int:
        return self._steps_trained

    @property
    def finetune_done(self) -> bool:
        return self._finetune_done

    def model_info(self) -> dict:
        params = self.model.num_params() if self.model else 0
        return {
            "status":        self._status,
            "params_m":      round(params / 1e6, 1),
            "vocab_size":    self.cfg.vocab_size,
            "steps_trained": self._steps_trained,
            "finetune_done": self._finetune_done,
            "device":        str(self.device),
        }

    # ── Loading ───────────────────────────────────────────────────────────

    def load_latest(self, blocking: bool = False):
        """
        Load the latest checkpoint (or finetune.pt if it exists).
        If blocking=False, loading runs in a background thread.
        """
        if blocking:
            self._load()
        else:
            t = threading.Thread(target=self._load, daemon=True)
            t.start()

    def _load(self):
        with self._lock:
            self._status = "loading"
            self._ready  = False
        try:
            # Load tokenizer
            if os.path.exists(self.cfg.tokenizer_path):
                self.tokenizer.load(self.cfg.tokenizer_path)
                print(f"[server] tokenizer loaded ({self.tokenizer.vocab_size} tokens)")
            else:
                print("[server] no tokenizer found — model will generate garbage until trained")

            # Build model skeleton
            model = CodeGPT(self.cfg).to(self.device)

            # Try finetune checkpoint first, then latest phase-1 checkpoint
            finetune_path = os.path.join(self.cfg.checkpoint_dir, "finetune.pt")
            if os.path.exists(finetune_path):
                ckpt = torch.load(finetune_path, map_location=self.device)
                model.load_state_dict(ckpt["model"])
                self._steps_trained = ckpt.get("step", 0)
                self._finetune_done = True
                print(f"[server] loaded fine-tuned weights from {finetune_path}")
            else:
                ckpt_path = latest_checkpoint(self.cfg)
                if ckpt_path:
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    model.load_state_dict(ckpt["model"])
                    self._steps_trained = ckpt.get("step", 0)
                    print(f"[server] loaded base weights from {ckpt_path} "
                          f"(step {self._steps_trained})")
                else:
                    print("[server] no checkpoint found — using random weights")

            model.eval()
            with self._lock:
                self.model   = model
                self._status = "ready"
                self._ready  = True
            print(f"[server] ready — {model.num_params()/1e6:.1f} M params on {self.device}")

        except Exception as e:
            with self._lock:
                self._status = f"error: {e}"
                self._ready  = False
            print(f"[server] ERROR during load: {e}")

    def reload(self):
        """Hot-reload weights after training completes."""
        self._ready = False
        self._status = "loading"
        self.load_latest(blocking=True)

    # ── Generation ────────────────────────────────────────────────────────

    async def stream_chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator yielding decoded token strings for a chat conversation.

        Compatible with the old ollama_client.stream_chat() interface so
        callers don't need to change.
        """
        if not self._ready or self.model is None:
            yield "[Model not ready — please train or wait for loading to complete]"
            return

        ids = build_chat_prompt(messages, system, self.tokenizer, self.cfg)
        async for tok in stream_generate(
            self.model, self.tokenizer, ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=self.device,
        ):
            yield tok

    def generate_sync(
        self,
        prompt: str,
        max_new: int = 200,
        temperature: float = 0.8,
    ) -> str:
        """Blocking generation — useful for quick command-line tests."""
        if not self._ready or self.model is None:
            return "[Model not ready]"
        return sample(self.model, self.tokenizer, prompt,
                      max_new_tokens=max_new, temperature=temperature,
                      device=self.device)


# ── Module-level singleton (imported by ai_coder.py / evolution.py) ──────────

_server: Optional[LLMServer] = None


def get_server() -> LLMServer:
    global _server
    if _server is None:
        _server = LLMServer()
        _server.load_latest(blocking=False)
    return _server
