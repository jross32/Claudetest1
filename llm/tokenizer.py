"""
BPE Tokenizer — built entirely from scratch.

Algorithm:
  1. Start with a seed vocabulary of single bytes (0-255) plus special tokens.
  2. Count every adjacent pair of token-ids in the corpus.
  3. Merge the most-frequent pair into a new token; repeat until vocab_size reached.
  4. encode() and decode() use the learned merge table.

No external libraries required — pure Python + stdlib.
"""
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Pre-tokenisation regex (splits on whitespace and operators before BPE) ──
# Mirrors the approach used in GPT-2 / tiktoken but simplified for code.
_PRETOK = re.compile(
    r"""(?x)
    \s+                          # whitespace runs (preserve indentation)
    | [a-zA-Z_]\w*               # identifiers / keywords
    | \d+(?:\.\d+)?(?:[eE][+-]?\d+)?  # numbers
    | \.{1,3}                    # dots
    | [+\-*/%&|^~<>=!]{1,2}      # operators (1 or 2 chars)
    | [][(){},;:@#\\]            # brackets / punctuation
    | \"\"\"[\s\S]*?\"\"\"       # triple-quoted strings
    | \'\'\'[\s\S]*?\'\'\'
    | \"[^\"\\]*(?:\\.[^\"\\]*)*\"  # double-quoted strings
    | \'[^\'\\]*(?:\\.[^\'\\]*)*\'  # single-quoted strings
    | \#[^\n]*                   # comments
    | .                          # anything else
    """,
    re.DOTALL,
)

# Special token strings (must NOT appear in normal text)
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch on raw text.
    """

    def __init__(self):
        # Maps token string → id and id → string
        self._tok2id: Dict[str, int] = {}
        self._id2tok: Dict[int, str] = {}
        # Ordered list of merge rules: (left_id, right_id) → merged_id
        self._merges: List[Tuple[int, int]] = []
        self._merge_map: Dict[Tuple[int, int], int] = {}
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, texts: List[str], vocab_size: int = 16_384) -> None:
        """Learn BPE merges from a list of text strings."""
        assert vocab_size >= 256 + len(SPECIAL_TOKENS), "vocab_size too small"

        # Step 1: seed vocabulary — special tokens first, then all 256 bytes
        self._tok2id = {}
        self._id2tok = {}
        for i, st in enumerate(SPECIAL_TOKENS):
            self._tok2id[st] = i
            self._id2tok[i] = st
        offset = len(SPECIAL_TOKENS)
        for b in range(256):
            ch = chr(b)
            self._tok2id[ch] = b + offset
            self._id2tok[b + offset] = ch
        next_id = 256 + offset

        # Step 2: pre-tokenise corpus into lists of char-level ids
        print(f"[tokenizer] pre-tokenising {len(texts)} texts …")
        corpus: List[List[int]] = []
        for text in texts:
            for word in _PRETOK.findall(text):
                ids = [self._tok2id.get(c, self._tok2id["<|unk|>"]) for c in word]
                if ids:
                    corpus.append(ids)

        # Step 3: BPE merge loop
        n_merges = vocab_size - next_id
        print(f"[tokenizer] learning {n_merges} BPE merges (vocab {next_id} → {vocab_size}) …")
        self._merges = []
        self._merge_map = {}

        for step in range(n_merges):
            # Count all adjacent pairs
            pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for seq in corpus:
                for a, b in zip(seq, seq[1:]):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < 2:
                break  # no pair appears more than once — stop early

            # Create new token for the merge
            left_str  = self._id2tok[best[0]]
            right_str = self._id2tok[best[1]]
            new_str   = left_str + right_str
            new_id    = next_id
            next_id  += 1
            self._tok2id[new_str] = new_id
            self._id2tok[new_id]  = new_str
            self._merges.append(best)
            self._merge_map[best] = new_id

            # Apply merge to corpus
            new_corpus = []
            for seq in corpus:
                new_seq: List[int] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i+1]) == best:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_corpus.append(new_seq)
            corpus = new_corpus

            if (step + 1) % 1000 == 0:
                print(f"[tokenizer]   step {step+1}/{n_merges}, vocab={next_id}")

        self._trained = True
        print(f"[tokenizer] done. final vocab size = {len(self._tok2id)}")

    # ── Encoding / Decoding ───────────────────────────────────────────────

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert text to a list of token ids."""
        ids: List[int] = []
        if add_bos:
            ids.append(self._tok2id["<|bos|>"])

        offset = len(SPECIAL_TOKENS)
        for word in _PRETOK.findall(text):
            # Check if the whole word is a special token
            if word in self._tok2id:
                ids.append(self._tok2id[word])
                continue
            # Start as char-level ids
            seq = [self._tok2id.get(c, self._tok2id["<|unk|>"]) for c in word]
            # Apply merge rules in order
            for merge, merged_id in self._merge_map.items():
                if len(seq) < 2:
                    break
                new_seq: List[int] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i+1]) == merge:
                        new_seq.append(merged_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                seq = new_seq
            ids.extend(seq)

        if add_eos:
            ids.append(self._tok2id["<|eos|>"])
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token ids back to a string."""
        parts = []
        special_ids = {self._tok2id[t] for t in SPECIAL_TOKENS if t in self._tok2id}
        for i in ids:
            tok = self._id2tok.get(i, "")
            if skip_special and i in special_ids:
                continue
            parts.append(tok)
        return "".join(parts)

    def token_to_id(self, token: str) -> int:
        return self._tok2id.get(token, self._tok2id.get("<|unk|>", 3))

    def id_to_token(self, idx: int) -> str:
        return self._id2tok.get(idx, "<|unk|>")

    @property
    def vocab_size(self) -> int:
        return len(self._tok2id)

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "tok2id": self._tok2id,
            "merges": self._merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[tokenizer] saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._tok2id = {k: int(v) for k, v in data["tok2id"].items()}
        self._id2tok = {int(v): k for k, v in data["tok2id"].items()}
        self._merges = [tuple(m) for m in data["merges"]]
        self._merge_map = {tuple(m): self._tok2id[self._id2tok[m[0]] + self._id2tok[m[1]]]
                           for m in self._merges
                           if (self._id2tok[m[0]] + self._id2tok[m[1]]) in self._tok2id}
        self._trained = True
        print(f"[tokenizer] loaded {len(self._tok2id)} tokens from {path}")
