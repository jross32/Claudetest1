"""
Dataset utilities — collects code files from the local installation and wraps
them in a PyTorch Dataset with FIM augmentation.

Supported languages: Python, JavaScript, TypeScript (+ common automation
packages: Playwright, Selenium, Puppeteer, BeautifulSoup, Scrapy, etc.)
"""
import os
import sys
import random
import sysconfig
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from llm.config import ModelConfig
from llm.fim import fim_transform

# Extensions to collect by default — Python + JS/TS ecosystem
DEFAULT_EXTENSIONS = (".py", ".js", ".ts", ".mjs", ".cjs", ".jsx", ".tsx")

# Directories to skip unconditionally (binaries, build artifacts, etc.)
_SKIP_DIRS = {
    "__pycache__", "test", "tests", "docs", "doc",
    "dist", "build", ".git", ".svn", ".hg",
    "coverage", "fixtures", "examples",
}

# Directories containing high-value automation / scraping libraries
_AUTOMATION_PACKAGES = {
    "playwright", "selenium", "puppeteer", "pyppeteer",
    "beautifulsoup4", "bs4", "scrapy", "requests_html",
    "mechanize", "httpx", "aiohttp", "requests",
    "lxml", "parsel", "cssselect", "urllib3",
}


def _find_node_modules(roots: set) -> set:
    """
    Search common locations for node_modules to collect JS/TS automation code.
    Looks relative to the project root and a few well-known npm prefix paths.
    """
    extra: set = set()
    candidates = [
        os.path.join(os.getcwd(), "node_modules"),
        os.path.expanduser("~/.npm"),
        "/usr/local/lib/node_modules",
        "/usr/lib/node_modules",
    ]
    for c in candidates:
        if os.path.isdir(c):
            extra.add(c)
    return extra


# ── Source file collection ────────────────────────────────────────────────────

def collect_code_files(
    extensions: tuple = DEFAULT_EXTENSIONS,
    extra_dirs: Optional[List[str]] = None,
    max_file_size: int = 500_000,    # skip files larger than 500 KB
    min_file_size: int = 64,         # skip near-empty files
    include_js: bool = True,         # include JS/TS from node_modules
) -> List[str]:
    """
    Collect source code from:
    - Python stdlib + site-packages (.py)
    - Installed automation packages (playwright, selenium, scrapy, etc.)
    - node_modules for JS/TS automation code (.js, .ts, etc.)
    - Any extra_dirs supplied by the user

    Returns a list of file contents as strings.
    """
    search_roots: set = set()

    # ── Python sources ──────────────────────────────────────────────────────
    stdlib = sysconfig.get_path("stdlib")
    if stdlib and os.path.isdir(stdlib):
        search_roots.add(stdlib)

    platstdlib = sysconfig.get_path("platstdlib")
    if platstdlib and os.path.isdir(platstdlib):
        search_roots.add(platstdlib)

    for path in sys.path:
        if "site-packages" in path and os.path.isdir(path):
            search_roots.add(path)
            # Also add any automation sub-packages explicitly
            for pkg in _AUTOMATION_PACKAGES:
                pkgdir = os.path.join(path, pkg)
                if os.path.isdir(pkgdir):
                    search_roots.add(pkgdir)

    # ── JS/TS sources ───────────────────────────────────────────────────────
    if include_js and any(ext in extensions for ext in (".js", ".ts", ".mjs")):
        for nm in _find_node_modules(search_roots):
            search_roots.add(nm)

    # ── User-supplied directories ───────────────────────────────────────────
    if extra_dirs:
        for d in extra_dirs:
            if os.path.isdir(d):
                search_roots.add(d)

    texts: List[str] = []
    seen:  set = set()
    total_bytes = 0

    for root in sorted(search_roots):
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden dirs and known junk dirs
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".")
                and d not in _SKIP_DIRS
            ]
            for fname in filenames:
                if not any(fname.endswith(ext) for ext in extensions):
                    continue
                # Skip minified JS (usually *.min.js or very long single-line files)
                if fname.endswith(".min.js") or fname.endswith(".min.ts"):
                    continue
                fpath = os.path.realpath(os.path.join(dirpath, fname))
                if fpath in seen:
                    continue
                seen.add(fpath)
                try:
                    sz = os.path.getsize(fpath)
                    if sz < min_file_size or sz > max_file_size:
                        continue
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    # Extra filter: skip minified content (single line > 500 chars)
                    first_line = text.split("\n", 1)[0]
                    if len(first_line) > 500:
                        continue
                    texts.append(text)
                    total_bytes += len(text)
                except (OSError, PermissionError):
                    pass

    mb = total_bytes / 1_048_576
    print(f"[dataset] collected {len(texts)} files, {mb:.1f} MB of source code")
    return texts


# ── Docstring pair extraction (for instruction fine-tuning) ───────────────────

def extract_docstring_pairs(texts: List[str]) -> List[tuple]:
    """
    Walk source files and extract (docstring, function_body) pairs.
    Returns a list of (instruction_str, code_str) tuples.
    """
    import ast
    pairs = []
    for text in texts:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node)
            if not docstring or len(docstring) < 20:
                continue
            # Reconstruct body without the docstring node
            body_nodes = node.body[1:] if isinstance(node.body[0], ast.Expr) else node.body
            if not body_nodes:
                continue
            try:
                # Use the raw source lines
                start = node.lineno - 1
                end   = (body_nodes[-1].end_lineno or node.end_lineno)
                lines = text.splitlines()
                body  = "\n".join(lines[start:end])
                if len(body) > 50:
                    pairs.append((docstring.strip(), body.strip()))
            except Exception:
                pass
    print(f"[dataset] extracted {len(pairs)} docstring pairs")
    return pairs


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class CodeDataset(Dataset):
    """
    Sliding-window dataset over a flat token array.
    Each sample is a window of `seq_len` tokens; targets are shifted by 1.
    FIM augmentation is applied at `fim_rate` probability.
    """

    def __init__(
        self,
        token_ids: List[int],
        cfg: ModelConfig,
        fim_rate: float = 0.5,
    ):
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.cfg     = cfg
        self.seq_len = cfg.max_seq_len
        self.fim_rate = fim_rate
        # Number of complete windows
        self.n = max(0, (len(self.data) - 1) // self.seq_len)
        print(f"[dataset] {len(self.data):,} tokens → {self.n:,} training windows")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].tolist()

        # Pad if needed (last window may be short)
        if len(chunk) < self.seq_len + 1:
            chunk += [self.cfg.pad_id] * (self.seq_len + 1 - len(chunk))

        # Apply FIM augmentation
        chunk = fim_transform(chunk, self.cfg, self.fim_rate)

        # Ensure exactly seq_len+1 tokens after FIM (truncate or pad)
        if len(chunk) > self.seq_len + 1:
            chunk = chunk[: self.seq_len + 1]
        while len(chunk) < self.seq_len + 1:
            chunk.append(self.cfg.pad_id)

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


class InstructDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    Loss is computed only on the assistant (output) tokens.
    """

    def __init__(
        self,
        pairs: List[tuple],       # (instruction, code) pairs
        tokenizer,
        cfg: ModelConfig,
    ):
        self.samples = []
        sys_tok  = cfg.sys_id
        user_tok = cfg.user_id
        asst_tok = cfg.asst_id
        eos_tok  = cfg.eos_id
        pad_tok  = cfg.pad_id
        seq_len  = cfg.max_seq_len

        system_text = (
            "You are an expert coding assistant. "
            "Write clean, efficient, well-documented code."
        )
        sys_ids = [sys_tok] + tokenizer.encode(system_text)

        for instruction, code in pairs:
            user_ids = [user_tok] + tokenizer.encode(instruction)
            asst_ids = [asst_tok] + tokenizer.encode(code) + [eos_tok]
            full = sys_ids + user_ids + asst_ids

            # Build label mask: -100 (ignore) for everything before assistant turn
            labels = [-100] * (len(sys_ids) + len(user_ids)) + asst_ids

            # Truncate / pad to seq_len
            full   = full[:seq_len]
            labels = labels[:seq_len]
            pad_n  = seq_len - len(full)
            full   += [pad_tok] * pad_n
            labels += [-100]   * pad_n

            self.samples.append((
                torch.tensor(full,   dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
