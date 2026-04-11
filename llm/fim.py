"""
Fill-In-the-Middle (FIM) data augmentation.

With probability `fim_rate`, a training sequence is rearranged from:
    [prefix ... middle ... suffix]
into:
    [FIM_PREFIX prefix FIM_SUFFIX suffix FIM_MIDDLE middle EOS]

This trains the model to predict the missing middle of code given the
surrounding context — the same technique used in Codex and StarCoder.
"""
import random
from typing import List
from llm.config import ModelConfig


def fim_transform(
    ids: List[int],
    cfg: ModelConfig,
    fim_rate: float = 0.5,
) -> List[int]:
    """
    Optionally apply FIM rearrangement to a list of token ids.
    Returns the (possibly transformed) sequence.
    """
    if random.random() >= fim_rate or len(ids) < 8:
        return ids

    # Pick two random cut points
    lo = random.randint(1, len(ids) - 2)
    hi = random.randint(lo + 1, len(ids) - 1)

    prefix = ids[:lo]
    middle = ids[lo:hi]
    suffix = ids[hi:]

    return (
        [cfg.fim_prefix_id]
        + prefix
        + [cfg.fim_suffix_id]
        + suffix
        + [cfg.fim_middle_id]
        + middle
        + [cfg.eos_id]
    )


def build_fim_prompt(prefix: str, suffix: str, tokenizer) -> List[int]:
    """
    Build a FIM-formatted prompt for inference (cursor completion).
    prefix  = code before the cursor
    suffix  = code after the cursor
    Returns token ids ready to feed into CodeGPT.generate().
    """
    cfg = ModelConfig()
    return (
        [cfg.fim_prefix_id]
        + tokenizer.encode(prefix)
        + [cfg.fim_suffix_id]
        + tokenizer.encode(suffix)
        + [cfg.fim_middle_id]
    )
