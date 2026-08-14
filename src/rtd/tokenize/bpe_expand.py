"""
BPE vocabulary expansion for Phase 2 -- see protocol document, Section 1
Step 5 ("Fit BPE merges on a RefinedWeb sample; extend Model A/B's
embedding and output layers to include the new text subword vocabulary,
while preserving the existing music-token embeddings unchanged").

Uses Hugging Face `tokenizers` (fast BPE trainer) for the merge-fitting
step, and `transformers`' `resize_token_embeddings` for the model-side
expansion. Both are optional deps (see pyproject.toml [train] extra).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def fit_bpe_merges(texts: Iterable[str], *, vocab_size: int, save_path: str | Path) -> None:
    """Fit a byte-level BPE tokenizer on `texts` and save it to `save_path`.
    `vocab_size` is the TOTAL vocab size for the new text tokenizer (not an
    increment) -- the merge step with the music vocab happens separately,
    in `expand_model_embeddings` below."""
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))


def merge_vocabs(music_vocab: dict[str, int], text_bpe_vocab: dict[str, int]) -> dict[str, int]:
    """Combine a music-token vocab (from ABCTokenizer / RemiTokenizerWrapper)
    with a newly-fit text BPE vocab into one id space, music tokens first so
    existing Phase-1 embedding rows keep their ids unchanged.

    Returns the combined vocab; call `expand_model_embeddings` with
    `len(combined_vocab)` as the new size.
    """
    combined = dict(music_vocab)
    next_id = max(combined.values(), default=-1) + 1
    for tok in text_bpe_vocab:
        if tok not in combined:
            combined[tok] = next_id
            next_id += 1
    return combined


def expand_model_embeddings(model, new_vocab_size: int):
    """Grow `model`'s input/output embedding matrices to `new_vocab_size`,
    in place, preserving all existing rows (this is exactly
    `transformers`' `resize_token_embeddings` -- wrapped here so callers
    don't need to know that detail, and so we have one place to add
    custom init for the new rows if the default turns out not to be good
    enough)."""
    model.resize_token_embeddings(new_vocab_size)
    return model
