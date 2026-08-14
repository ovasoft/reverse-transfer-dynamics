"""
Needle-in-a-Haystack test -- see protocol document, Section 4 (Experiment
2.3), Steps 1-3.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NIAHDocument:
    text: str
    context_length_tokens: int
    needle_depth_pct: int
    needle_fact: str
    question: str
    answer: str


DEFAULT_LENGTHS = (2000, 4000, 6000, 8000)
DEFAULT_DEPTHS_PCT = tuple(range(0, 101, 10))  # 0%, 10%, ..., 100%

# A small pool of synthetic (fact, question, answer) triples so the needle
# is never something the model could answer from pretraining knowledge --
# swap in your own for a larger, less guessable pool before real runs.
DEFAULT_NEEDLES = [
    ("The secret code word for this experiment is zephyrine-9.", "What is the secret code word?", "zephyrine-9"),
    ("The reference tune's tempo was fixed at 137 beats per minute.", "What was the tune's tempo?", "137 beats per minute"),
    ("Checkpoint Delta-7 was trained on exactly 812 shards.", "How many shards was Checkpoint Delta-7 trained on?", "812"),
]


def build_niah_document(
    filler_text_source: list[str],  # pool of sentences/paragraphs to pad with (e.g. RefinedWeb passages)
    *,
    context_length_tokens: int,
    needle_depth_pct: int,
    tokenizer,  # anything with .encode(str) -> list[int] and .decode(list[int]) -> str
    needle: tuple[str, str, str] | None = None,
    seed: int = 0,
) -> NIAHDocument:
    """Section 4, Step 1: sample filler up to `context_length_tokens`,
    insert `needle` at `needle_depth_pct` (0 = very start, 100 = very end).
    """
    if not 0 <= needle_depth_pct <= 100:
        raise ValueError("needle_depth_pct must be in [0, 100]")

    rng = random.Random(seed)
    fact, question, answer = needle or rng.choice(DEFAULT_NEEDLES)

    filler = " ".join(rng.sample(filler_text_source, k=min(len(filler_text_source), 200)))
    filler_ids = tokenizer.encode(filler, add_bos_eos=False)
    if len(filler_ids) < context_length_tokens:
        # repeat filler if the source pool isn't big enough to hit the target length
        reps = (context_length_tokens // max(len(filler_ids), 1)) + 1
        filler_ids = (filler_ids * reps)[:context_length_tokens]
    else:
        filler_ids = filler_ids[:context_length_tokens]

    needle_ids = tokenizer.encode(fact, add_bos_eos=False)
    insert_at = int(len(filler_ids) * (needle_depth_pct / 100.0))
    combined_ids = filler_ids[:insert_at] + needle_ids + filler_ids[insert_at:]

    return NIAHDocument(
        text=tokenizer.decode(combined_ids),
        context_length_tokens=context_length_tokens,
        needle_depth_pct=needle_depth_pct,
        needle_fact=fact,
        question=question,
        answer=answer,
    )


def build_niah_suite(
    filler_text_source: list[str],
    *,
    tokenizer,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    depths_pct: tuple[int, ...] = DEFAULT_DEPTHS_PCT,
    seed: int = 0,
) -> list[NIAHDocument]:
    """Section 4, Step 1, full sweep: every (length, depth) combination."""
    docs = []
    i = 0
    for length in lengths:
        for depth in depths_pct:
            docs.append(
                build_niah_document(
                    filler_text_source,
                    context_length_tokens=length,
                    needle_depth_pct=depth,
                    tokenizer=tokenizer,
                    seed=seed + i,
                )
            )
            i += 1
    return docs


def score_retrieval(model_answer: str, expected_answer: str) -> bool:
    """Section 4, Step 2: exact-match-ish scoring. Lowercased substring
    match rather than strict equality, since models rarely echo the answer
    with identical formatting/punctuation -- tighten this if it proves too
    lenient on your model's actual outputs."""
    return expected_answer.strip().lower() in model_answer.strip().lower()


def build_accuracy_matrix(
    docs: list[NIAHDocument],
    answers: list[str],
) -> dict[tuple[int, int], bool]:
    """Section 4, Step 3: (context_length, needle_depth) -> retrieved bool.
    Feed this into a pandas pivot / heatmap for the actual 2D matrix."""
    if len(docs) != len(answers):
        raise ValueError("docs and answers must be the same length")
    return {
        (d.context_length_tokens, d.needle_depth_pct): score_retrieval(a, d.answer)
        for d, a in zip(docs, answers)
    }
