"""
Zero-shot evaluation wrapper -- see protocol document, Section 3
(Experiment 2.2), Steps 1-2: BLiMP, MNLI, MMLU, HumanEval, all scored
zero-shot (no fine-tuning, no classifier head).

Wraps EleutherAI's `lm-evaluation-harness` rather than reimplementing
benchmark loading/scoring -- it already supports all four tasks with the
exact scoring methods the protocol calls for (log-likelihood comparison
for BLiMP/MNLI/MMLU, generation + execution for HumanEval). This module's
job is just to adapt our SequenceModelAdapter interface to what lm-eval
expects (an object with a `generate`/log-likelihood-scoring interface),
and to package results the way our checkpoint/regression tooling wants
them.

Requires `lm-eval` (pyproject.toml [evalh] extra) -- heavy dependency,
not installed/execution-tested in the environment this repo was built in.
"""
from __future__ import annotations

from dataclasses import dataclass

# Section 3 Step 1: exactly these four tasks, this is the fixed set referenced
# throughout the protocol document (Sections 5, 7 both re-use this list).
BENCHMARK_TASKS = {
    "blimp": "blimp",               # lm-eval task group name
    "mnli": "mnli",                 # scored via label-token log-likelihood, not a classifier head
    "mmlu": "mmlu",                 # scored via answer-choice-token log-likelihood
    "humaneval": "humaneval",       # zero/few-shot generation, pass@1
}


@dataclass
class BenchmarkResult:
    benchmark: str
    accuracy: float
    n_examples: int
    raw: dict  # full lm-eval result dict, kept for anything not captured above


def evaluate_checkpoint(model_adapter, tokenizer, *, tasks: list[str] | None = None, limit: int | None = None) -> dict[str, BenchmarkResult]:
    """Run the zero-shot battery against one checkpoint.

    `model_adapter` must be a SequenceModelAdapter (src/rtd/models/base.py);
    this function wraps it in lm-eval's HFLM-style interface. `limit` caps
    the number of examples per task -- use a small limit (e.g. 50) for fast
    iteration/debugging, and None (full benchmark) for real results.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or list(BENCHMARK_TASKS.values())

    # lm-eval's HFLM wraps a transformers-style model directly; our
    # TransformerAdapter's underlying HF model is exposed as `.model`, so
    # this works for that adapter as-is. Non-HF-backed adapters (Mamba,
    # sLSTM, RNN) will need a small custom lm_eval.api.model.LM subclass
    # instead of HFLM -- see lm-eval's docs on adding a custom model, and
    # note that HFLM's log-likelihood scoring (which BLiMP/MNLI/MMLU all
    # rely on) needs `attention_or_state`-independent access to per-token
    # logits, which every SequenceModelAdapter already exposes via
    # `forward().logits` regardless of architecture.
    lm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer)

    raw_results = simple_evaluate(model=lm, tasks=tasks, limit=limit)

    out: dict[str, BenchmarkResult] = {}
    for task_name, metrics in raw_results["results"].items():
        acc_key = next((k for k in metrics if k.startswith("acc")), None)
        out[task_name] = BenchmarkResult(
            benchmark=task_name,
            accuracy=metrics.get(acc_key, float("nan")) if acc_key else float("nan"),
            n_examples=raw_results.get("n-samples", {}).get(task_name, {}).get("effective", -1),
            raw=metrics,
        )
    return out


def accuracy_gain_rate(acc_prev: float, acc_curr: float, tokens_between: int) -> float:
    """Section 3 Step 3, second bullet: accuracy points gained per billion
    Phase 2 tokens, the accuracy-side analog of dL/dt."""
    billions = tokens_between / 1e9
    if billions <= 0:
        raise ValueError("tokens_between must be positive")
    return (acc_curr - acc_prev) / billions
