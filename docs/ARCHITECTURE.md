# Architecture map: protocol document -> code

This maps each section of the project protocol document ("Reverse Transfer
Dynamics... Experimental Protocol") to where it lives (or will live) in
this codebase. Section/step numbers refer to that document.

## Section 1 -- Shared Pretraining Pipeline

| Step | What it does | Where |
|---|---|---|
| 1 | Acquire + shard music corpus | `src/rtd/data/acquisition.py`, `src/rtd/data/sharding.py` |
| 2 | Compute M1/M2/M3 per shard | `src/rtd/data/metrics.py` |
| 3 | Tokenize to Phase 1 budget | `src/rtd/tokenize/abc_tokenizer.py`, `src/rtd/tokenize/remi_tokenizer.py` |
| 4 | Phase 1 music pretraining | `src/rtd/train/trainer.py::run_phase1` |
| 5 | Expand vocabulary for text | `src/rtd/tokenize/bpe_expand.py` |
| 6 | Phase 2 text adaptation + replay | `src/rtd/train/trainer.py::run_phase2`, `src/rtd/train/replay.py`, `src/rtd/train/checkpoint.py` |

## Section 2 -- Experiment 2.1 (Representational Layer Recycling)

Not yet built. Depends on: `SequenceModelAdapter.get_attention_or_state()`
(`src/rtd/models/base.py`) for the extraction step, and Phase 1/2
checkpoints existing. The unsupervised functional-signature and clustering
work (Steps 3-4) would naturally live in a new `src/rtd/interp/` module --
this is good Week 2+ scope for whoever owns the discovery track.

## Section 3 -- Experiment 2.2 (Convergence Velocity)

| Step | What it does | Where |
|---|---|---|
| 1-2 | BLiMP/MNLI/MMLU/HumanEval, zero-shot | `src/rtd/eval/harness.py` |
| 3 | Accuracy gain rate (accuracy-side dL/dt) | `src/rtd/eval/harness.py::accuracy_gain_rate` |

Note why BLiMP replaces CoLA: `src/rtd/eval/harness.py`'s module docstring
and `BENCHMARK_TASKS` -- CoLA is a fine-tuned-classifier benchmark by
construction, which doesn't fit the checkpoint-by-checkpoint zero-shot
loop this experiment runs; BLiMP is zero-shot by design (minimal-pairs,
log-likelihood comparison).

## Section 4 -- Experiment 2.3 (Needle-in-a-Haystack)

`src/rtd/eval/niah.py` -- document construction, retrieval scoring, and
the 2D (context length x needle depth) accuracy matrix builder.

## Section 5 -- Experiment 2.4 (Composition Sweep)

| Step | What it does | Where |
|---|---|---|
| 1 | Latin Hypercube sample over M1xM2xM3 | `src/rtd/composition/sweep.py::sample_composition_targets` |
| 1 (cont.) | Build corpus for a target point | `src/rtd/composition/sweep.py::build_corpus_for_target`, `achieved_metrics` |
| 2-4 | Pretrain composition models + baseline | `src/rtd/train/trainer.py` (reused at smaller scale) |
| 5 | Evaluate | `src/rtd/eval/harness.py` (same battery as 2.2) |
| 6-9 | Regression + diagnostics | `src/rtd/composition/regression.py` |

## Section 6 -- Experiment 2.5 (Cross-Architecture Generalization)

| Step | What it does | Where |
|---|---|---|
| 1 | Instantiate 6 architectures | `src/rtd/models/{transformer,mamba,slstm,hymba,rnn}.py`, all implementing `src/rtd/models/base.py::SequenceModelAdapter` |
| 3 | Interpretability substitute per architecture | `SequenceModelAdapter.get_attention_or_state()` -- each adapter's docstring explains its own architecture-specific return shape |
| 5 | Extend composition sweep with Architecture factor | `src/rtd/composition/regression.py::fit_ols(..., include_architecture=True)` |

Architecture adapter status: Transformer/GPT2 fully implemented
(`transformer.py`) and is the reference/golden-path adapter. RNN is the
Week 1 exercise (`rnn.py`, unimplemented on purpose). Mamba, sLSTM, Hymba
are stubs (`mamba.py`, `slstm.py`, `hymba.py`) with detailed implementation
notes in each file's docstring -- real Week 2+ engineering work.

## Section 7 -- Experiment 2.6 (Circuit Localization)

Not yet built. This is the biggest piece of unbuilt scope in the repo.
Needs: matched control-model training (reuses `trainer.py` with
`replay_fraction=0.0` and no Phase 1), then layer/head/channel ablation
and TSI computation, which would naturally live in a new
`src/rtd/circuits/` module built on top of each adapter's
`get_attention_or_state()`.

## Cross-cutting

- `src/rtd/models/base.py` -- the interface every architecture adapter
  implements; read this before touching any specific architecture file.
- `configs/` -- YAML configs (toy scale for tests/exercises, real 0.6B
  scale for actual runs).
- `tests/` -- one test file per module above; torch-dependent tests use
  `pytest.importorskip("torch")` and skip cleanly if it's not installed.
