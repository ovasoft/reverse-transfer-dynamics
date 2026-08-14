# Reverse Transfer Dynamics -- starter codebase

Companion codebase for the project protocol document *"Reverse Transfer
Dynamics: Quantifying Structural Bias in Music-to-Text Language Models"*
(the six-experiment protocol, Sections 1-7). This repo is infrastructure,
not results -- it exists so the team spends October-February running
experiments instead of building plumbing.

**Start here:** `docs/ARCHITECTURE.md` maps every protocol section/step to
the file that implements it. `WEEK1_EXERCISE.md` is the Week 1
onboarding exercise -- work through that before touching real experiment
work.

## Status (honest, as of repo creation)

Built and tested:
- Data pipeline: genre sharding, M1/M2/M3 metrics, ABC + REMI tokenizers
  (`src/rtd/data/`, `src/rtd/tokenize/`)
- Composition-sweep sampler + regression tooling for Experiment 2.4
  (`src/rtd/composition/`)
- NIAH document/scoring for Experiment 2.3 (`src/rtd/eval/niah.py`)
- Checkpoint manager + replay mixer (`src/rtd/train/checkpoint.py`,
  `replay.py`)
- The common model-adapter interface, plus a full reference
  implementation for Transformer/GPT2 (`src/rtd/models/base.py`,
  `transformer.py`)

Written but **not execution-verified** (this repo was built in a sandbox
without reliable access to install `torch` -- see below): the Phase 1/2
trainer (`src/rtd/train/trainer.py`) and the eval-harness wrapper
(`src/rtd/eval/harness.py`). These were reviewed carefully and have tests
waiting for them (`tests/test_trainer_smoke.py`,
`tests/test_transformer_adapter.py`), but **running those tests for the
first time, in a real environment, is genuinely part of Week 1** --
don't assume they're correct just because they read correctly.

Not built yet, intentionally left for the team:
- The RNN architecture adapter (`src/rtd/models/rnn.py`) -- this is the
  Week 1 exercise, see `WEEK1_EXERCISE.md`.
- Mamba/sLSTM/Hymba adapters (`src/rtd/models/{mamba,slstm,hymba}.py`) --
  stubs with detailed implementation notes in each docstring; real Week 2+
  engineering work for whoever owns Experiment 2.5.
- Experiment 2.1's exploratory-discovery pipeline (functional signatures,
  clustering, post-hoc characterization) and Experiment 2.6's circuit
  localization (control models, ablation, TSI, activation patching) --
  no code yet, `docs/ARCHITECTURE.md` notes what module these should
  become.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # lightweight deps, no torch
pytest -q                        # should pass, with torch/lm-eval tests skipped

pip install -e ".[train]"        # adds torch + transformers
pytest -q                        # torch-gated tests should now run too

pip install -e ".[arch]"         # mamba-ssm + xlstm, once you're implementing those adapters
pip install -e ".[evalh]"        # lm-eval, once running real benchmark evaluation
```

## Layout

```
src/rtd/
  data/        genre sharding, dataset acquisition, M1/M2/M3 metrics
  tokenize/    ABC tokenizer, REMI (MidiTok) wrapper, BPE vocab expansion
  models/      SequenceModelAdapter interface + one file per architecture
  train/       Phase 1/2 trainer, checkpoint manager, replay mixer
  eval/        BLiMP/MNLI/MMLU/HumanEval wrapper, NIAH
  composition/ Experiment 2.4's Latin Hypercube sweep + regression
configs/       toy config (tests/exercise) and real 0.6B config, plus fixed seeds
data_samples/  tiny real toy corpus (4 ABC tunes, 3 synthetic MIDI files) for tests/exercise
tests/         one file per module above
docs/          architecture map
```

## A note on dataset sources

`src/rtd/data/acquisition.py` has real download logic for Lakh MIDI and
TheSession-data (URLs verified via web search at repo creation time), and
an honest `TODO_VERIFY` stub for EMelodyGen (its hosting location wasn't
independently confirmed). Re-verify all of these before a real pull --
dataset hosting moves, and `record_provenance()` exists so you always know
exactly which version you trained on.
