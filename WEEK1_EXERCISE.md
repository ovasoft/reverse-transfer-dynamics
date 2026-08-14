# Week 1: Consolidation Exercise

Goal: by the end of this week you should be able to read this codebase
with confidence, trust that it does what the protocol document says, and
be ready to start real experiment work in Week 2. This is not meant to be
stressful -- work through it at a reasonable pace, ask questions, and use
the demo/report walkthrough at the end of the week to surface anything
that's still confusing.

Read `docs/ARCHITECTURE.md` first -- it maps every step in the protocol
document to the file that implements it. Keep it open as a reference while
you work through the tasks below.

## 1. Environment setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

You should see everything pass except modules that need `torch`/`lm-eval`
(those `skip` rather than fail, if the `[train]`/`[evalh]` extras aren't
installed yet). Then:

```
pip install -e ".[train]"
pytest -q
```

Now the torch-gated tests (`test_transformer_adapter.py`,
`test_trainer_smoke.py`) should run too. **This step matters**: those two
test files were written and reviewed carefully but could not be executed
in the environment this repo was built in (no GPU/network access to
install torch there) -- you are the first people to actually run them. If
anything fails, that's real, useful signal; don't assume the code is
correct just because it reads correctly. Report anything you find.

## 2. Run the toy pipeline end-to-end

```
PYTHONPATH=src python3 scripts/toy_pipeline_demo.py
```

Read `src/rtd/data/metrics.py`'s docstrings alongside the output. Answer
for yourself (write a few sentences, doesn't need to be formal):
  - Why does M1 use a compression ratio specifically, rather than e.g.
    counting repeated substrings directly?
  - Why might M1 behave noisily on very short sequences (see the caveat
    the script prints)? What would you need to change to trust it more?
  - Looking at `find_pairs_music21_heuristic` in the same file: what kind
    of real musical dependency would this heuristic miss entirely?

## 3. Scavenger hunt

For each of these steps from the protocol document, find the file and
function that implements it (or would implement it, for the architecture
stubs) and write down the path:

  - Section 1, Step 2 (M1/M2/M3 computation)
  - Section 1, Step 6 (Phase 2 replay mixing)
  - Section 3, Step 1 (BLiMP replacing CoLA, and why)
  - Section 4, Step 1 (NIAH document construction)
  - Section 5, Step 1 (Latin Hypercube composition sampling)
  - Section 6, Step 3 (the RAS / attention-substitute table)
  - Section 7 (control models / TSI) -- this one isn't built yet. Skim
    the protocol document's Section 7 and sketch (in words, no code needed
    yet) which existing module(s) it would most naturally extend.

## 4. Implement the RNN adapter

`src/rtd/models/rnn.py` has the interface and hints; it's left
unimplemented on purpose (see that file's module docstring for why). Get
`pytest tests/test_rnn_exercise.py -v` fully green. If you get stuck,
`src/rtd/models/transformer.py` implements the exact same interface for a
different architecture -- read it side by side with the RNN skeleton.

## 5. End-of-week walkthrough

Be ready to show, briefly, for whichever piece you worked on most: what it
does, why it's built the way it is (tie it back to a specific step in the
protocol document), and anything you think is wrong, unclear, or worth
changing before Week 2 starts. Finding a real bug or a bad assumption this
week is a good outcome, not a bad one -- it's much cheaper to fix now than
after a real training run depends on it.
