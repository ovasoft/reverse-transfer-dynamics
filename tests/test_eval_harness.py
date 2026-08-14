import pytest

from rtd.eval.harness import accuracy_gain_rate, BENCHMARK_TASKS


def test_benchmark_tasks_covers_the_fixed_four():
    assert set(BENCHMARK_TASKS.keys()) == {"blimp", "mnli", "mmlu", "humaneval"}


def test_accuracy_gain_rate_basic():
    # +0.04 accuracy over 800M tokens -> 0.05 per billion tokens
    rate = accuracy_gain_rate(acc_prev=0.20, acc_curr=0.24, tokens_between=800_000_000)
    assert rate == pytest.approx(0.05)


def test_accuracy_gain_rate_negative_for_regression():
    rate = accuracy_gain_rate(acc_prev=0.30, acc_curr=0.25, tokens_between=800_000_000)
    assert rate < 0


def test_accuracy_gain_rate_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        accuracy_gain_rate(0.2, 0.3, tokens_between=0)


def test_evaluate_checkpoint_requires_lm_eval():
    pytest.importorskip("lm_eval")  # only meaningful once the [evalh] extra is installed
