import pytest

from rtd.train.replay import ReplayMixer, cycle


def test_replay_fraction_roughly_matches_over_many_draws():
    primary = iter(f"P{i}" for i in range(100_000))
    replay = cycle(lambda: iter(f"R{i}" for i in range(10)))

    mixer = ReplayMixer(primary, replay, replay_fraction=0.05, seed=0)
    sources = [src for _, (_, src) in zip(range(20_000), mixer)]

    replay_rate = sources.count("replay") / len(sources)
    assert 0.03 < replay_rate < 0.07  # loose bound, this is a stochastic draw


def test_zero_replay_fraction_never_draws_replay():
    primary = iter(range(1000))
    replay = cycle(lambda: iter(["should never appear"]))

    mixer = ReplayMixer(primary, replay, replay_fraction=0.0)
    sources = [src for _, (_, src) in zip(range(500), mixer)]
    assert all(s == "primary" for s in sources)


def test_invalid_replay_fraction_raises():
    with pytest.raises(ValueError):
        ReplayMixer(iter([]), iter([]), replay_fraction=1.0)
    with pytest.raises(ValueError):
        ReplayMixer(iter([]), iter([]), replay_fraction=-0.1)


def test_cycle_restarts_exhausted_stream():
    gen = cycle(lambda: iter([1, 2, 3]))
    drawn = [next(gen) for _ in range(7)]
    assert drawn == [1, 2, 3, 1, 2, 3, 1]
