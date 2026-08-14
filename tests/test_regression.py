import numpy as np
import pandas as pd

from rtd.composition.regression import (
    CompositionRow,
    to_dataframe,
    fit_ols,
    fit_random_forest,
    compute_vif,
)


def _synthetic_rows(n=200, seed=0) -> list[CompositionRow]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        fmt = rng.choice(["ABC", "MIDI"])
        genre = rng.choice(["Classical", "Jazz", "Electronic", "Folk"])
        m1 = rng.uniform(0, 1)
        m2 = rng.uniform(0, 1)
        m3 = rng.uniform(1, 200)
        # ground truth: delta_p driven mostly by m1, plus noise -- regression
        # should recover a positive, significant m1 coefficient
        delta_p = 2.0 * m1 + 0.01 * m3 + rng.normal(0, 0.05)
        rows.append(CompositionRow(format=fmt, genre=genre, m1=m1, m2=m2, m3=m3, delta_p=delta_p))
    return rows


def test_to_dataframe_has_expected_columns():
    df = to_dataframe(_synthetic_rows(10))
    assert {"format", "genre", "m1", "m2", "m3", "delta_p"}.issubset(df.columns)


def test_ols_recovers_dominant_m1_effect():
    df = to_dataframe(_synthetic_rows(300, seed=1))
    result = fit_ols(df)
    assert result.params["m1"] > 1.0  # true coefficient is 2.0, allow slack
    assert result.pvalues["m1"] < 0.01


def test_random_forest_ranks_m1_as_important():
    df = to_dataframe(_synthetic_rows(300, seed=2))
    rf, X = fit_random_forest(df)
    importances = dict(zip(X.columns, rf.feature_importances_))
    assert importances["m1"] > importances["m2"]


def test_compute_vif_runs_and_returns_series():
    df = to_dataframe(_synthetic_rows(100, seed=3))
    vif = compute_vif(df)
    assert set(vif.index) == {"m1", "m2", "m3"}
    assert (vif > 0).all()
