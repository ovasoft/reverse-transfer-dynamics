"""
Regression analysis for the composition sweep -- see protocol document,
Section 5 (Experiment 2.4), Steps 7-9, and Section 6 (Experiment 2.5),
Step 5 for the Architecture-extended version.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class CompositionRow:
    format: str
    genre: str
    m1: float
    m2: float
    m3: float
    delta_p: float
    architecture: str | None = None  # only set once Experiment 2.5 extends the sweep
    benchmark: str = "blimp"


def to_dataframe(rows: list[CompositionRow]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def fit_ols(df: pd.DataFrame, *, include_architecture: bool = False):
    """Fit the regression from Section 5 Step 8 (or Section 6 Step 5 if
    `include_architecture=True`, adding Architecture and its interactions
    with M1-M3)."""
    if include_architecture:
        formula = (
            "delta_p ~ C(format) + C(genre) + m1 + m2 + m3 + C(architecture) "
            "+ m1:C(architecture) + m2:C(architecture) + m3:C(architecture)"
        )
    else:
        formula = "delta_p ~ C(format) + C(genre) + m1 + m2 + m3"
    return smf.ols(formula, data=df).fit()


def fit_random_forest(df: pd.DataFrame, *, include_architecture: bool = False) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """Non-linear robustness check (Section 5 Step 8, second bullet).
    Returns the fitted model and the one-hot-encoded feature frame used, so
    feature_importances_ can be matched back to column names."""
    cat_cols = ["format", "genre"] + (["architecture"] if include_architecture else [])
    X = pd.get_dummies(df[cat_cols + ["m1", "m2", "m3"]], columns=cat_cols)
    y = df["delta_p"]
    rf = RandomForestRegressor(n_estimators=200, random_state=0)
    rf.fit(X, y)
    return rf, X


def compute_vif(df: pd.DataFrame, columns: list[str] = ("m1", "m2", "m3")) -> pd.Series:
    """Variance inflation factor for the continuous predictors (Section 5
    Step 9: "check regression diagnostics: VIF for collinearity between
    M1-M3 and the naturally-varying Genre mix"). Pass genre-encoded dummy
    columns in `columns` too if you want VIF against genre specifically."""
    X = df[list(columns)].astype(float)
    X = X.assign(const=1.0)  # VIF is computed against a design matrix with an intercept
    vifs = {
        col: variance_inflation_factor(X.values, i)
        for i, col in enumerate(X.columns)
        if col != "const"
    }
    return pd.Series(vifs, name="VIF")
