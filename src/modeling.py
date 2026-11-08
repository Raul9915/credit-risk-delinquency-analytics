"""Probability-of-default (PD) model: logistic-regression baseline vs. gradient boosting.

Design choices
- Stratified train / validation / test split (the data has no calendar dates,
  so an out-of-time split is not possible - stated as a limitation).
- Every preprocessing step is fitted on the training split only (no leakage).
- No class re-weighting: the output must be a probability, not just a ranking.
- Optional isotonic calibration fitted on the validation split.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

log = logging.getLogger(__name__)

ID_COL, TARGET_COL = "SK_ID_CURR", "TARGET"
# Columns kept for business reporting (also used as features).
SEGMENT_COLS = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]


@dataclass
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]

    @property
    def all(self) -> list[str]:
        return self.categorical + self.numeric


def feature_spec(df: pd.DataFrame) -> FeatureSpec:
    """Infer feature columns. ID and TARGET are always excluded."""
    candidates = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    categorical = [c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in candidates if c not in categorical]
    return FeatureSpec(numeric=numeric, categorical=categorical)


def clean_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = df.copy()
    out[spec.numeric] = out[spec.numeric].astype("float64").replace([np.inf, -np.inf], np.nan)
    cats = out[spec.categorical].astype("object")
    out[spec.categorical] = cats.where(cats.notna(), np.nan)
    return out


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_data(df: pd.DataFrame, test_size: float, valid_size: float, seed: int) -> Splits:
    rest, test = train_test_split(df, test_size=test_size, stratify=df[TARGET_COL],
                                  random_state=seed)
    train, valid = train_test_split(rest, test_size=valid_size, stratify=rest[TARGET_COL],
                                    random_state=seed)
    return Splits(train.reset_index(drop=True), valid.reset_index(drop=True),
                  test.reset_index(drop=True))


def build_logistic(spec: FeatureSpec, params: dict[str, Any], seed: int) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                      subsample=200_000, random_state=seed)),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005)),
    ])
    pre = ColumnTransformer([("cat", categorical, spec.categorical),
                             ("num", numeric, spec.numeric)])
    model = LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    return Pipeline([("pre", pre), ("model", model)])


def build_gradient_boosting(spec: FeatureSpec, params: dict[str, Any], seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                encoded_missing_value=np.nan), spec.categorical),
         ("num", "passthrough", spec.numeric)])
    is_categorical = [True] * len(spec.categorical) + [False] * len(spec.numeric)
    model = HistGradientBoostingClassifier(categorical_features=is_categorical,
                                           random_state=seed, **params)
    return Pipeline([("pre", pre), ("model", model)])


class CalibratedModel:
    """Wraps a fitted model and maps its scores through an isotonic regression."""

    def __init__(self, base: Pipeline, calibrator: IsotonicRegression | None):
        self.base = base
        self.calibrator = calibrator

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated score: strictly ranks applicants (used for risk bands)."""
        return self.base.predict_proba(X)[:, 1]

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_score(X)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw


@dataclass
class TrainingResult:
    spec: FeatureSpec
    splits: Splits
    models: dict[str, CalibratedModel]
    validation_auc: dict[str, float]
    selected: str
    band_edges: np.ndarray = field(default_factory=lambda: np.array([]))


def train_models(df: pd.DataFrame, cfg: dict[str, Any]) -> TrainingResult:
    seed = cfg["random_seed"]
    spec = feature_spec(df)
    df = clean_frame(df, spec)
    splits = split_data(df, cfg["test_size"], cfg["validation_size"], seed)
    log.info("Split sizes - train: %s, valid: %s, test: %s | default rate train: %.2f%%",
             f"{len(splits.train):,}", f"{len(splits.valid):,}", f"{len(splits.test):,}",
             100 * splits.train[TARGET_COL].mean())

    builders = {
        "logistic_regression": build_logistic(spec, cfg["logistic_regression"], seed),
        "gradient_boosting": build_gradient_boosting(spec, cfg["gradient_boosting"], seed),
    }
    X_tr, y_tr = splits.train[spec.all], splits.train[TARGET_COL]
    X_va, y_va = splits.valid[spec.all], splits.valid[TARGET_COL]

    models, val_auc = {}, {}
    for name, pipe in builders.items():
        log.info("Training %s on %d features...", name, len(spec.all))
        pipe.fit(X_tr, y_tr)
        raw_va = pipe.predict_proba(X_va)[:, 1]
        calibrator = None
        if cfg.get("calibrate", True):
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            calibrator.fit(raw_va, y_va)
        models[name] = CalibratedModel(pipe, calibrator)
        val_auc[name] = float(roc_auc_score(y_va, raw_va))
        log.info("  validation AUC %s: %.4f", name, val_auc[name])

    selected = max(val_auc, key=val_auc.get)
    log.info("Selected model (highest validation AUC): %s", selected)

    # Risk bands: score quintiles on the validation split (never on the test split).
    # The raw score is used because isotonic calibration creates ties (a step
    # function) that could collapse two bands; the ranking is identical.
    n_bands = len(cfg["risk_bands"])
    val_score = models[selected].predict_score(X_va)
    edges = np.quantile(val_score, np.linspace(0, 1, n_bands + 1)[1:-1])
    return TrainingResult(spec, splits, models, val_auc, selected, edges)


def assign_band(pd_values: np.ndarray, edges: np.ndarray, labels: list[str]) -> np.ndarray:
    """Map PDs to bands: below edges[0] -> labels[0] (lowest risk), and so on."""
    return np.asarray(labels, dtype=object)[np.searchsorted(edges, pd_values, side="right")]
