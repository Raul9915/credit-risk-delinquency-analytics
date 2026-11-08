"""Model utilities, metrics and the full model/report step on the mini dataset."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.evaluation import classification_metrics, cutoff_strategy, decile_table, ks_statistic
from src.modeling import assign_band, feature_spec


def test_feature_spec_never_includes_id_or_target():
    df = pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0], "AMT_CREDIT": [1.0], "CODE_GENDER": ["F"]})
    spec = feature_spec(df)
    assert "TARGET" not in spec.all and "SK_ID_CURR" not in spec.all
    assert spec.categorical == ["CODE_GENDER"] and spec.numeric == ["AMT_CREDIT"]


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_metrics_gini_relation():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    m = classification_metrics(y, p)
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)


def test_assign_band_order():
    bands = assign_band(np.array([0.01, 0.05, 0.5]), np.array([0.02, 0.1]), ["A", "B", "C"])
    assert bands.tolist() == ["A", "B", "C"]


def test_cutoff_strategy_monotone_and_complete():
    y = np.array([0] * 90 + [1] * 10)
    p = np.linspace(0, 1, 100)          # the 10 bads have the highest PD
    cut = cutoff_strategy(y, p, 50, 100).set_index("approval_rate_pct")
    assert cut.loc[90, "default_rate_approved"] == 0
    assert cut.loc[100, "default_rate_approved"] == pytest.approx(0.1)
    assert cut.loc[90, "share_defaults_avoided"] == pytest.approx(1.0)


def test_decile_table_captures_all_defaults():
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 2, 1000), rng.uniform(size=1000)
    d = decile_table(y, p)
    assert len(d) == 10 and d["n_clients"].sum() == 1000
    assert d["cum_share_defaults_captured"].iloc[-1] == pytest.approx(1.0)


def test_end_to_end_model_and_reports(built_paths, settings):
    from src.modeling import train_models
    from src.reporting import build_reports
    from src.visualize import build_charts

    cfg = {**settings["modeling"],
           "gradient_boosting": {**settings["modeling"]["gradient_boosting"], "max_iter": 20,
                                 "min_samples_leaf": 5, "early_stopping": False},
           "permutation_importance": {"sample_size": 200, "n_repeats": 1}}
    s = {**settings, "modeling": cfg}
    feats = duckdb.sql(f"SELECT * FROM '{(built_paths.layer('gold') / 'client_features.parquet').as_posix()}'").df()
    result = train_models(feats, cfg)
    tables = build_reports(built_paths, result, s)
    build_charts(built_paths, result, tables)

    expected = {"portfolio_kpis", "transition_matrix", "roll_rates", "vintage_curves",
                "risk_segments", "model_metrics", "model_deciles", "cutoff_strategy",
                "feature_importance"}
    written = {p.stem for p in built_paths.data_studio.glob("*.csv")}
    assert expected <= written
    for csv in built_paths.data_studio.glob("*.csv"):        # Data Studio field-name rules
        header = csv.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert all(h.replace("_", "").isalnum() and not h[0].isdigit() for h in header), csv.name
    assert (built_paths.reports / "results_summary.md").exists()
    assert len(list(built_paths.images.glob("*.png"))) == 8
