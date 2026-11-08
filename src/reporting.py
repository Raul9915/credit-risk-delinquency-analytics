"""Reporting layer: Data Studio extracts (CSV), metrics and an auto-generated summary.

Every number written here is computed from the data; nothing is hard-coded.
CSV rules for Google Data Studio file upload: UTF-8, comma-separated, field
names with letters, digits and underscores only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import Paths
from src.evaluation import classification_metrics, cutoff_strategy, decile_table
from src.modeling import SEGMENT_COLS, TARGET_COL, TrainingResult, assign_band

log = logging.getLogger(__name__)
MIN_AT_RISK = 100   # vintage points with fewer credits at risk are flagged as unreliable


def _read(paths: Paths, layer: str, table: str, columns: str = "*") -> pd.DataFrame:
    file = (paths.layer(layer) / f"{table}.parquet").as_posix()
    return duckdb.sql(f"SELECT {columns} FROM '{file}'").df()


def _write_csv(df: pd.DataFrame, folder: Path, name: str) -> Path:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    float_cols = out.select_dtypes("float").columns
    out[float_cols] = out[float_cols].round(6)
    path = folder / f"{name}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    return path


# --------------------------------------------------------------------- portfolio
def portfolio_kpis(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["AGE_BAND"] = pd.cut(df["AGE_YEARS"], [0, 25, 35, 45, 55, 65, 120],
                            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
                            right=False).astype(str)
    df["BUREAU_HISTORY"] = np.where(df["HAS_BUREAU_HISTORY"] == 1, "With bureau history",
                                    "No bureau history")
    df["OVERALL"] = "All applicants"
    segments = {"Overall": "OVERALL", "Contract type": "NAME_CONTRACT_TYPE",
                "Income type": "NAME_INCOME_TYPE", "Education": "NAME_EDUCATION_TYPE",
                "Age band": "AGE_BAND", "Bureau history": "BUREAU_HISTORY"}
    frames = []
    for seg_type, col in segments.items():
        g = (df.groupby(col, dropna=False)[TARGET_COL].agg(n_clients="size", n_defaults="sum")
               .reset_index().rename(columns={col: "segment_value"}))
        g.insert(0, "segment_type", seg_type)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["segment_value"] = out["segment_value"].fillna("Unknown").astype(str)
    out["default_rate"] = out["n_defaults"] / out["n_clients"]
    return out


# -------------------------------------------------------------------- delinquency
def roll_rates(transitions: pd.DataFrame) -> pd.DataFrame:
    t = transitions.copy()
    t["move"] = np.select(
        [t["TO_ORDER"] == -1, t["TO_ORDER"] > t["FROM_ORDER"],
         t["TO_ORDER"] == t["FROM_ORDER"]],
        ["closed", "roll_forward", "stay"], default="improve")
    wide = (t.pivot_table(index=["CREDIT_TYPE", "FROM_STATE", "FROM_ORDER"], columns="move",
                          values="N_TRANSITIONS", aggfunc="sum", fill_value=0)
              .reset_index())
    for col in ("closed", "roll_forward", "stay", "improve"):
        if col not in wide:
            wide[col] = 0
    total = wide[["closed", "roll_forward", "stay", "improve"]].sum(axis=1)
    out = wide[["CREDIT_TYPE", "FROM_STATE", "FROM_ORDER"]].copy()
    out["n_transitions"] = total
    for col in ("roll_forward", "stay", "improve", "closed"):
        out[f"pct_{col}"] = wide[col] / total
    return out.sort_values(["CREDIT_TYPE", "FROM_ORDER"]).reset_index(drop=True)


# ------------------------------------------------------------------------ model
def score_test(result: TrainingResult, labels: list[str]) -> pd.DataFrame:
    test = result.splits.test
    model = result.models[result.selected]
    out = test[["SK_ID_CURR", TARGET_COL, *SEGMENT_COLS]].copy()
    out["PD"] = model.predict_pd(test[result.spec.all])
    out["SCORE"] = model.predict_score(test[result.spec.all])
    out["RISK_BAND"] = assign_band(out["SCORE"].to_numpy(), result.band_edges, labels)
    return out


def model_metrics(result: TrainingResult) -> pd.DataFrame:
    rows = []
    for name, model in result.models.items():
        for split in ("valid", "test"):
            data = getattr(result.splits, split)
            m = classification_metrics(data[TARGET_COL].to_numpy(),
                                       model.predict_pd(data[result.spec.all]))
            rows.append({"model": name, "split": split,
                         "selected": int(name == result.selected), **m})
    return pd.DataFrame(rows)


def risk_segments(scores: pd.DataFrame) -> pd.DataFrame:
    g = (scores.groupby(["RISK_BAND", *SEGMENT_COLS], dropna=False)
               .agg(n_clients=(TARGET_COL, "size"), n_defaults=(TARGET_COL, "sum"),
                    sum_pd=("PD", "sum"))
               .reset_index())
    g[SEGMENT_COLS] = g[SEGMENT_COLS].fillna("Unknown")
    g["default_rate"] = g["n_defaults"] / g["n_clients"]
    g["avg_pd"] = g["sum_pd"] / g["n_clients"]
    return g


def feature_importance(result: TrainingResult, sample_size: int, n_repeats: int,
                       seed: int, top: int = 20) -> pd.DataFrame:
    """Permutation importance (drop in ROC-AUC) of the selected model on test data."""
    test = result.splits.test
    sample = test.sample(n=min(sample_size, len(test)), random_state=seed)
    pipe = result.models[result.selected].base
    imp = permutation_importance(pipe, sample[result.spec.all], sample[TARGET_COL],
                                 scoring="roc_auc", n_repeats=n_repeats,
                                 random_state=seed, n_jobs=-1)
    df = pd.DataFrame({"feature": result.spec.all, "auc_drop_mean": imp.importances_mean,
                       "auc_drop_std": imp.importances_std})
    return df.sort_values("auc_drop_mean", ascending=False).head(top).reset_index(drop=True)


# ----------------------------------------------------------------------- export
def build_reports(paths: Paths, result: TrainingResult, settings: dict) -> dict[str, pd.DataFrame]:
    mcfg, scfg = settings["modeling"], settings["strategy"]
    features = _read(paths, "gold", "client_features",
                     "SK_ID_CURR, TARGET, AGE_YEARS, HAS_BUREAU_HISTORY, "
                     + ", ".join(SEGMENT_COLS))
    transitions = _read(paths, "gold", "fct_transitions")
    vintage = _read(paths, "gold", "vintage_curves")
    vintage["IS_RELIABLE"] = (vintage["N_AT_RISK"] >= MIN_AT_RISK).astype(int)

    scores = score_test(result, mcfg["risk_bands"])
    y, p, rank = (scores[TARGET_COL].to_numpy(), scores["PD"].to_numpy(),
                  scores["SCORE"].to_numpy())

    tables = {
        "portfolio_kpis": portfolio_kpis(features),
        "transition_matrix": transitions.sort_values(["CREDIT_TYPE", "FROM_ORDER", "TO_ORDER"]),
        "roll_rates": roll_rates(transitions),
        "vintage_curves": vintage.sort_values(["CREDIT_TYPE", "VINTAGE_COHORT", "MONTHS_ON_BOOK"]),
        "risk_segments": risk_segments(scores),
        "model_metrics": model_metrics(result),
        "model_deciles": decile_table(y, p, rank_score=rank),
        "cutoff_strategy": cutoff_strategy(y, p, scfg["approval_rate_min"],
                                           scfg["approval_rate_max"], rank_score=rank),
        "feature_importance": feature_importance(
            result, mcfg["permutation_importance"]["sample_size"],
            mcfg["permutation_importance"]["n_repeats"], mcfg["random_seed"]),
    }
    for name, df in tables.items():
        _write_csv(df, paths.data_studio, name)
        log.info("Data Studio extract: %s.csv (%s rows)", name, f"{len(df):,}")

    scores.to_parquet(paths.layer("gold") / "test_scores.parquet", index=False)
    write_metrics_json(paths, result, tables)
    write_summary(paths, result, tables)
    return tables


def write_metrics_json(paths: Paths, result: TrainingResult, tables: dict) -> None:
    payload = {
        "selected_model": result.selected,
        "validation_auc": result.validation_auc,
        "risk_band_score_edges": [float(e) for e in result.band_edges],
        "metrics": tables["model_metrics"].to_dict(orient="records"),
        "n_features": len(result.spec.all),
    }
    (paths.reports / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.{digits}f}%"


def write_summary(paths: Paths, result: TrainingResult, tables: dict) -> None:
    """Markdown summary with the actual results of the last run."""
    mm = tables["model_metrics"]
    test_sel = mm[(mm["split"] == "test") & (mm["selected"] == 1)].iloc[0]
    overall = tables["portfolio_kpis"].query("segment_type == 'Overall'").iloc[0]
    rr = tables["roll_rates"].query("CREDIT_TYPE == 'All'")
    vin = tables["vintage_curves"].query("CREDIT_TYPE == 'All' and IS_RELIABLE == 1")
    bands = (tables["risk_segments"].groupby("RISK_BAND")[["n_clients", "n_defaults"]].sum())
    bands["default_rate"] = bands["n_defaults"] / bands["n_clients"]
    cut = tables["cutoff_strategy"].set_index("approval_rate_pct")

    lines = [
        "# Results summary (auto-generated)",
        "",
        "Generated by `python -m src.pipeline` from the Kaggle Home Credit Default Risk data. ",
        "All figures are computed on historical data; strategy figures are **estimates** on a ",
        "held-out test set, not realised business results.",
        "",
        "## Portfolio",
        f"- Applicants: {int(overall['n_clients']):,} | observed default rate (TARGET): "
        f"{_pct(overall['default_rate'], 2)}",
        "",
        "## Model performance (held-out test set)",
        "",
        "| Model | AUC | Gini | KS | PR-AUC | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in mm[mm["split"] == "test"].iterrows():
        tag = " (selected)" if r["selected"] else ""
        lines.append(f"| {r['model']}{tag} | {r['auc']:.3f} | {r['gini']:.3f} | {r['ks']:.3f} "
                     f"| {r['pr_auc']:.3f} | {r['brier']:.4f} |")
    lines += ["", "## Delinquency roll rates (bureau credits, all types)", "",
              "| From state | Roll forward | Stay | Improve | Closed |", "|---|---|---|---|---|"]
    for _, r in rr.iterrows():
        lines.append(f"| {r['FROM_STATE']} | {_pct(r['pct_roll_forward'])} | {_pct(r['pct_stay'])} "
                     f"| {_pct(r['pct_improve'])} | {_pct(r['pct_closed'])} |")
    lines += ["", "## Cumulative 90+ DPD rate by months on book (all credit types)", "",
              "| Cohort | 12 m | 24 m | 36 m |", "|---|---|---|---|"]
    for cohort, g in vin.groupby("VINTAGE_COHORT"):
        g = g.set_index("MONTHS_ON_BOOK")["CUMULATIVE_DEFAULT_RATE"]
        lines.append(f"| {cohort} | " + " | ".join(_pct(g.get(k, np.nan), 2) for k in (12, 24, 36))
                     + " |")
    lines += ["", "## Risk bands (test set)", "", "| Band | Clients | Default rate |", "|---|---|---|"]
    for band, r in bands.iterrows():
        lines.append(f"| {band} | {int(r['n_clients']):,} | {_pct(r['default_rate'], 2)} |")
    lines += ["", "## Cut-off strategy (test set, estimated)", "",
              "| Approval rate | Default rate of approved | Defaults avoided |", "|---|---|---|"]
    for rate in (70, 80, 90, 100):
        if rate in cut.index:
            r = cut.loc[rate]
            lines.append(f"| {rate}% | {_pct(r['default_rate_approved'], 2)} "
                         f"| {_pct(r['share_defaults_avoided'])} |")
    lines += ["", f"Selected model: **{result.selected}**; test AUC {test_sel['auc']:.3f}.", ""]
    (paths.reports / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
