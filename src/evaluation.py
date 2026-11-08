"""Model evaluation and credit-strategy tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max distance between the score CDFs of goods and bads."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true: np.ndarray, pd_hat: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, pd_hat)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, pd_hat),
        "pr_auc": float(average_precision_score(y_true, pd_hat)),
        "brier": float(brier_score_loss(y_true, pd_hat)),
        "observed_default_rate": float(np.mean(y_true)),
        "mean_predicted_pd": float(np.mean(pd_hat)),
    }


def _rank_key(pd_hat: np.ndarray, rank_score: np.ndarray | None) -> np.ndarray:
    """Ranking key. Calibrated PDs contain ties (isotonic steps), so a strictly
    monotone raw score is preferred for ordering when available."""
    return np.asarray(pd_hat if rank_score is None else rank_score)


def decile_table(y_true: np.ndarray, pd_hat: np.ndarray, n_bins: int = 10,
                 rank_score: np.ndarray | None = None) -> pd.DataFrame:
    """Decile 1 = highest predicted risk. Includes calibration and lift columns."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat),
                       "key": _rank_key(pd_hat, rank_score)})
    df = df.sort_values("key", ascending=False, kind="mergesort").reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    out = df.groupby("decile").agg(n_clients=("y", "size"), n_defaults=("y", "sum"),
                                   avg_pd=("pd", "mean"), min_pd=("pd", "min"),
                                   max_pd=("pd", "max")).reset_index()
    base_rate = df["y"].mean()
    out["default_rate"] = out["n_defaults"] / out["n_clients"]
    out["lift"] = out["default_rate"] / base_rate
    out["cum_share_defaults_captured"] = out["n_defaults"].cumsum() / out["n_defaults"].sum()
    out["cum_share_clients"] = out["n_clients"].cumsum() / out["n_clients"].sum()
    return out


def cutoff_strategy(y_true: np.ndarray, pd_hat: np.ndarray,
                    approval_min: int = 50, approval_max: int = 100,
                    rank_score: np.ndarray | None = None) -> pd.DataFrame:
    """Approve the X% of applicants with the lowest PD, for X in [min, max].

    Shows the trade-off between volume (approval rate) and risk (default rate of
    the approved book). Values are estimates on historical, held-out data.
    """
    order = np.argsort(_rank_key(pd_hat, rank_score), kind="mergesort")
    y = np.asarray(y_true)[order]
    p = np.maximum.accumulate(np.asarray(pd_hat)[order])   # PD threshold at each cut
    n, total_defaults = len(y), y.sum()
    cum_defaults = np.cumsum(y)
    rows = []
    for rate in range(approval_min, approval_max + 1):
        k = max(1, int(round(n * rate / 100)))
        approved_defaults = cum_defaults[k - 1]
        rows.append({
            "approval_rate_pct": rate,
            "pd_threshold": float(p[k - 1]),
            "n_approved": k,
            "n_rejected": n - k,
            "default_rate_approved": approved_defaults / k,
            "default_rate_rejected": (total_defaults - approved_defaults) / (n - k) if n > k else np.nan,
            "share_defaults_avoided": (total_defaults - approved_defaults) / total_defaults,
            "baseline_default_rate": total_defaults / n,
        })
    return pd.DataFrame(rows)
