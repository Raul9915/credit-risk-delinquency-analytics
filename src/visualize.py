"""Static charts for the README (PNG, light surface, colour-blind-safe palette)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.config import Paths  # noqa: E402
from src.modeling import TARGET_COL, TrainingResult  # noqa: E402

log = logging.getLogger(__name__)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
STATE_ORDER = ["Current", "DPD 1-30", "DPD 31-60", "DPD 61-90", "DPD 91-120", "DPD 120+"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10, "lines.linewidth": 2,
})


def _save(fig: plt.Figure, folder: Path, name: str) -> Path:
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart: %s", path.name)
    return path


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def transition_heatmap(transitions: pd.DataFrame, folder: Path) -> Path:
    t = transitions[transitions["CREDIT_TYPE"] == "All"]
    m = (t.pivot_table(index="FROM_STATE", columns="TO_STATE", values="PROBABILITY", aggfunc="sum")
          .reindex(index=STATE_ORDER, columns=STATE_ORDER + ["Closed"]).fillna(0))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.grid(False)
    ax.imshow(m.to_numpy(), cmap=BLUE_RAMP, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, f"{100 * v:.1f}%" if v >= 0.0005 else "·", ha="center", va="center",
                    fontsize=8.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=0)
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.set_xlabel("Status in month m+1")
    ax.set_ylabel("Status in month m")
    ax.set_title("Monthly delinquency transition matrix", pad=22)
    _subtitle(ax, "Credit Bureau credits, all types. Each row sums to 100%.")
    return _save(fig, folder, "transition_matrix")


def roll_rate_chart(roll: pd.DataFrame, folder: Path) -> Path:
    r = roll[roll["CREDIT_TYPE"] == "All"].set_index("FROM_STATE").reindex(STATE_ORDER[:-1])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(r))
    ax.bar(x, r["pct_roll_forward"] * 100, width=0.55, color=SERIES[0])
    for xi, v in zip(x, r["pct_roll_forward"] * 100):
        ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, r.index)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("% rolling to a worse bucket next month")
    ax.set_title("Roll-forward rate by delinquency bucket", pad=22)
    _subtitle(ax, "Share of credits that move to a worse status the following month.")
    return _save(fig, folder, "roll_rates")


def vintage_chart(vintage: pd.DataFrame, folder: Path) -> Path:
    v = vintage[(vintage["CREDIT_TYPE"] == "All") & (vintage["IS_RELIABLE"] == 1)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for color, (cohort, g) in zip(SERIES, v.groupby("VINTAGE_COHORT")):
        ax.plot(g["MONTHS_ON_BOOK"], g["CUMULATIVE_DEFAULT_RATE"] * 100, color=color, label=cohort)
    ax.set_xlabel("Months on book")
    ax.set_ylabel("Cumulative 90+ DPD rate (%)")
    ax.legend(title="Credit opened", loc="upper left")
    ax.set_title("Vintage curves: cumulative default rate by cohort", pad=22)
    _subtitle(ax, "Kaplan-Meier estimate; points with fewer than 100 credits at risk are hidden.")
    return _save(fig, folder, "vintage_curves")


def risk_band_chart(segments: pd.DataFrame, folder: Path) -> Path:
    b = segments.groupby("RISK_BAND")[["n_clients", "n_defaults"]].sum()
    b["rate"] = 100 * b["n_defaults"] / b["n_clients"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(b))
    ax.bar(x, b["rate"], width=0.55, color=SERIES[0])
    for xi, v in zip(x, b["rate"]):
        ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [f"Band {k}" for k in b.index])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Observed default rate by risk band", pad=22)
    _subtitle(ax, "Held-out test set. Bands are PD quintiles defined on the validation set.")
    return _save(fig, folder, "risk_bands")


def roc_chart(result: TrainingResult, folder: Path) -> Path:
    test = result.splits.test
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
    for color, (name, model) in zip(SERIES, result.models.items()):
        fpr, tpr, _ = roc_curve(test[TARGET_COL], model.predict_pd(test[result.spec.all]))
        ax.plot(fpr, tpr, color=color, label=name.replace("_", " ").capitalize())
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.set_title("ROC curve (test set)", pad=10)
    return _save(fig, folder, "roc_curves")


def calibration_chart(deciles: pd.DataFrame, folder: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    top = max(deciles["avg_pd"].max(), deciles["default_rate"].max()) * 105
    ax.plot([0, top], [0, top], color=INK_2, lw=1, ls="--")
    ax.plot(deciles["avg_pd"] * 100, deciles["default_rate"] * 100, color=SERIES[0],
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlabel("Mean predicted PD (%)")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Calibration by score decile (test set)", pad=10)
    return _save(fig, folder, "calibration")


def cutoff_chart(cutoff: pd.DataFrame, folder: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(cutoff["approval_rate_pct"], cutoff["default_rate_approved"] * 100, color=SERIES[0])
    base = cutoff["baseline_default_rate"].iloc[0] * 100
    ax.axhline(base, color=INK_2, lw=1, ls="--")
    ax.text(cutoff["approval_rate_pct"].min(), base, " approve everyone", va="bottom",
            fontsize=9, color=INK_2)
    ax.set_xlabel("Approval rate (%) - lowest-PD applicants approved first")
    ax.set_ylabel("Default rate of approved book (%)")
    ax.set_title("Cut-off strategy: volume vs. risk", pad=22)
    _subtitle(ax, "Estimated on the held-out test set.")
    return _save(fig, folder, "cutoff_strategy")


def importance_chart(importance: pd.DataFrame, folder: Path, top: int = 15) -> Path:
    imp = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(imp["feature"], imp["auc_drop_mean"], color=SERIES[0], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Top {top} features (permutation importance)", pad=10)
    return _save(fig, folder, "feature_importance")


def build_charts(paths: Paths, result: TrainingResult, tables: dict[str, pd.DataFrame]) -> None:
    folder = paths.images
    transition_heatmap(tables["transition_matrix"], folder)
    roll_rate_chart(tables["roll_rates"], folder)
    vintage_chart(tables["vintage_curves"], folder)
    risk_band_chart(tables["risk_segments"], folder)
    roc_chart(result, folder)
    calibration_chart(tables["model_deciles"], folder)
    cutoff_chart(tables["cutoff_strategy"], folder)
    importance_chart(tables["feature_importance"], folder)
