"""Delinquency rules: status mapping, transition matrix and vintage curves."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.config import SQL_DIR
from src.transform import render_sql


def _query(sql_file: str, views: dict[str, pd.DataFrame], params: dict | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    for name, df in views.items():
        con.register(name, df)
    sql = render_sql((SQL_DIR / sql_file).read_text(encoding="utf-8"), params or {})
    return con.execute(sql).df()


def test_status_mapping_follows_kaggle_dictionary():
    raw = pd.DataFrame({"SK_ID_BUREAU": [1] * 8, "MONTHS_BALANCE": range(-7, 1),
                        "STATUS": ["0", "1", "2", "3", "4", "5", "C", "X"]})
    out = _query("silver/03_bureau_balance.sql", {"bronze_bureau_balance": raw})
    assert out["STATUS_ORDER"].tolist()[:7] == [0, 1, 2, 3, 4, 5, -1]
    assert pd.isna(out["STATUS_ORDER"].iloc[7])           # X = unknown
    assert out["STATUS_LABEL"].tolist() == ["Current", "DPD 1-30", "DPD 31-60", "DPD 61-90",
                                            "DPD 91-120", "DPD 120+", "Closed", "Unknown"]


def _monthly(statuses: dict[int, list[int]]) -> pd.DataFrame:
    labels = {-1: "Closed", 0: "Current", 1: "DPD 1-30", 2: "DPD 31-60", 3: "DPD 61-90",
              4: "DPD 91-120", 5: "DPD 120+"}
    rows = []
    for credit, seq in statuses.items():
        for k, s in enumerate(seq):
            rows.append({"SK_ID_BUREAU": credit, "SK_ID_CURR": 1, "CREDIT_TYPE_GROUP": "Car loan",
                         "MONTHS_BALANCE": k - len(seq), "STATUS_ORDER": s,
                         "STATUS_LABEL": labels[s], "MONTHS_ON_BOOK": k,
                         "VINTAGE_COHORT": "1-2y before app"})
    return pd.DataFrame(rows)


def test_transition_probabilities():
    # Credit 1: Current -> Current -> DPD 1-30 -> Current ; Credit 2: Current -> DPD 1-30 -> DPD 31-60
    monthly = _monthly({1: [0, 0, 1, 0], 2: [0, 1, 2]})
    out = _query("gold/02_fct_transitions.sql", {"gold_fct_bureau_monthly_status": monthly})
    allt = out[out["CREDIT_TYPE"] == "All"].set_index(["FROM_STATE", "TO_STATE"])["PROBABILITY"]
    # From Current: 3 transitions -> 1 stay, 2 to DPD 1-30
    assert allt[("Current", "Current")] == pytest.approx(1 / 3)
    assert allt[("Current", "DPD 1-30")] == pytest.approx(2 / 3)
    # From DPD 1-30: 1 cure, 1 roll forward
    assert allt[("DPD 1-30", "Current")] == pytest.approx(0.5)
    assert allt[("DPD 1-30", "DPD 31-60")] == pytest.approx(0.5)
    # 'All' and the credit-type level must agree when there is a single type
    car = out[out["CREDIT_TYPE"] == "Car loan"]["N_TRANSITIONS"].sum()
    assert car == out[out["CREDIT_TYPE"] == "All"]["N_TRANSITIONS"].sum() == 5


def test_vintage_curve_is_kaplan_meier():
    # 4 credits observed 0..3. Credit 1 defaults (status 4) at mob 2; credit 2 is
    # censored after mob 1; credits 3 and 4 survive.
    monthly = _monthly({1: [0, 1, 4, 5], 2: [0, 0], 3: [0, 0, 0, 0], 4: [0, 0, 0, 0]})
    params = {"default_status_min": 4, "vintage_max_months_on_book": 3}
    perf = _query("gold/03_dim_credit_performance.sql",
                  {"gold_fct_bureau_monthly_status": monthly}, params)
    assert perf.set_index("SK_ID_BUREAU").loc[1, "FIRST_DEFAULT_MOB"] == 2
    curve = _query("gold/04_vintage_curves.sql", {"gold_dim_credit_performance": perf}, params)
    c = curve[curve["CREDIT_TYPE"] == "All"].set_index("MONTHS_ON_BOOK")
    # At mob 2 only 3 credits are at risk (credit 2 left after mob 1): hazard = 1/3
    assert c.loc[2, "N_AT_RISK"] == 3
    assert c.loc[2, "CUMULATIVE_DEFAULT_RATE"] == pytest.approx(1 / 3)
    assert c.loc[1, "CUMULATIVE_DEFAULT_RATE"] == 0
    assert c["CUMULATIVE_DEFAULT_RATE"].is_monotonic_increasing
