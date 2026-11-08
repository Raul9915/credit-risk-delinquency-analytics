"""Test fixtures.

The tests never use Kaggle data (it cannot be redistributed and is ~2.5 GB).
Instead, ``write_mini_raw`` writes a few hundred rows of *made-up* records with
the same column names as the Kaggle files, only to exercise the code paths.
No result, chart or report in this repository comes from these records.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Paths, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_paths(base: Path) -> Paths:
    paths = Paths(raw=base / "raw", root=base, reports=base / "reports",
                  images=base / "images", models=base / "models")
    paths.ensure()
    return paths


def write_mini_raw(raw: Path, n_clients: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n_clients)
    ext2 = rng.uniform(0, 1, n_clients)
    target = (rng.uniform(0, 1, n_clients) < 0.05 + 0.3 * (1 - ext2) ** 3).astype(int)
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n_clients),
        "CODE_GENDER": rng.choice(["M", "F", "XNA"], n_clients, p=[0.45, 0.54, 0.01]),
        "AMT_INCOME_TOTAL": rng.uniform(5e4, 4e5, n_clients).round(),
        "AMT_CREDIT": rng.uniform(1e5, 1e6, n_clients).round(),
        "AMT_ANNUITY": rng.uniform(5e3, 5e4, n_clients).round(),
        "AMT_GOODS_PRICE": rng.uniform(1e5, 1e6, n_clients).round(),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], n_clients),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education"],
                                          n_clients),
        "DAYS_BIRTH": -rng.integers(21 * 365, 68 * 365, n_clients),
        "DAYS_EMPLOYED": np.where(rng.uniform(size=n_clients) < 0.1, 365243,
                                  -rng.integers(0, 15 * 365, n_clients)),
        "EXT_SOURCE_1": np.where(rng.uniform(size=n_clients) < 0.5, np.nan,
                                 rng.uniform(0, 1, n_clients)),
        "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": rng.uniform(0, 1, n_clients),
    })
    app.to_csv(raw / "application_train.csv", index=False)

    bureau_rows, bb_rows = [], []
    sk_bureau = 5000000
    for cid in ids:
        for _ in range(rng.integers(0, 4)):
            sk_bureau += 1
            days_credit = -int(rng.integers(30, 2500))
            bureau_rows.append({
                "SK_ID_CURR": cid, "SK_ID_BUREAU": sk_bureau,
                "CREDIT_ACTIVE": rng.choice(["Active", "Closed", "Bad debt"], p=[0.4, 0.58, 0.02]),
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": days_credit, "CREDIT_DAY_OVERDUE": int(rng.integers(0, 5)),
                "DAYS_CREDIT_ENDDATE": days_credit + 720, "DAYS_ENDDATE_FACT": np.nan,
                "AMT_CREDIT_MAX_OVERDUE": np.nan, "CNT_CREDIT_PROLONG": 0,
                "AMT_CREDIT_SUM": float(rng.uniform(1e4, 5e5)),
                "AMT_CREDIT_SUM_DEBT": float(rng.uniform(0, 1e5)),
                "AMT_CREDIT_SUM_LIMIT": 0.0, "AMT_CREDIT_SUM_OVERDUE": 0.0,
                "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan", "Another type of loan"]),
                "DAYS_CREDIT_UPDATE": -10, "AMT_ANNUITY": np.nan,
            })
            open_month = round(days_credit / 30.4375)
            state = 0
            for m in range(open_month, 1):
                r = rng.uniform()
                if state == 0:
                    state = 1 if r < 0.05 else 0
                elif state < 5:
                    state = state + 1 if r < 0.3 else (0 if r < 0.8 else state)
                bb_rows.append({"SK_ID_BUREAU": sk_bureau, "MONTHS_BALANCE": m,
                                "STATUS": "X" if r > 0.98 else str(state)})
    pd.DataFrame(bureau_rows).to_csv(raw / "bureau.csv", index=False)
    pd.DataFrame(bb_rows).to_csv(raw / "bureau_balance.csv", index=False)

    prev = pd.DataFrame({
        "SK_ID_PREV": np.arange(len(ids)) + 2000000, "SK_ID_CURR": ids,
        "NAME_CONTRACT_TYPE": "Consumer loans",
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], len(ids)),
        "AMT_APPLICATION": rng.uniform(1e4, 2e5, len(ids)), "AMT_CREDIT": rng.uniform(1e4, 2e5, len(ids)),
        "AMT_ANNUITY": rng.uniform(1e3, 1e4, len(ids)), "CNT_PAYMENT": 12,
        "DAYS_DECISION": -rng.integers(30, 2000, len(ids)),
        "DAYS_FIRST_DUE": 365243, "DAYS_LAST_DUE": -100, "DAYS_TERMINATION": -50,
    })
    prev.to_csv(raw / "previous_application.csv", index=False)
    inst = pd.DataFrame({
        "SK_ID_PREV": prev["SK_ID_PREV"], "SK_ID_CURR": ids, "NUM_INSTALMENT_VERSION": 1,
        "NUM_INSTALMENT_NUMBER": 1, "DAYS_INSTALMENT": -100.0,
        "DAYS_ENTRY_PAYMENT": -100.0 + rng.integers(-5, 20, len(ids)),
        "AMT_INSTALMENT": 1000.0, "AMT_PAYMENT": rng.choice([1000.0, 800.0], len(ids)),
    })
    inst.to_csv(raw / "installments_payments.csv", index=False)
    pd.DataFrame({"SK_ID_PREV": prev["SK_ID_PREV"], "SK_ID_CURR": ids, "MONTHS_BALANCE": -1,
                  "CNT_INSTALMENT": 12, "CNT_INSTALMENT_FUTURE": 6, "NAME_CONTRACT_STATUS": "Active",
                  "SK_DPD": rng.integers(0, 3, len(ids)), "SK_DPD_DEF": 0}
                 ).to_csv(raw / "POS_CASH_balance.csv", index=False)
    pd.DataFrame({"SK_ID_PREV": prev["SK_ID_PREV"], "SK_ID_CURR": ids, "MONTHS_BALANCE": -1,
                  "AMT_BALANCE": rng.uniform(0, 5e4, len(ids)), "AMT_CREDIT_LIMIT_ACTUAL": 5e4,
                  "SK_DPD": 0, "SK_DPD_DEF": 0}).to_csv(raw / "credit_card_balance.csv", index=False)


@pytest.fixture(scope="session")
def settings() -> dict:
    return load_settings()


@pytest.fixture(scope="session")
def built_paths(tmp_path_factory, settings) -> Paths:
    """Mini dataset processed through bronze, silver and gold (shared by tests)."""
    from src.ingest import ingest_to_bronze
    from src.transform import build_silver_and_gold

    paths = make_paths(tmp_path_factory.mktemp("mini"))
    write_mini_raw(paths.raw)
    ingest_to_bronze(paths, settings["sources"])
    build_silver_and_gold(paths, dict(settings["delinquency"]))
    return paths
