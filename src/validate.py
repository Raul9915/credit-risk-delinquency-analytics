"""Data-quality gates. The pipeline stops if any check fails."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb

from src.config import Paths
from src.transform import register_layer

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    sql: str            # must return a single integer: the number of violations


CHECKS = [
    Check("silver.application: SK_ID_CURR is unique",
          "SELECT COUNT(*) - COUNT(DISTINCT SK_ID_CURR) FROM silver_application"),
    Check("silver.application: TARGET in {0, 1}",
          "SELECT COUNT(*) FROM silver_application WHERE TARGET NOT IN (0, 1) OR TARGET IS NULL"),
    Check("silver.bureau: SK_ID_BUREAU is unique",
          "SELECT COUNT(*) - COUNT(DISTINCT SK_ID_BUREAU) FROM silver_bureau"),
    Check("silver.bureau_balance: one status per credit and month",
          """SELECT COUNT(*) FROM (SELECT SK_ID_BUREAU, MONTHS_BALANCE FROM silver_bureau_balance
             GROUP BY ALL HAVING COUNT(*) > 1)"""),
    Check("gold.client_features: one row per applicant",
          """SELECT ABS((SELECT COUNT(*) FROM gold_client_features)
                      - (SELECT COUNT(*) FROM silver_application))
             + (SELECT COUNT(*) - COUNT(DISTINCT SK_ID_CURR) FROM gold_client_features)"""),
    Check("gold.fct_transitions: probabilities sum to 1 per from-state",
          """SELECT COUNT(*) FROM (SELECT CREDIT_TYPE, FROM_STATE, SUM(PROBABILITY) p
             FROM gold_fct_transitions GROUP BY ALL HAVING ABS(p - 1) > 1e-9)"""),
    Check("gold.vintage_curves: cumulative default rate within [0, 1]",
          """SELECT COUNT(*) FROM gold_vintage_curves
             WHERE CUMULATIVE_DEFAULT_RATE < 0 OR CUMULATIVE_DEFAULT_RATE > 1"""),
    Check("gold.vintage_curves: cumulative default rate is non-decreasing",
          """SELECT COUNT(*) FROM (
                SELECT CUMULATIVE_DEFAULT_RATE - LAG(CUMULATIVE_DEFAULT_RATE) OVER (
                    PARTITION BY VINTAGE_COHORT, CREDIT_TYPE ORDER BY MONTHS_ON_BOOK) AS d
                FROM gold_vintage_curves) WHERE d < -1e-12"""),
]


class DataQualityError(RuntimeError):
    pass


def run_checks(paths: Paths) -> list[tuple[str, int]]:
    con = duckdb.connect()
    try:
        for layer in ("silver", "gold"):
            register_layer(con, paths, layer)
        results = [(c.name, int(con.execute(c.sql).fetchone()[0])) for c in CHECKS]
    finally:
        con.close()
    failed = [(name, n) for name, n in results if n != 0]
    for name, n in results:
        log.info("[%s] %s", "PASS" if n == 0 else f"FAIL ({n})", name)
    if failed:
        raise DataQualityError(f"{len(failed)} data-quality check(s) failed: {failed}")
    return results
