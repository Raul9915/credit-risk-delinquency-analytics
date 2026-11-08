"""Bronze -> Silver -> Gold on a tiny made-up dataset, plus the data-quality gates."""
from __future__ import annotations

import duckdb
import pytest

from src.ingest import MissingSourceFileError, check_raw_files
from src.transform import render_sql
from src.validate import run_checks


def _read(paths, layer, table):
    return duckdb.sql(f"SELECT * FROM '{(paths.layer(layer) / f'{table}.parquet').as_posix()}'").df()


def test_missing_raw_file_is_reported(tmp_path, settings):
    with pytest.raises(MissingSourceFileError, match="application_train.csv"):
        check_raw_files(tmp_path, settings["sources"])


def test_unknown_sql_parameter_fails_loudly():
    with pytest.raises(KeyError):
        render_sql("SELECT ${not_defined}", {})


def test_data_quality_checks_pass(built_paths):
    results = run_checks(built_paths)
    assert all(n == 0 for _, n in results)


def test_silver_cleaning_rules(built_paths):
    app = _read(built_paths, "silver", "application")
    assert (app["DAYS_EMPLOYED"] == 365243).sum() == 0
    assert app["FLAG_DAYS_EMPLOYED_ANOM"].sum() > 0
    assert "XNA" not in set(app["CODE_GENDER"].dropna())
    assert "_source_file" not in app.columns


def test_client_features_one_row_per_applicant(built_paths):
    feats = _read(built_paths, "gold", "client_features")
    app = _read(built_paths, "silver", "application")
    assert len(feats) == len(app)
    assert feats["SK_ID_CURR"].is_unique
