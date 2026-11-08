"""End-to-end pipeline.

    python -m src.pipeline                 # download -> bronze -> silver/gold -> checks -> model -> reports
    python -m src.pipeline --skip-download # use the CSVs already in data/raw
    python -m src.pipeline --until gold    # stop after the data layers
"""
from __future__ import annotations

import argparse
import logging
import time

import joblib

from src.config import get_paths, load_settings
from src.download import download_competition
from src.ingest import ingest_to_bronze
from src.transform import build_silver_and_gold
from src.validate import run_checks

STEPS = ("download", "bronze", "gold", "model")
log = logging.getLogger("pipeline")


def sql_params(settings: dict) -> dict:
    return dict(settings["delinquency"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true",
                        help="do not call the Kaggle API; use files in data/raw")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--until", choices=STEPS, default="model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    settings = load_settings()
    paths = get_paths(settings)
    paths.ensure()
    start = time.time()

    if not args.skip_download:
        download_competition(settings["kaggle"]["competition"], paths.raw,
                             list(settings["sources"]), force=args.force_download)
    if args.until == "download":
        return

    log.info("== Bronze ==")
    ingest_to_bronze(paths, settings["sources"])
    if args.until == "bronze":
        return

    log.info("== Silver & Gold ==")
    build_silver_and_gold(paths, sql_params(settings))
    log.info("== Data-quality checks ==")
    run_checks(paths)
    if args.until == "gold":
        return

    # Heavy imports only when modelling.
    import duckdb

    from src.modeling import train_models
    from src.reporting import build_reports
    from src.visualize import build_charts

    log.info("== PD model ==")
    features = duckdb.sql(
        f"SELECT * FROM '{(paths.layer('gold') / 'client_features.parquet').as_posix()}'").df()
    result = train_models(features, settings["modeling"])
    joblib.dump(result.models[result.selected], paths.models / "pd_model.joblib")

    log.info("== Reports & charts ==")
    tables = build_reports(paths, result, settings)
    build_charts(paths, result, tables)
    log.info("Done in %.1f min. See reports/results_summary.md", (time.time() - start) / 60)


if __name__ == "__main__":
    main()
