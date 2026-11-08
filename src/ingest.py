"""Bronze layer: raw Kaggle CSVs -> Parquet, unchanged content plus load metadata."""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from src.config import Paths

log = logging.getLogger(__name__)


class MissingSourceFileError(FileNotFoundError):
    """Raised when an expected Kaggle CSV is not in the raw folder."""


def check_raw_files(raw_dir: Path, sources: dict[str, str]) -> None:
    missing = [name for name in sources if not (raw_dir / name).exists()]
    if missing:
        raise MissingSourceFileError(
            f"Missing files in {raw_dir}: {', '.join(missing)}. "
            "See data/README.md for download instructions."
        )


def ingest_to_bronze(paths: Paths, sources: dict[str, str]) -> dict[str, int]:
    """Convert every source CSV into ``bronze/<table>.parquet``.

    Returns the row count per table so the pipeline can log and validate it.
    """
    check_raw_files(paths.raw, sources)
    bronze = paths.layer("bronze")
    counts: dict[str, int] = {}
    con = duckdb.connect()
    try:
        for file_name, table in sources.items():
            src = (paths.raw / file_name).as_posix()
            dst = (bronze / f"{table}.parquet").as_posix()
            con.execute(
                f"""
                COPY (
                    SELECT *,
                           '{file_name}'      AS _source_file,
                           current_timestamp  AS _loaded_at
                    FROM read_csv_auto('{src}', header = true, sample_size = -1)
                ) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            counts[table] = con.execute(f"SELECT count(*) FROM '{dst}'").fetchone()[0]
            log.info("bronze.%s: %s rows", table, f"{counts[table]:,}")
    finally:
        con.close()
    return counts
