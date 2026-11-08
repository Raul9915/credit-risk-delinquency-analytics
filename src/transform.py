"""Silver and Gold layers: run the SQL files in ``sql/<layer>/`` with DuckDB.

Each SQL file is a single SELECT. The file name (without its numeric prefix)
becomes the table name: ``sql/gold/02_fct_transitions.sql`` -> ``gold_fct_transitions``,
materialised as ``data/gold/fct_transitions.parquet``. Downstream files reference
upstream tables as views over those Parquet files, so memory stays bounded.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import duckdb

from src.config import SQL_DIR, Paths

log = logging.getLogger(__name__)
_PREFIX = re.compile(r"^\d+_")
_PARAM = re.compile(r"\$\{(\w+)\}")


def table_name(sql_file: Path) -> str:
    return _PREFIX.sub("", sql_file.stem)


def render_sql(sql: str, params: dict[str, Any]) -> str:
    """Substitute ``${name}`` placeholders; fail loudly on unknown names."""
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            raise KeyError(f"SQL parameter '{key}' is not defined in settings.yaml")
        return str(params[key])
    return _PARAM.sub(_sub, sql)


def register_layer(con: duckdb.DuckDBPyConnection, paths: Paths, layer: str) -> list[str]:
    """Create ``<layer>_<table>`` views over every Parquet file of a layer."""
    names = []
    for parquet in sorted(paths.layer(layer).glob("*.parquet")):
        view = f"{layer}_{parquet.stem}"
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM '{parquet.as_posix()}'")
        names.append(view)
    return names


def run_layer(con: duckdb.DuckDBPyConnection, paths: Paths, layer: str,
              params: dict[str, Any], sql_dir: Path = SQL_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    out_dir = paths.layer(layer)
    for sql_file in sorted((sql_dir / layer).glob("*.sql")):
        name = table_name(sql_file)
        target = (out_dir / f"{name}.parquet").as_posix()
        query = render_sql(sql_file.read_text(encoding="utf-8"), params)
        con.execute(f"COPY ({query}) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.execute(f"CREATE OR REPLACE VIEW {layer}_{name} AS SELECT * FROM '{target}'")
        counts[name] = con.execute(f"SELECT count(*) FROM {layer}_{name}").fetchone()[0]
        log.info("%s.%s: %s rows", layer, name, f"{counts[name]:,}")
    return counts


def build_silver_and_gold(paths: Paths, params: dict[str, Any],
                          sql_dir: Path = SQL_DIR) -> dict[str, dict[str, int]]:
    # Allow DuckDB to spill to disk: the bureau_balance self-join has ~27M rows.
    spill = paths.root / ".duckdb_tmp"
    con = duckdb.connect(config={"temp_directory": spill.as_posix(),
                                 "preserve_insertion_order": False})
    try:
        register_layer(con, paths, "bronze")
        return {layer: run_layer(con, paths, layer, params, sql_dir)
                for layer in ("silver", "gold")}
    finally:
        con.close()
