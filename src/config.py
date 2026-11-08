"""Configuration loading. Keeps paths and parameters out of the code."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = REPO_ROOT / "config" / "settings.yaml"
SQL_DIR = REPO_ROOT / "sql"
LAYERS = ("bronze", "silver", "gold")


@dataclass(frozen=True)
class Paths:
    """Resolved, absolute project paths."""

    raw: Path
    root: Path
    reports: Path
    images: Path
    models: Path

    def layer(self, name: str) -> Path:
        if name not in LAYERS:
            raise ValueError(f"Unknown layer '{name}'. Expected one of {LAYERS}.")
        return self.root / name

    @property
    def data_studio(self) -> Path:
        return self.reports / "data_studio" / "data"

    def ensure(self) -> None:
        for directory in (self.raw, *(self.layer(l) for l in LAYERS), self.reports,
                          self.images, self.models, self.data_studio):
            directory.mkdir(parents=True, exist_ok=True)


def load_settings(path: Path = SETTINGS_FILE) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_paths(settings: dict[str, Any] | None = None, base: Path = REPO_ROOT) -> Paths:
    """Return absolute paths, resolved against ``base`` (the repo root by default)."""
    settings = settings or load_settings()
    return Paths(**{key: base / value for key, value in settings["paths"].items()})
