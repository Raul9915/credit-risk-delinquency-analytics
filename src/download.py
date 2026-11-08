"""Download the Home Credit Default Risk data straight from Kaggle.

Requirements (one-time setup, see data/README.md):
  1. A Kaggle account that has accepted the competition rules.
  2. An API token: kaggle.json in ~/.kaggle/ (Linux/Mac) or
     C:\\Users\\<user>\\.kaggle\\ (Windows), or the KAGGLE_USERNAME / KAGGLE_KEY
     environment variables. The token is never stored in this repository.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


class KaggleSetupError(RuntimeError):
    """Raised when the Kaggle client is missing, unauthenticated or unauthorised."""


def _authenticated_api():
    try:
        # Importing the package already tries to authenticate, so a missing token
        # surfaces here as a generic error (OSError / SystemExit in some versions).
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise KaggleSetupError("Install the Kaggle client: pip install kaggle") from exc
    except (Exception, SystemExit) as exc:
        raise KaggleSetupError(
            "Kaggle authentication failed. Create a token at kaggle.com -> Settings -> API "
            "-> 'Create New Token' and place kaggle.json in your .kaggle folder."
        ) from exc
    return api


def download_competition(competition: str, raw_dir: Path, required: list[str],
                         force: bool = False) -> list[Path]:
    """Download and unzip the competition files into ``raw_dir``.

    Skips the download when every required CSV is already present (unless ``force``).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in required if not (raw_dir / name).exists()]
    if not missing and not force:
        log.info("All %d source files already in %s - skipping download.", len(required), raw_dir)
        return [raw_dir / name for name in required]

    api = _authenticated_api()
    log.info("Downloading '%s' from Kaggle (~700 MB compressed)...", competition)
    try:
        api.competition_download_files(competition, path=str(raw_dir), force=force, quiet=False)
    except Exception as exc:
        raise KaggleSetupError(
            f"Download failed. Make sure you accepted the rules at "
            f"https://www.kaggle.com/competitions/{competition}/rules "
            f"with the same account as your API token. Original error: {exc}"
        ) from exc

    for archive in raw_dir.glob("*.zip"):
        log.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        archive.unlink()

    still_missing = [name for name in required if not (raw_dir / name).exists()]
    if still_missing:
        raise KaggleSetupError(f"Files not found after extraction: {still_missing}")
    return [raw_dir / name for name in required]
