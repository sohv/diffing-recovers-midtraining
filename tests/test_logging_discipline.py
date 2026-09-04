# enforces the logging loop in docs/rules.md: every run directory carries config.json and reaches the logs.
# uv run -m pytest tests/test_logging_discipline.py -v

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
RAW = REPO_ROOT / "results" / "raw"
RESEARCH_LOG = REPO_ROOT / "research_log.md"
RESULT_LOG = REPO_ROOT / "result_log.md"

RUN_DIR = re.compile(r"^(\d{6})_.+")
LOG_HEADING = re.compile(r"^## (\d{6}) ", re.MULTILINE)
# a smoke test only checks that the code runs, so it has nothing to log. see docs/rules.md
SMOKE = re.compile(r"^\d{6}_smoke(_|$)")


def run_dirs() -> list[Path]:
    return sorted(path for path in RAW.glob("*") if path.is_dir() and RUN_DIR.match(path.name))


def logged_run_dirs() -> list[Path]:
    return [path for path in run_dirs() if not SMOKE.match(path.name)]


def logged_dates(log: Path) -> set[str]:
    return set(LOG_HEADING.findall(log.read_text()))


def test_every_run_directory_has_a_config_json():
    runs = run_dirs()
    if not runs:
        pytest.skip("no runs in results/raw yet")
    missing = [path.name for path in runs if not (path / "config.json").exists()]
    assert not missing, f"no config.json, so these runs cannot be reproduced: {missing}"


def test_latest_run_reached_both_logs():
    runs = logged_run_dirs()
    if not runs:
        pytest.skip("no logged runs in results/raw yet")
    latest = runs[-1]
    date = RUN_DIR.match(latest.name).group(1)
    unlogged = [log.name for log in (RESEARCH_LOG, RESULT_LOG) if date not in logged_dates(log)]
    assert not unlogged, f"{latest.name} finished but never reached {unlogged}. see docs/rules.md"


def test_run_directory_names_are_dated():
    stray = [path.name for path in RAW.glob("*") if path.is_dir() and not RUN_DIR.match(path.name)]
    assert not stray, f"run directories are YYMMDD_description_v1, so these can't be ordered by date: {stray}"
