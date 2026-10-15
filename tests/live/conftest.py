"""Live tests are opt-in: they load the real models and spend Groq tokens, so they never run in a normal `pytest` pass."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    wanted = "live" in (config.getoption("-m") or "") or os.getenv("RUN_LIVE") == "1"
    if wanted:
        return
    skip = pytest.mark.skip(reason="live end-to-end tests: run with `pytest -m live tests/live` (or RUN_LIVE=1)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
