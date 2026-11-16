import json
from pathlib import Path

import pytest

from eodh_workflow.testing import stage_item
from golden_data import copy_staged

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def input_item() -> dict:
    """One real item from the input collection. Its assets are replaced by synthetic rasters when staged.
    tests/helpers/ uses it."""
    return json.loads((FIXTURES / "input_item.json").read_text())


@pytest.fixture
def staged(tmp_path, input_item) -> Path:
    """A directory laid out the way ADES stage-in writes it, for `input_item`, with synthetic rasters."""
    return stage_item(input_item, tmp_path / "staged")


@pytest.fixture
def l2a_staged(tmp_path) -> Path:
    """A writable copy of the staged golden sentinel-2-c1-l2a item (real data over the Oxford AOI)."""
    return copy_staged("l2a", tmp_path / "l2a_staged")


@pytest.fixture
def ard_staged(tmp_path) -> Path:
    """A writable copy of the staged golden sentinel2_ard item (real data over the Oxford AOI)."""
    return copy_staged("ard", tmp_path / "ard_staged")
