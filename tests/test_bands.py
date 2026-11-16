"""Stage 2: ARD and L2A items, the shared band() reader, and remote asset hrefs."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import rasterio

import workflow
from eodh_workflow import read_staged_item
from golden_data import item_json_path, on_grid, polygon_mask, read_output, run_ndvi

ROOT = Path(workflow.__file__).resolve().parent.parent

# Bands with these common names, and the brightness rule from the plan's Decisions.
EXPECTED_BANDS = {
    "l2a": {"red": ("red.tif", 1), "nir": ("nir.tif", 1), "scale": 0.0001, "offset": -0.1},
    "ard": {"red": ("cog.tif", 3), "nir": ("cog.tif", 7), "scale": 1, "offset": 0},
}


def _rewrite_hrefs(staged: Path, href_for) -> None:
    """Replace each asset href in the staged item with href_for(asset key, local file path)."""
    path = item_json_path(staged)
    item = json.loads(path.read_text())
    for key, asset in item["assets"].items():
        asset["href"] = href_for(key, (path.parent / asset["href"]).resolve())
    path.write_text(json.dumps(item))


@pytest.mark.parametrize("name", ["l2a", "ard"])
def test_stage2_workflow_picks_red_and_nir_by_common_name(name, request, tmp_path):
    staged = request.getfixturevalue(f"{name}_staged")
    rules = EXPECTED_BANDS[name]
    item_dir = item_json_path(staged).parent
    raw = {}
    for band in ("red", "nir"):
        filename, index = rules[band]
        with rasterio.open(item_dir / filename) as src:
            raw[band] = src.read(index).astype("float64") * rules["scale"] + rules["offset"]
            staged_transform = src.transform
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (raw["nir"] - raw["red"]) / (raw["nir"] + raw["red"])

    data, transform, crs = read_output(run_ndvi(staged, tmp_path / "out"))
    expected = on_grid(expected, staged_transform, data.shape, transform)
    inside = polygon_mask(data.shape, transform, crs)
    assert inside.sum() > 40000
    np.testing.assert_allclose(data[inside], expected[inside], rtol=1e-5, atol=1e-6)


def test_stage2_band_gives_local_path_for_relative_href_and_url_for_absolute_href(ard_staged):
    from common import band

    item_path = item_json_path(ard_staged)
    local = band(read_staged_item(ard_staged), "nir")
    assert Path(str(local[0])) == (item_path.parent / "cog.tif").resolve()
    assert local[1] == 7

    for href in ("https://example.com/a/cog.tif", "http://example.com/a/cog.tif", "file:///data/a/cog.tif"):
        _rewrite_hrefs(ard_staged, lambda key, path: href)
        remote = band(read_staged_item(ard_staged), "red")
        assert str(remote[0]) == href
        assert remote[1] == 3


def test_stage2_products_read_bands_only_through_common_band():
    products = list((ROOT / "src" / "products").glob("*.py"))
    assert products
    for path in products:
        source = path.read_text()
        assert "asset_path(" not in source, path
        assert not re.search(r"\b(stac_item|item)\.band\(", source), path


@pytest.mark.parametrize("name", ["l2a", "ard"])
def test_stage2_file_url_hrefs_give_the_same_output_as_relative_hrefs(name, request, tmp_path):
    staged = request.getfixturevalue(f"{name}_staged")
    relative, relative_transform, _ = read_output(run_ndvi(staged, tmp_path / "relative"))

    _rewrite_hrefs(staged, lambda key, path: path.as_uri())
    absolute, absolute_transform, _ = read_output(run_ndvi(staged, tmp_path / "absolute"))

    assert absolute_transform == relative_transform
    np.testing.assert_array_equal(absolute, relative)


def test_stage2_dockerfile_sets_gdal_options_for_remote_cogs():
    dockerfile = (ROOT / "Dockerfile").read_text()
    for name, value in [
        ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"),
        ("GDAL_HTTP_MAX_RETRY", r"\d+"),
        ("GDAL_HTTP_RETRY_DELAY", r"[\d.]+"),
        ("GDAL_HTTP_MULTIRANGE", "YES"),
        ("VSI_CACHE", "TRUE"),
    ]:
        assert re.search(rf"\b{name}=\"?{value}\"?(\s|\\|$)", dockerfile), name


def test_stage2_failed_remote_read_names_the_href(l2a_staged, tmp_path):
    # Nothing listens on port 9 of the loopback address, so the read fails at once without a network.
    _rewrite_hrefs(l2a_staged, lambda key, path: f"http://127.0.0.1:9/{key}.tif")
    with pytest.raises(Exception, match=re.escape("http://127.0.0.1:9/")):
        run_ndvi(l2a_staged, tmp_path / "out")
