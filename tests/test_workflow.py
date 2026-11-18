"""Stage 1: the dispatcher in src/workflow.py, the AOI, and NDVI on the staged golden L2A item."""

import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform_bounds

import workflow
from eodh_workflow import WorkflowInputs
from golden_data import AOI_JSON, ITEM_IDS, box, item_json_path, polygon_mask, read_output, run_ndvi

SRC = Path(workflow.__file__).parent
L2A_ID = ITEM_IDS["l2a"]


@pytest.mark.parametrize("name", [None, "eodh-unknown"])
def test_stage1_missing_or_unknown_workflow_fails_listing_known_names(l2a_staged, tmp_path, name):
    with pytest.raises(Exception, match="eodh-ndvi") as error:
        run_ndvi(l2a_staged, tmp_path / "out", workflow=name)
    if name:
        assert name in str(error.value)


def test_stage1_products_and_shared_code_live_in_their_folders():
    products = [p for p in (SRC / "products").glob("*.py") if p.name != "__init__.py"]
    [ndvi] = [p for p in products if "ndvi" in p.stem]
    module = importlib.import_module(f"products.{ndvi.stem}")
    assert module.NAME == "eodh-ndvi"
    assert callable(module.run)
    assert [p for p in (SRC / "common").glob("*.py") if p.name != "__init__.py"] or (
        SRC / "common" / "__init__.py"
    ).read_text().strip()


NEW_PRODUCT = "eodh-test-only-product"


@pytest.fixture
def new_product_module():
    """A product module dropped into src/products/ for one test, then removed."""
    path = SRC / "products" / "zz_test_only_product.py"
    path.write_text(f"NAME = {NEW_PRODUCT!r}\n\n\ndef run(inputs, output_dir):\n    return ('ran', inputs.params)\n")
    importlib.invalidate_caches()
    importlib.reload(workflow)
    yield
    path.unlink()
    sys.modules.pop("products.zz_test_only_product", None)
    importlib.invalidate_caches()
    importlib.reload(workflow)


def test_stage1_new_product_module_is_found_without_changing_dispatcher(new_product_module, tmp_path):
    params = {"workflow": NEW_PRODUCT, "aoi": AOI_JSON}
    assert workflow.run(WorkflowInputs(params=params), tmp_path) == ("ran", params)

    with pytest.raises(Exception) as error:
        workflow.run(WorkflowInputs(params={"workflow": "eodh-unknown", "aoi": AOI_JSON}), tmp_path)
    assert "eodh-ndvi" in str(error.value) and NEW_PRODUCT in str(error.value)


def _feature(geometry):
    return {"type": "Feature", "properties": {}, "geometry": geometry}


AOI_BOX = json.loads(AOI_JSON)
AOI_MULTI = {"type": "MultiPolygon", "coordinates": [AOI_BOX["coordinates"]]}


@pytest.mark.parametrize(
    "aoi",
    [AOI_BOX, AOI_MULTI, _feature(AOI_BOX), _feature(AOI_MULTI)],
    ids=["polygon", "multipolygon", "feature-polygon", "feature-multipolygon"],
)
def test_stage1_aoi_accepts_polygon_multipolygon_or_feature_string(l2a_staged, tmp_path, aoi):
    data, _, _ = read_output(run_ndvi(l2a_staged, tmp_path / "out", aoi=json.dumps(aoi)))
    assert np.isfinite(data).sum() > 40000


@pytest.mark.parametrize(
    "aoi, received",
    [
        (None, "aoi"),
        (json.dumps({"type": "Point", "coordinates": [-1.33, 51.74]}), "Point"),
        (json.dumps({"type": "FeatureCollection", "features": [_feature(AOI_BOX)]}), "FeatureCollection"),
        (json.dumps(_feature({"type": "LineString", "coordinates": [[-1.34, 51.74], [-1.32, 51.75]]})), "LineString"),
        ("not a geojson", "not a geojson"),
    ],
    ids=["missing", "point", "feature-collection", "feature-linestring", "not-json"],
)
def test_stage1_aoi_rejects_anything_else_saying_what_was_received(l2a_staged, tmp_path, aoi, received):
    with pytest.raises(Exception, match=re.escape(received)):
        run_ndvi(l2a_staged, tmp_path / "out", aoi=aoi)


def test_stage1_output_covers_overlap_of_aoi_bbox_and_scene(l2a_staged, tmp_path):
    # The AOI runs past the east edge of the staged scene.
    aoi = box(-1.335, 51.740, -1.29, 51.750)
    [red] = l2a_staged.glob("*/red.tif")
    with rasterio.open(red) as src:
        scene, crs = src.bounds, src.crs
    aoi_bounds = transform_bounds("EPSG:4326", crs, -1.335, 51.740, -1.29, 51.750, densify_pts=21)
    assert aoi_bounds[2] > scene.right
    expected = (
        max(aoi_bounds[0], scene.left),
        max(aoi_bounds[1], scene.bottom),
        min(aoi_bounds[2], scene.right),
        min(aoi_bounds[3], scene.top),
    )

    spec = run_ndvi(l2a_staged, tmp_path / "out", aoi=json.dumps(aoi))
    with rasterio.open(spec.items[0].assets["data"].path) as out:
        assert out.crs == crs
        assert out.bounds.right == pytest.approx(scene.right, abs=0.01)
        assert out.bounds.top == pytest.approx(expected[3], abs=10.5)
        assert out.bounds.left == pytest.approx(expected[0], abs=10.5)
        assert out.bounds.bottom == pytest.approx(expected[1], abs=10.5)


def test_stage1_pixels_outside_aoi_polygon_are_nan(l2a_staged, tmp_path):
    triangle = {
        "type": "Polygon",
        "coordinates": [[[-1.345, 51.735], [-1.315, 51.735], [-1.345, 51.753], [-1.345, 51.735]]],
    }
    data, transform, crs = read_output(run_ndvi(l2a_staged, tmp_path / "out", aoi=json.dumps(triangle)))
    touched = polygon_mask(data.shape, transform, crs, triangle, all_touched=True)
    inside = polygon_mask(data.shape, transform, crs, triangle)
    assert (~touched).sum() > 1000
    assert np.isnan(data[~touched]).all()
    assert np.isfinite(data[inside]).all()


def test_stage1_aoi_that_does_not_overlap_fails(l2a_staged, tmp_path):
    london = box(-0.15, 51.49, -0.10, 51.52)
    with pytest.raises(Exception, match=re.escape(f"AOI does not overlap item {L2A_ID}")):
        run_ndvi(l2a_staged, tmp_path / "out", aoi=json.dumps(london))


def _set_pixels(path, values):
    # The staged rasters are COGs; GDAL refuses to update them in place without this open option.
    with rasterio.open(path, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as dst:
        data = dst.read(1)
        for (row, col), value in values.items():
            data[row, col] = value
        dst.write(data, 1)


def test_stage1_ndvi_scales_l2a_once_and_treats_values_at_or_below_zero_as_nodata(l2a_staged, tmp_path):
    [red] = l2a_staged.glob("*/red.tif")
    nir = red.with_name("nir.tif")
    with rasterio.open(red) as src:
        staged_transform = src.transform
        row, col = src.height // 2, src.width // 2
    # raw (red, nir) -> expected NDVI after scale 0.0001 and offset -0.1
    cases = {
        (row, col): (2000, 4000, 0.5),  # 0.1 and 0.3; unscaled values would give 1/3
        (row, col + 1): (1000, 4000, np.nan),  # red is 0 after scaling
        (row, col + 2): (4000, 1000, np.nan),  # nir is 0 after scaling
        (row, col + 3): (1001, 4000, (0.3 - 0.0001) / (0.3 + 0.0001)),
    }
    _set_pixels(red, {pixel: case[0] for pixel, case in cases.items()})
    _set_pixels(nir, {pixel: case[1] for pixel, case in cases.items()})

    data, transform, _ = read_output(run_ndvi(l2a_staged, tmp_path / "out"))
    for (r, c), (_, _, expected) in cases.items():
        x, y = xy(staged_transform, r, c)
        out_row, out_col = rowcol(transform, x, y)
        actual = data[out_row, out_col]
        if np.isnan(expected):
            assert np.isnan(actual), (r, c)
        else:
            assert actual == pytest.approx(expected, abs=1e-5), (r, c)


def test_stage1_l2a_without_scale_and_offset_fails_naming_item_and_asset(l2a_staged, tmp_path):
    path = item_json_path(l2a_staged)
    item = json.loads(path.read_text())
    del item["assets"]["red"]["raster:bands"]
    path.write_text(json.dumps(item))

    with pytest.raises(Exception) as raised:
        run_ndvi(l2a_staged, tmp_path / "out")
    assert L2A_ID in str(raised.value)
    assert "red" in str(raised.value)


def test_stage1_output_item_and_collection_follow_the_ndvi_product(l2a_staged, tmp_path):
    spec = run_ndvi(l2a_staged, tmp_path / "out")

    [item] = spec.items
    assert item.id == f"{L2A_ID}_ndvi"
    source = item.source_item if isinstance(item.source_item, dict) else item.source_item.item
    assert source["id"] == L2A_ID
    assert list(item.assets) == ["data"]
    assert [band["name"] for band in item.assets["data"].bands] == ["ndvi"]

    collection = spec.collection
    assert collection.id == "eodh-ndvi"
    assert collection.title and collection.description
    [render] = collection.renders.values()
    assert render["assets"] == ["data"]
    assert render["rescale"] == [[-0.2, 1]]
    assert render["colormap_name"] == "ylgn"
