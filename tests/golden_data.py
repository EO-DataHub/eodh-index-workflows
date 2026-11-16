"""Shared helpers for the NDVI tests: the staged golden fixtures, running the workflow, and grid maths."""

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import transform_geom

import workflow
from eodh_workflow import WorkflowInputs, read_staged_item

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
AOI = json.loads((GOLDEN / "aoi.geojson").read_text())
AOI_JSON = json.dumps(AOI)
ITEM_IDS = {name: json.loads((GOLDEN / name / "ndvi.json").read_text())["item_id"] for name in ("l2a", "ard")}


def copy_staged(name: str, dest: Path) -> Path:
    """A writable copy of the staged directory of golden fixture `name` (`l2a` or `ard`)."""
    return Path(shutil.copytree(GOLDEN / name / "staged", dest))


def item_json_path(staged: Path) -> Path:
    [path] = [p for p in staged.glob("*/*.json")]
    return path


def run_ndvi(staged: Path, output_dir: Path, **params):
    """Call workflow.run() as main.py does, with the eodh-ndvi workflow and the golden AOI by default."""
    params = {"workflow": "eodh-ndvi", "aoi": AOI_JSON, **params}
    params = {key: value for key, value in params.items() if value is not None}
    output_dir.mkdir(parents=True, exist_ok=True)
    return workflow.run(WorkflowInputs(stac_item=read_staged_item(staged), params=params), output_dir)


def read_output(spec):
    """(array, transform, crs) of the single item's `data` asset."""
    [item] = spec.items
    with rasterio.open(item.assets["data"].path) as src:
        return src.read(1), src.transform, src.crs


def read_golden(name: str):
    meta = json.loads((GOLDEN / name / "ndvi.json").read_text())
    return np.load(GOLDEN / name / "ndvi.npy"), Affine(*meta["transform"]), meta["crs"]


def polygon_mask(shape, transform, crs, aoi=AOI, all_touched=False) -> np.ndarray:
    """True for pixels whose centre is in the AOI (or, with `all_touched`, that touch it)."""
    geometry = transform_geom("EPSG:4326", crs, aoi)
    return rasterize([geometry], out_shape=shape, transform=transform, all_touched=all_touched).astype(bool)


def on_grid(array, transform, shape, target_transform) -> np.ndarray:
    """`array` cut (or NaN-padded) to `shape` on `target_transform`. The two grids must differ by whole pixels."""
    assert transform.a == target_transform.a and transform.e == target_transform.e, "different pixel sizes"
    col = (target_transform.c - transform.c) / transform.a
    row = (target_transform.f - transform.f) / transform.e
    assert abs(col - round(col)) < 1e-6 and abs(row - round(row)) < 1e-6, "grids are not whole pixels apart"
    col, row = round(col), round(row)
    out = np.full(shape, np.nan, dtype="float64")
    src_r0, src_c0 = max(row, 0), max(col, 0)
    dst_r0, dst_c0 = max(-row, 0), max(-col, 0)
    height = min(array.shape[0] - src_r0, shape[0] - dst_r0)
    width = min(array.shape[1] - src_c0, shape[1] - dst_c0)
    if height > 0 and width > 0:
        source = array[src_r0 : src_r0 + height, src_c0 : src_c0 + width]
        out[dst_r0 : dst_r0 + height, dst_c0 : dst_c0 + width] = source
    return out


def box(xmin, ymin, xmax, ymax) -> dict:
    return {"type": "Polygon", "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}
