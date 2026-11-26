"""Make the EOPro golden fixtures for the index parity tests (NDVI, EVI, SAVI, NBR and NDWI).

Run once, by hand, in an `eodh-workflows` environment at the EOPro commit below. It is not run in CI.

    cd ../eodh-workflows
    git checkout 38cce88 && uv sync
    uv run python ../eodh-index-workflows/scripts/make_golden.py

It needs network access to the Hub STAC API and to the item assets.

For each fixture item it writes, under `tests/fixtures/golden/<name>/`:

- `staged/`: a stage-in directory with real data. `catalog.json` -> `<item id>/<item id>.json`, and
  one small COG per staged asset, cut from the scene's own grid around the AOI. Hrefs are relative.
- `<index>.npy`, for each index in `INDICES`: EOPro's index for the item and the AOI, before EOPro
  reprojects to EPSG:4326.
- `<index>.json`: the grid of `<index>.npy` (CRS, transform, shape) and how it was made.

Every index of one item comes from the same EOPro read of its bands, so all of them share one grid.

It also writes `tests/fixtures/golden/aoi.geojson`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

EOPRO_COMMIT = "38cce88"
STAC_API = "https://eodatahub.org.uk/api/catalogue/stac/catalogs/public/catalogs"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden"

# About 2 km x 2 km west of Oxford. Both items contain it. See the plan's Decisions.
AOI_BOUNDS = (-1.345, 51.735, -1.315, 51.753)
AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [AOI_BOUNDS[0], AOI_BOUNDS[1]],
            [AOI_BOUNDS[2], AOI_BOUNDS[1]],
            [AOI_BOUNDS[2], AOI_BOUNDS[3]],
            [AOI_BOUNDS[0], AOI_BOUNDS[3]],
            [AOI_BOUNDS[0], AOI_BOUNDS[1]],
        ]
    ],
}

FIXTURES = {
    "l2a": {
        "url": f"{STAC_API}/element84/collections/sentinel-2-c1-l2a/items/S2B_T30UXC_20230608T112118_L2A",
        "staged_assets": ["red", "nir", "blue", "swir22", "green"],
    },
    "ard": {
        "url": (
            f"{STAC_API}/ceda-stac-catalogue/collections/sentinel2_ard/items/"
            "neodc.sentinel_ard.data.sentinel_2.2023.06.08."
            "S2B_20230608_latn518lonw0008_T30UXC_ORB037_20230608120350_utm30n_osgb"
        ),
        "staged_assets": ["cog"],
    },
}

# EOPro has no `sentinel-2-c1-l2a` path: `unify_asset_identifiers` refuses any collection it does not
# know. Its nearest path is `sentinel-2-l2a` (Earth Search), the same kind of item: one asset per
# band, with `raster:bands` scale and offset that stackstac applies. The golden L2A array runs that
# path. `resolve_rescale_params` then gives scale 1, offset 0, so the data is scaled once.
EOPRO_L2A_COLLECTION = "sentinel-2-l2a"

# The assets EOPro's CWL path (`legacy/raster/calculator.py`, `calculate`) reads for L2A, plus
# `swir22`, which NBR needs and that path does not read (its NBR fails on L2A). `swir22` is 20 m. stackstac
# puts every asset on the finest asset grid (10 m) with nearest resampling, as `NBR.compute` does, so
# adding it does not change the grid or the other bands.
EOPRO_L2A_ASSETS = ["blue", "green", "red", "rededge1", "nir", "scl", "swir22"]

# The EOPro index classes (`spectral/indices.py`) that get golden arrays, by output file name.
INDICES = ["ndvi", "evi", "savi", "nbr", "ndwi"]

# The band names EOPro gives the ARD `cog` asset (`ds/utils.py`, `prepare_s2_ard_data_array`).
EOPRO_ARD_BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]

# Extra pixels around the AOI window in the staged rasters, so the AOI's bounding box is inside them.
PAD_PIXELS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output folder (default {DEFAULT_OUT})")
    parser.add_argument("--only", choices=sorted(FIXTURES), help="Make one fixture only")
    args = parser.parse_args()

    _setup_eopro_settings()
    _check_eopro_commit()

    import pystac

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aoi.geojson").write_text(json.dumps(AOI, indent=2) + "\n")

    for name, fixture in FIXTURES.items():
        if args.only and name != args.only:
            continue
        print(f"[{name}] reading {fixture['url']}")
        item = pystac.Item.from_file(fixture["url"])
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_staged(item, fixture["staged_assets"], out_dir / "staged")
        raster_arr, meta = _eopro_bands_l2a(item) if name == "l2a" else _eopro_bands_ard(item)
        for index in INDICES:
            golden = _eopro_index(index, raster_arr, meta)
            index_meta = {**meta, "eopro_path": meta["eopro_path"].format(index=index.upper())}
            _write_golden(index, golden, index_meta, item, out_dir)
        print(f"[{name}] wrote {out_dir}")


def _setup_eopro_settings() -> None:
    """EOPro reads its settings when its modules are imported. The indices need none of them."""
    os.environ.setdefault("SENTINEL_HUB__CLIENT_ID", "unused")
    os.environ.setdefault("SENTINEL_HUB__CLIENT_SECRET", "unused")
    os.environ.setdefault("EODH__STAC_API_ENDPOINT", "https://eodatahub.org.uk/api/catalogue/stac")


def _check_eopro_commit() -> None:
    import eodh_workflows

    repo = Path(eodh_workflows.__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot read the eodh-workflows commit from {repo}: {exc}") from exc
    if not head.startswith(EOPRO_COMMIT):
        raise SystemExit(f"eodh-workflows is at {head[:7]}, not {EOPRO_COMMIT}. Run `git checkout {EOPRO_COMMIT}`.")


# --- EOPro golden arrays ---------------------------------------------------------------------------


def _eopro_index(index: str, raster_arr: Any, meta: dict[str, Any]) -> Any:
    """EOPro's `calculate_index` of the index class named `index`, on bands read by EOPro."""
    from eodh_workflows.workflows.spectral import indices

    scale, offset = meta["eopro_rescale"]
    calculator = getattr(indices, index.upper())
    return calculator.calculate_index(raster_arr=raster_arr, rescale_factor=scale, rescale_offset=offset)


def _eopro_bands_l2a(item: Any) -> tuple[Any, dict[str, Any]]:
    """EOPro's CWL path for L2A with `clip=True`, up to (not including) its reprojection to EPSG:4326."""
    from eodh_workflows.workflows.ds.utils import prepare_data_array
    from eodh_workflows.workflows.spectral.utils import resolve_rescale_params

    eopro_item = item.clone()
    eopro_item.collection_id = EOPRO_L2A_COLLECTION
    raster_arr = prepare_data_array(item=eopro_item, bbox=AOI_BOUNDS, assets=EOPRO_L2A_ASSETS)
    scale, offset = resolve_rescale_params(collection_name=eopro_item.collection_id, item_datetime=item.datetime)

    # stackstac's x/y coordinates are pixel top-left corners, so take the transform from stackstac,
    # not from the coordinates.
    transform = raster_arr.attrs.get("transform")
    if transform is None:
        raise SystemExit("stackstac did not record a transform on the array")
    return raster_arr, {
        "eopro_path": "prepare_data_array (stackstac) -> resolve_rescale_params -> {index}.calculate_index",
        "eopro_collection_id": EOPRO_L2A_COLLECTION,
        "eopro_assets": EOPRO_L2A_ASSETS,
        "stackstac_rescale_applied": True,
        "stackstac_resampling": "nearest (default), onto the finest asset grid",
        "eopro_rescale": [scale, offset],
        "bbox_latlon": list(AOI_BOUNDS),
        "transform": list(transform)[:6],
        "transform_source": "stackstac attrs",
    }


def _eopro_bands_ard(item: Any) -> tuple[Any, dict[str, Any]]:
    """EOPro's ARD read with an AOI, without its reprojection to EPSG:4326.

    EOPro reprojects inside `_prepare_s2_ard_data_array_clip`, so the reprojection is switched off
    while it runs. Only the `cog` asset is read: the indices do not use the `cloud` band. Every band
    is in the one `cog` asset, so the bands share one grid.
    """
    from eodh_workflows.workflows.ds.utils import _prepare_s2_ard_data_array_clip
    from eodh_workflows.workflows.spectral.utils import resolve_rescale_params
    from rioxarray.raster_array import RasterArray

    original_reproject = RasterArray.reproject
    RasterArray.reproject = lambda self, *args, **kwargs: self._obj
    try:
        cog = _prepare_s2_ard_data_array_clip(
            item, aoi=AOI, asset_name="cog", assign_coords={"band": EOPRO_ARD_BANDS}
        ).where(lambda x: x > 0, other=float("nan"))
    finally:
        RasterArray.reproject = original_reproject

    scale, offset = resolve_rescale_params(collection_name=item.collection_id, item_datetime=item.datetime)
    return cog, {
        "eopro_path": "_prepare_s2_ard_data_array_clip (no reprojection) -> where > 0 -> {index}.calculate_index",
        "eopro_collection_id": item.collection_id,
        "eopro_rescale": [scale, offset],
        "clip": "rasterio.mask.mask(all_touched=True, crop=True) to the AOI polygon",
        "transform": list(cog.rio.transform())[:6],
        "transform_source": "rioxarray (pixel-centre coordinates)",
    }


def _write_golden(index: str, golden: Any, meta: dict[str, Any], item: Any, out_dir: Path) -> None:
    import numpy as np

    array = np.asarray(golden.squeeze().values)
    if array.ndim != 2:
        raise SystemExit(f"Expected a 2-D {index} array, got shape {array.shape}")
    np.save(out_dir / f"{index}.npy", array)
    crs = golden.rio.crs
    meta = {
        "item_id": item.id,
        "collection": item.collection_id,
        "datetime": item.datetime.isoformat(),
        "eopro_commit": EOPRO_COMMIT,
        "crs": crs.to_string() if crs is not None else None,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "valid_pixels": int(np.isfinite(array).sum()),
        "aoi": "../aoi.geojson",
        **meta,
    }
    (out_dir / f"{index}.json").write_text(json.dumps(meta, indent=2) + "\n")


# --- Staged real-data fixtures ---------------------------------------------------------------------


def _write_staged(item: Any, asset_keys: list[str], staged_dir: Path) -> None:
    """Write a stage-in directory in the layout of `eodh_workflow.testing.stage_item`, with real data."""
    import shutil

    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    item_dict = copy.deepcopy(item.to_dict(include_self_link=False, transform_hrefs=False))
    item_dir = staged_dir / item.id
    item_dir.mkdir(parents=True)

    staged_assets = {}
    for key in asset_keys:
        asset = item_dict["assets"][key]
        shape, transform = _write_window(asset["href"], item_dir / f"{key}.tif")
        staged_assets[key] = {**asset, "href": f"./{key}.tif", "proj:shape": shape, "proj:transform": transform}
        print(f"  staged {key}: {shape[1]} x {shape[0]} px")
    item_dict["assets"] = staged_assets
    item_dict["links"] = []  # stage-in drops every item link
    for field in ("proj:shape", "proj:transform", "proj:bbox"):
        item_dict["properties"].pop(field, None)  # they describe the whole scene, not the staged window

    (item_dir / f"{item.id}.json").write_text(json.dumps(item_dict, indent=2) + "\n")
    catalog = {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": "catalog",
        "description": "staged",
        "links": [{"rel": "item", "href": f"./{item.id}/{item.id}.json", "type": "application/json"}],
    }
    (staged_dir / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


def _write_window(href: str, path: Path) -> tuple[list[int], list[float]]:
    """Copy the AOI window of a remote raster, on its own grid, to a small local COG."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    with rasterio.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *AOI_BOUNDS, densify_pts=21)
        # Round out to whole pixels by hand: rasterio's rounding methods differ between versions.
        exact = from_bounds(*bounds, transform=src.transform)
        col0 = math.floor(exact.col_off) - PAD_PIXELS
        row0 = math.floor(exact.row_off) - PAD_PIXELS
        col1 = math.ceil(exact.col_off + exact.width) + PAD_PIXELS
        row1 = math.ceil(exact.row_off + exact.height) + PAD_PIXELS
        window = Window(col0, row0, col1 - col0, row1 - row0).intersection(Window(0, 0, src.width, src.height))
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = {
            "driver": "COG",
            "width": data.shape[2],
            "height": data.shape[1],
            "count": data.shape[0],
            "dtype": data.dtype,
            "crs": src.crs,
            "transform": transform,
            "nodata": src.nodata,
            "compress": "deflate",
            "blocksize": 256,
        }
        descriptions = src.descriptions
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, description in enumerate(descriptions, start=1):
            if description:
                dst.set_band_description(i, description)
    return [int(data.shape[1]), int(data.shape[2])], list(transform)[:6]


if __name__ == "__main__":
    main()
