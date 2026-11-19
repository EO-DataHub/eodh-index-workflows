"""
The area of interest: parse the `aoi` parameter, and write a raster clipped to it.

The output covers the overlap of the AOI's bounding box and the scene, on the scene's own pixel
grid. Pixels whose centre is outside the AOI polygon are NaN. The polygon becomes a mask raster
on that grid, which `write_cog_chunked` reads as one more input, so no chunk needs the geometry.
"""

import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from affine import Affine
from rasterio.dtypes import dtype_rev, typename_fwd
from rasterio.errors import RasterioIOError
from rasterio.features import rasterize
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds, transform_geom

from eodh_workflow import Band, write_cog_chunked

from .bands import gdal_path

AOI_TYPES = ("Polygon", "MultiPolygon")

# Points per edge when the AOI's bounding box is reprojected, so curved edges are followed.
_DENSIFY_POINTS = 21


def parse_aoi(value: Optional[str]) -> dict:
    """The AOI geometry (EPSG:4326) from the `aoi` parameter: a GeoJSON Polygon, MultiPolygon, or a
    Feature with one of those, as a string."""
    if value is None or not str(value).strip():
        raise ValueError("The aoi parameter is required: a GeoJSON Polygon, MultiPolygon or Feature, as a string")
    try:
        geojson = json.loads(value)
    except (TypeError, ValueError):
        raise ValueError(f"The aoi parameter is not GeoJSON. Received: {_preview(value)}") from None
    if not isinstance(geojson, dict):
        raise ValueError(f"The aoi parameter must be a GeoJSON object. Received: {_preview(value)}")

    received = geojson.get("type")
    geometry = geojson
    if received == "Feature":
        geometry = geojson.get("geometry") or {}
        received = f"Feature with geometry {geometry.get('type')}"
    if geometry.get("type") not in AOI_TYPES or not geometry.get("coordinates"):
        raise ValueError(
            "The aoi parameter must be a GeoJSON Polygon, MultiPolygon, or a Feature with one of those. "
            f"Received: {received}"
        )
    return {"type": geometry["type"], "coordinates": geometry["coordinates"]}


def _preview(value, limit: int = 200) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _geometry_bounds(geometry: dict) -> Tuple[float, float, float, float]:
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    rings = [ring for polygon in polygons for ring in polygon]
    xs = [point[0] for ring in rings for point in ring]
    ys = [point[1] for ring in rings for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


def aoi_grid(item_id: str, src, geometry: dict) -> Tuple[Tuple[int, int, int, int], Affine, np.ndarray]:
    """Pixel window (col, row, width, height) of the scene `src` that the AOI's bounding box covers,
    its transform, and the polygon mask on it (True where the pixel centre is inside the AOI)."""
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", src.crs, *_geometry_bounds(geometry), densify_pts=_DENSIFY_POINTS
    )
    inverse = ~src.transform
    corners = [inverse @ (x, y) for x in (left, right) for y in (bottom, top)]
    col0 = max(math.floor(min(c for c, _ in corners)), 0)
    row0 = max(math.floor(min(r for _, r in corners)), 0)
    col1 = min(math.ceil(max(c for c, _ in corners)), src.width)
    row1 = min(math.ceil(max(r for _, r in corners)), src.height)
    if col1 <= col0 or row1 <= row0:
        raise ValueError(f"AOI does not overlap item {item_id}")

    width, height = col1 - col0, row1 - row0
    transform = src.transform @ Affine.translation(col0, row0)
    projected = transform_geom("EPSG:4326", src.crs, geometry)
    touched = rasterize([projected], out_shape=(height, width), transform=transform, all_touched=True)
    if not touched.any():
        raise ValueError(f"AOI does not overlap item {item_id}")
    inside = rasterize([projected], out_shape=(height, width), transform=transform).astype(bool)
    return (col0, row0, width, height), transform, inside


def _vrt_band(src, band: Band, window: Tuple[int, int, int, int], transform: Affine, path: Path) -> Band:
    """A one-band VRT of `band` cut to `window`, so every input shares the output grid."""
    col, row, width, height = window
    data_type = typename_fwd[dtype_rev[src.dtypes[band.index - 1]]]
    nodata = src.nodatavals[band.index - 1]
    nodata_xml = f"<NoDataValue>{nodata!r}</NoDataValue>" if nodata is not None else ""
    path.write_text(
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f"<SRS>{escape(src.crs.to_wkt())}</SRS>"
        f"<GeoTransform>{', '.join(repr(v) for v in transform.to_gdal())}</GeoTransform>"
        f'<VRTRasterBand dataType="{data_type}" band="1">{nodata_xml}'
        "<SimpleSource>"
        f'<SourceFilename relativeToVRT="0">{escape(gdal_path(band.path))}</SourceFilename>'
        f"<SourceBand>{band.index}</SourceBand>"
        f'<SrcRect xOff="{col}" yOff="{row}" xSize="{width}" ySize="{height}"/>'
        f'<DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>'
        "</SimpleSource></VRTRasterBand></VRTDataset>"
    )
    return Band(path, 1)


@contextmanager
def _open_band(band: Band) -> Iterator[DatasetReader]:
    """Open the file of `band`, local or remote. A failed open names the band's location."""
    try:
        src = rasterio.open(gdal_path(band.path))
    except RasterioIOError as error:
        raise RasterioIOError(f"Could not read band {band.index} at {band.path}: {error}") from error
    with src:
        yield src


def _write_mask(path: Path, inside: np.ndarray, crs, transform: Affine) -> Band:
    """The polygon mask as a raster: 1 inside, nodata (0) outside."""
    profile = {
        "driver": "GTiff",
        "width": inside.shape[1],
        "height": inside.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(inside.astype("uint8"), 1)
    return Band(path, 1)


def write_aoi_cog(
    item_id: str,
    bands: Sequence[Band],
    geometry: dict,
    output_cog: Path,
    compute: Callable[..., np.ndarray],
) -> Path:
    """Write `compute(*arrays)` over the AOI as a float32 COG, on the grid of `bands[0]`.

    NaN outside the AOI polygon and wherever an input pixel is nodata.
    """
    with _open_band(bands[0]) as reference:
        window, transform, inside = aoi_grid(item_id, reference, geometry)
        crs, scene_transform = reference.crs, reference.transform

    with tempfile.TemporaryDirectory(prefix="aoi-") as workdir:
        workdir = Path(workdir)
        clipped = []
        for number, band in enumerate(bands):
            with _open_band(band) as src:
                if src.crs != crs or src.transform != scene_transform:
                    raise ValueError(f"Band {band.path} is not on the same grid as {bands[0].path}")
                clipped.append(_vrt_band(src, band, window, transform, workdir / f"band{number}.vrt"))
        mask = _write_mask(workdir / "aoi_mask.tif", inside, crs, transform)

        def compute_inside(*arrays: np.ndarray) -> np.ndarray:
            return compute(*arrays[:-1])

        return write_cog_chunked([*clipped, mask], output_cog, compute_inside)
