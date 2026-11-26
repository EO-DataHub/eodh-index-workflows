"""NDVI, (NIR - Red) / (NIR + Red), with EOPro's rules: reflectance <= 0 is nodata, no cloud mask."""

from pathlib import Path

import numpy as np

from common import band, brightness, item_input, parse_aoi, write_aoi_cog
from eodh_workflow import CollectionSpec, ItemSpec, OutputSpec, RasterAsset, WorkflowInputs

NAME = "eodh-ndvi"

RENDERS = {
    "ndvi": {
        "title": "NDVI",
        "assets": ["data"],
        "rescale": [[-0.2, 1]],
        "colormap_name": "ylgn",
    }
}


def reflectance(raw: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Raw values scaled once to reflectance, in float64. Values <= 0 after scaling are NaN."""
    values = raw.astype("float64") * scale + offset
    values[values <= 0] = np.nan
    return values


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - red) / (nir + red)


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    item = item_input(inputs)
    geometry = parse_aoi(inputs.params.get("aoi"))
    red_rule, nir_rule = brightness(item, "red"), brightness(item, "nir")

    def compute(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        return ndvi(reflectance(red, *red_rule), reflectance(nir, *nir_rule)).astype("float32")

    item_id = f"{item.id}_ndvi"
    cog = output_dir / f"{item_id}.tif"
    write_aoi_cog(item.id, [band(item, "red"), band(item, "nir")], geometry, cog, compute)

    return OutputSpec(
        collection=CollectionSpec(
            id=NAME,
            title="NDVI",
            description=(
                "Normalised Difference Vegetation Index, (NIR - Red) / (NIR + Red), from Sentinel-2 "
                "surface reflectance over the area of interest."
            ),
            keywords=["ndvi", "vegetation", "sentinel-2"],
            renders=RENDERS,
        ),
        items=[ItemSpec(id=item_id, source_item=item, assets={"data": RasterAsset(cog, bands=[{"name": "ndvi"}])})],
    )
