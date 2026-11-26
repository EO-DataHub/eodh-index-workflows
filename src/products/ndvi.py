"""NDVI, (NIR - Red) / (NIR + Red), with EOPro's rules: reflectance <= 0 is nodata, no cloud mask."""

from pathlib import Path

import numpy as np

from common import run_index_product
from eodh_workflow import OutputSpec, WorkflowInputs

NAME = "eodh-ndvi"


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - red) / (nir + red)


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    return run_index_product(
        inputs,
        output_dir,
        name=NAME,
        index="ndvi",
        bands=("red", "nir"),
        formula=ndvi,
        title="NDVI",
        description=(
            "Normalised Difference Vegetation Index, (NIR - Red) / (NIR + Red), from Sentinel-2 "
            "surface reflectance over the area of interest."
        ),
        keywords=["ndvi", "vegetation", "sentinel-2"],
        render={"rescale": [[-0.2, 1]], "colormap_name": "ylgn"},
    )
