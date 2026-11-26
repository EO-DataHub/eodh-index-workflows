"""SAVI as EOPro computes it, (NIR - Red) / ((NIR + Red + L) * (1 + L)) with L = 0.5, and EOPro's rules."""

from pathlib import Path

import numpy as np

from common import run_index_product
from eodh_workflow import OutputSpec, WorkflowInputs

NAME = "eodh-savi"

# Soil brightness factor. EOPro uses xarray-spatial 0.4.0, which divides by (1 + L) rather than
# multiplying by it as the textbook SAVI does; this module matches EOPro.
L = 0.5


def savi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - red) / ((nir + red + L) * (1 + L))


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    return run_index_product(
        inputs,
        output_dir,
        name=NAME,
        index="savi",
        bands=("nir", "red"),
        formula=savi,
        title="SAVI",
        description=(
            "Soil Adjusted Vegetation Index, (NIR - Red) / ((NIR + Red + 0.5) * 1.5), from Sentinel-2 "
            "surface reflectance over the area of interest."
        ),
        keywords=["savi", "vegetation", "sentinel-2"],
        render={"rescale": [[-1, 1]], "colormap_name": "ylgn"},
    )
