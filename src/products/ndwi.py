"""NDWI, (Green - NIR) / (Green + NIR), McFeeters' water index, with EOPro's rules."""

from pathlib import Path

import numpy as np

from common import run_index_product
from eodh_workflow import OutputSpec, WorkflowInputs

NAME = "eodh-ndwi"


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (green - nir) / (green + nir)


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    return run_index_product(
        inputs,
        output_dir,
        name=NAME,
        index="ndwi",
        bands=("green", "nir"),
        formula=ndwi,
        title="NDWI",
        description=(
            "Normalised Difference Water Index, (Green - NIR) / (Green + NIR), from Sentinel-2 surface "
            "reflectance over the area of interest. High values are water."
        ),
        keywords=["ndwi", "water", "sentinel-2"],
        render={"rescale": [[-1, 1]], "colormap_name": "rdbu"},
    )
