"""NBR, (NIR - SWIR22) / (NIR + SWIR22), on the 10 m NIR grid, with EOPro's rules."""

from pathlib import Path

import numpy as np

from common import run_index_product
from eodh_workflow import OutputSpec, WorkflowInputs

NAME = "eodh-nbr"


def nbr(nir: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - swir22) / (nir + swir22)


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    # nir comes first, so the output is on its 10 m grid; the 20 m swir22 is read onto it by nearest.
    return run_index_product(
        inputs,
        output_dir,
        name=NAME,
        index="nbr",
        bands=("nir", "swir22"),
        formula=nbr,
        title="NBR",
        description=(
            "Normalised Burn Ratio, (NIR - SWIR2) / (NIR + SWIR2), from Sentinel-2 surface reflectance "
            "over the area of interest."
        ),
        keywords=["nbr", "burn", "fire", "sentinel-2"],
        render={"rescale": [[-1, 1]], "colormap_name": "rdylgn"},
    )
