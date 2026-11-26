"""EVI, G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L), with EOPro's coefficients and rules."""

from pathlib import Path

import numpy as np

from common import run_index_product
from eodh_workflow import OutputSpec, WorkflowInputs

NAME = "eodh-evi"

# EOPro takes these from the xarray-spatial 0.4.0 defaults for `evi`. They are copied here, so
# the image does not depend on xarray-spatial; the golden arrays check them.
G = 2.5
C1 = 6.0
C2 = 7.5
L = 1.0


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return G * (nir - red) / (nir + C1 * red - C2 * blue + L)


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    return run_index_product(
        inputs,
        output_dir,
        name=NAME,
        index="evi",
        bands=("nir", "red", "blue"),
        formula=evi,
        title="EVI",
        description=(
            "Enhanced Vegetation Index, 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1), from "
            "Sentinel-2 surface reflectance over the area of interest."
        ),
        keywords=["evi", "vegetation", "sentinel-2"],
        render={"rescale": [[-1, 1]], "colormap_name": "ylgn"},
    )
