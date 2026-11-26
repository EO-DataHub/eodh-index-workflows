"""A single-index product: bands from one item, scaled to reflectance, combined by one formula over the AOI."""

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from eodh_workflow import CollectionSpec, ItemSpec, OutputSpec, RasterAsset, WorkflowInputs

from .aoi import parse_aoi, write_aoi_cog
from .bands import band, brightness
from .fetch import item_input


def reflectance(raw: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Raw values scaled once to reflectance, in float64. Values <= 0 after scaling are NaN."""
    values = raw.astype("float64") * scale + offset
    values[values <= 0] = np.nan
    return values


def run_index_product(
    inputs: WorkflowInputs,
    output_dir: Path,
    *,
    name: str,
    index: str,
    bands: Sequence[str],
    formula: Callable[..., np.ndarray],
    title: str,
    description: str,
    keywords: Sequence[str],
    render: dict,
) -> OutputSpec:
    """Write one index over the AOI as a float32 COG, and return its `OutputSpec`.

    name: the workflow ID, which is also the catalog ID.
    index: short name of the index, for the item ID suffix, the band name and the render key.
    bands: STAC common names of the input bands. The output is on the grid of the first one; the
        others are read onto it by nearest neighbour when their grid differs.
    formula: called with each band's reflectance as a keyword argument named by its common name.
        Reflectance <= 0 is NaN before the formula sees it.
    render: the render without `assets`, for example `{"rescale": [[-1, 1]], "colormap_name": "ylgn"}`.
        Its `title` defaults to `title`.
    """
    item = item_input(inputs)
    geometry = parse_aoi(inputs.params.get("aoi"))
    names = list(bands)
    rules = [brightness(item, common_name) for common_name in names]

    def compute(*raw: np.ndarray) -> np.ndarray:
        values = {common_name: reflectance(data, *rule) for common_name, data, rule in zip(names, raw, rules)}
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(formula(**values)).astype("float32")

    item_id = f"{item.id}_{index}"
    cog = output_dir / f"{item_id}.tif"
    write_aoi_cog(item.id, [band(item, common_name) for common_name in names], geometry, cog, compute)

    renders = {index: {"title": title, "assets": ["data"], **render}}
    return OutputSpec(
        collection=CollectionSpec(
            id=name,
            title=title,
            description=description,
            keywords=list(keywords),
            renders=renders,
        ),
        items=[ItemSpec(id=item_id, source_item=item, assets={"data": RasterAsset(cog, bands=[{"name": index}])})],
    )
