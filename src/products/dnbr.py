"""dNBR, NBR(baseline) - NBR(input), over the AOI, and its burn-severity classes.

The input item is the image after the fire. The baseline item, before the fire, is found by
`find_baseline_item` from the `baseline` date range. Both NBRs are on the grid of the input item's
nir band, so positive dNBR is a loss of NBR since the baseline: burnt ground.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pystac
import rasterio

from common import band, brightness, find_baseline_item, item_input, parse_aoi, reflectance, write_aoi_cog
from eodh_workflow import CollectionSpec, ItemSpec, OutputSpec, RasterAsset, StacItemInput, WorkflowInputs
from eodh_workflow.cog import cog_from_gtiff, temp_path_for
from products.nbr import nbr

NAME = "eodh-dnbr"
BASELINE_PARAM = "baseline"
BANDS = ("nir", "swir22")

# USGS burn-severity classes from dNBR, after Key & Benson (2006), "Landscape Assessment (LA):
# Sampling and analysis methods", USDA Forest Service Gen. Tech. Rep. RMRS-GTR-164-CD. Each bound is
# the lowest dNBR of the class above it, so a pixel equal to a bound goes in the higher class.
SEVERITY_BOUNDS = (-0.25, -0.10, 0.10, 0.27, 0.44, 0.66)
# (class value, label, RGBA colour), lowest dNBR first. Colours are the usual USGS / UN-SPIDER palette.
SEVERITY_CLASSES = (
    (1, "Enhanced regrowth, high", (122, 135, 55, 255)),
    (2, "Enhanced regrowth, low", (172, 190, 77, 255)),
    (3, "Unburned", (10, 224, 66, 255)),
    (4, "Low severity", (255, 248, 22, 255)),
    (5, "Moderate-low severity", (255, 175, 40, 255)),
    (6, "Moderate-high severity", (255, 100, 27, 255)),
    (7, "High severity", (164, 31, 214, 255)),
)
SEVERITY_NODATA = 0

# dNBR is shown from -DNBR_RANGE to DNBR_RANGE, red for burnt (positive) and green for regrowth.
DNBR_RANGE = 1.0


def severity(dnbr: np.ndarray) -> np.ndarray:
    """The burn-severity class (1-7) of each dNBR value, and SEVERITY_NODATA where dNBR is NaN."""
    classes = (np.digitize(dnbr, SEVERITY_BOUNDS) + 1).astype("uint8")
    classes[~np.isfinite(dnbr)] = SEVERITY_NODATA
    return classes


def _write_severity(dnbr_cog: Path, output_cog: Path) -> Path:
    """Classify the dNBR COG block by block into a uint8 COG."""
    with rasterio.open(dnbr_cog) as src:
        profile = {
            "driver": "GTiff",
            "width": src.width,
            "height": src.height,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": SEVERITY_NODATA,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "bigtiff": "IF_SAFER",
        }
        temp_path = temp_path_for(output_cog)
        with rasterio.open(temp_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                dst.write(severity(src.read(1, window=window)), 1, window=window)
    cog_from_gtiff(temp_path, output_cog, overview_resampling="nearest")
    return output_cog


def _self_href(item: StacItemInput) -> Optional[str]:
    for link in item.item.get("links") or []:
        if link.get("rel") == "self" and link.get("href"):
            return link["href"]
    for link in item.item.get("links") or []:
        if link.get("rel") == "collection" and link.get("href"):
            return f"{link['href'].rstrip('/')}/items/{item.id}"
    return None


def run(inputs: WorkflowInputs, output_dir: Path) -> OutputSpec:
    date_range = (inputs.params.get(BASELINE_PARAM) or "").strip()
    if not date_range:
        raise ValueError("The baseline parameter is required: a date range before the item, as start/end")
    item = item_input(inputs)
    geometry = parse_aoi(inputs.params.get("aoi"))
    baseline = find_baseline_item(item, geometry, date_range)

    rules = [brightness(source, common_name) for source in (item, baseline) for common_name in BANDS]
    bands = [band(source, common_name) for source in (item, baseline) for common_name in BANDS]

    def compute(after_nir, after_swir22, before_nir, before_swir22) -> np.ndarray:
        after_nir, after_swir22, before_nir, before_swir22 = (
            reflectance(data, *rule) for data, rule in zip((after_nir, after_swir22, before_nir, before_swir22), rules)
        )
        after = nbr(nir=after_nir, swir22=after_swir22).astype("float32").astype("float64")
        before = nbr(nir=before_nir, swir22=before_swir22).astype("float32").astype("float64")
        return (before - after).astype("float32")

    item_id = f"{item.id}_dnbr"
    dnbr_cog = output_dir / f"{item_id}.tif"
    # nir of the input item comes first, so both NBRs are on its grid over the AOI.
    write_aoi_cog(item.id, bands, geometry, dnbr_cog, compute)
    severity_cog = _write_severity(dnbr_cog, output_dir / f"{item.id}_burn_severity.tif")

    input_href = (inputs.params.get("stac_item") or "").strip() or _self_href(item)
    baseline_href = _self_href(baseline)
    properties = {
        "eodh:baseline_item_id": baseline.id,
        "eodh:baseline_datetime": baseline.properties.get("datetime"),
    }
    if baseline_href:
        properties["eodh:baseline_item_href"] = baseline_href

    def add_derived_from(catalog: pystac.Catalog) -> None:
        for output_item in catalog.get_items(recursive=True):
            for href in dict.fromkeys(h for h in (input_href, baseline_href) if h):
                output_item.add_link(pystac.Link("derived_from", href, media_type=pystac.MediaType.GEOJSON))

    renders = {
        "dnbr": {
            "title": "dNBR",
            "assets": ["dnbr"],
            "rescale": [[-DNBR_RANGE, DNBR_RANGE]],
            "colormap_name": "rdylgn_r",
        },
        "severity": {
            "title": "Burn severity",
            "assets": ["severity"],
            # The classes are already 1-7; the identity rescale keeps them for the colormap.
            "rescale": [[0, 255]],
            "colormap": {str(value): list(colour) for value, _, colour in SEVERITY_CLASSES},
        },
    }
    classes_doc = ", ".join(f"{value} {label}" for value, label, _ in SEVERITY_CLASSES)
    return OutputSpec(
        collection=CollectionSpec(
            id=NAME,
            title="dNBR",
            description=(
                "Differenced Normalised Burn Ratio, NBR(baseline) - NBR(item), from Sentinel-2 surface "
                "reflectance over the area of interest, and the USGS burn-severity classes from it "
                f"({classes_doc})."
            ),
            keywords=["dnbr", "nbr", "burn", "burn severity", "fire", "change", "sentinel-2"],
            renders=renders,
        ),
        items=[
            ItemSpec(
                id=item_id,
                source_item=item,
                assets={
                    "dnbr": RasterAsset(dnbr_cog, bands=[{"name": "dnbr"}], title="dNBR (COG)"),
                    "severity": RasterAsset(
                        severity_cog,
                        bands=[{"name": "burn_severity"}],
                        title="Burn severity (COG)",
                        description=f"USGS burn-severity classes: {classes_doc}. 0 is nodata.",
                    ),
                },
                extra_properties=properties,
            )
        ],
        customise=add_derived_from,
    )
