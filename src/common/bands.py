"""Band lookup by STAC common name, and the brightness rule for each supported collection."""

from typing import Optional, Tuple

from eodh_workflow import Band, StacItemInput
from eodh_workflow.inputs import asset_bands, band_common_name, is_geotiff

L2A = "sentinel-2-c1-l2a"
ARD = "sentinel2_ard"
SUPPORTED_COLLECTIONS = (L2A, ARD)


def require_item(stac_item: Optional[StacItemInput]) -> StacItemInput:
    if stac_item is None:
        raise ValueError("This workflow needs a staged STAC item (--stac_item_dir), and none was given")
    return stac_item


def find_band(stac_item: StacItemInput, common_name: str) -> Tuple[str, int]:
    """(asset key, 1-based band index) of the band with STAC common name `common_name`.

    Single-band assets win over multi-band ones; an asset keyed by the common name is the fallback.
    """
    candidates = []
    for key, asset in stac_item.assets.items():
        if not is_geotiff(asset):
            continue
        bands = asset_bands(asset)
        for index, entry in enumerate(bands, start=1):
            if band_common_name(entry) == common_name:
                candidates.append((len(bands), key, index))
    if candidates:
        _, key, index = min(candidates, key=lambda candidate: candidate[0])
        return key, index
    if common_name in stac_item.assets and is_geotiff(stac_item.assets[common_name]):
        return common_name, 1

    found = sorted(
        {name for asset in stac_item.assets.values() for name in map(band_common_name, asset_bands(asset)) if name}
    )
    raise KeyError(f"Item {stac_item.id!r} has no band with common name {common_name!r}. Common names: {found}")


def band(stac_item: StacItemInput, common_name: str) -> Band:
    """The raster band with STAC common name `common_name`, as (path, 1-based index)."""
    key, index = find_band(stac_item, common_name)
    return Band(stac_item.asset_path(key), index)


def brightness(stac_item: StacItemInput, common_name: str) -> Tuple[float, float]:
    """(scale, offset) that turns the raw values of a band into reflectance, by collection.

    sentinel-2-c1-l2a: the asset's `raster:bands` scale and offset, applied once.
    sentinel2_ard: the raw values, scale 1 and offset 0.
    """
    collection = stac_item.item.get("collection")
    if collection == ARD:
        return 1.0, 0.0
    if collection == L2A:
        key, index = find_band(stac_item, common_name)
        raster_bands = stac_item.assets[key].get("raster:bands") or []
        entry = raster_bands[index - 1] if len(raster_bands) >= index else {}
        return float(entry.get("scale", 1.0)), float(entry.get("offset", 0.0))
    raise ValueError(
        f"Item {stac_item.id!r} is in collection {collection!r}; supported collections: {list(SUPPORTED_COLLECTIONS)}"
    )
