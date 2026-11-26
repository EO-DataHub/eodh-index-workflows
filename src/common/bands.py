"""Band lookup by STAC common name, and the brightness rule for each supported collection."""

from pathlib import Path
from typing import Tuple, Union
from urllib.parse import unquote, urlparse

from eodh_workflow import Band, StacItemInput
from eodh_workflow.inputs import asset_bands, band_common_name, is_geotiff

L2A = "sentinel-2-c1-l2a"
ARD = "sentinel2_ard"
SUPPORTED_COLLECTIONS = (L2A, ARD)


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


REMOTE_SCHEMES = ("http://", "https://", "file://")


def is_remote(href: str) -> bool:
    """True for an absolute http(s):// or file:// href, read in place rather than from stage-in."""
    return href.lower().startswith(REMOTE_SCHEMES)


def band(stac_item: StacItemInput, common_name: str) -> Band:
    """The raster band with STAC common name `common_name`, as (location, 1-based index).

    The location is the local path for a relative staged href, and the href itself, as a string,
    for an absolute http(s):// or file:// href.
    """
    key, index = find_band(stac_item, common_name)
    href = stac_item.assets[key].get("href") or ""
    scheme = urlparse(href).scheme
    if scheme and not is_remote(href):
        raise ValueError(
            f"Cannot read asset {key!r} of item {stac_item.id!r} at {href}: only http(s):// and file:// hrefs "
            "can be read without credentials"
        )
    if is_remote(href):
        return Band(href, index)
    return Band(stac_item.asset_path(key), index)


def gdal_path(location: Union[str, Path]) -> str:
    """The name GDAL opens for a band location: /vsicurl/ for http(s), the local path for file://."""
    text = str(location)
    lower = text.lower()
    if lower.startswith(("http://", "https://")):
        return f"/vsicurl/{text}"
    if lower.startswith("file://"):
        parsed = urlparse(text)
        if parsed.netloc.lower() not in ("", "localhost"):
            raise ValueError(f"Cannot read {text!r}: a file:// href must have no host or 'localhost'")
        return unquote(parsed.path)
    return text


def brightness(stac_item: StacItemInput, common_name: str) -> Tuple[float, float]:
    """(scale, offset) that turns the raw values of a band into reflectance, by collection.

    sentinel-2-c1-l2a: the asset's `raster:bands` scale and offset, applied once. A missing scale
    fails; a missing offset is 0, as in the STAC raster extension.
    sentinel2_ard: the raw values, scale 1 and offset 0.
    """
    collection = stac_item.item.get("collection")
    if collection == ARD:
        return 1.0, 0.0
    if collection == L2A:
        key, index = find_band(stac_item, common_name)
        raster_bands = stac_item.assets[key].get("raster:bands") or []
        entry = raster_bands[index - 1] if len(raster_bands) >= index else {}
        if "scale" not in entry:
            raise ValueError(
                f"Item {stac_item.id!r} asset {key!r} band {index} has no raster:bands scale; "
                f"it is needed to turn {collection} values into reflectance"
            )
        return float(entry["scale"]), float(entry.get("offset", 0.0))
    raise ValueError(
        f"Item {stac_item.id!r} is in collection {collection!r}; supported collections: {list(SUPPORTED_COLLECTIONS)}"
    )
