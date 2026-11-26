"""
The baseline item for a change product: the clearest item of the input item's own collection, on the
same tile, inside a date range before the input item.

The search goes to the input item's own catalogue: the `collection` link up to `/collections/<id>`,
plus `/search`. Sentinel-2 L2A results carry `grid:code` and `eo:cloud_cover`, so only the chosen
item is fetched. Sentinel-2 ARD results carry only `datetime`, so each candidate is fetched to read
`eo:cloud_cover`, `proj:code` and `proj:transform`.
"""

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Tuple
from urllib.parse import urlencode, urljoin

from shapely.geometry import shape

from eodh_workflow import StacItemInput

from .fetch import FETCH_TIMEOUT, fetch_item, get_json

logger = logging.getLogger(__name__)

# The search stops after this many results; the best item found so far is used.
MAX_SEARCH_ITEMS = 200
SEARCH_PAGE_SIZE = 100

_ARD_TILE = re.compile(r"_T(\d{2}[A-Z]{3})_")


@dataclass
class _Candidate:
    item: dict
    fetched: Optional[StacItemInput] = None


def find_baseline_item(
    item: StacItemInput, aoi: dict, date_range: str, timeout: float = FETCH_TIMEOUT
) -> StacItemInput:
    """The baseline item for `item`: same collection and tile, footprint covering `aoi`, datetime in
    `date_range` (`start/end`, dates or datetimes), least cloud first, then closest in time to `item`.

    Raises ValueError for a bad range, and LookupError when no item matches.
    """
    item_time = _item_datetime(item.item)
    if item_time is None:
        raise ValueError(f"Item {item.id!r} has no datetime, so a baseline range cannot be checked against it")
    start, end = _parse_range(date_range, item_time)

    collection = item.item.get("collection")
    search_url = _search_url(item.item)
    tile = _tile(item.item)
    area = shape(aoi)
    query = {
        "collections": collection,
        "bbox": ",".join(repr(float(v)) for v in area.bounds),
        "datetime": f"{_rfc3339(start)}/{_rfc3339(end)}",
        "limit": SEARCH_PAGE_SIZE,
    }
    url: Optional[str] = f"{search_url}?{urlencode(query)}"

    best: Optional[_Candidate] = None
    best_key = None
    seen_ids = set()
    count = 0
    while url:
        page = get_json(url, timeout, what="STAC search")
        features = page.get("features") or []
        for result in features:
            count += 1
            result_id = result.get("id")
            if result_id == item.id or result_id in seen_ids:
                continue
            seen_ids.add(result_id)
            candidate = _match(result, tile, area, search_url, timeout)
            if candidate is None:
                continue
            key = _sort_key(candidate.item, item_time)
            if best_key is None or key < best_key:
                best, best_key = candidate, key
        url = _next_link(page, url) if features else None
        if url and count >= MAX_SEARCH_ITEMS:
            logger.warning(f"Baseline search stopped after {count} results; using the best item found so far")
            break

    if best is None:
        raise LookupError(
            f"No baseline item in collection {collection!r} for the range {date_range!r} "
            f"on tile {_describe(tile, item.id)}"
        )
    chosen = best.fetched or fetch_item(_item_url(best.item, search_url), timeout)
    logger.info(
        f"Baseline item {chosen.id!r} ({chosen.properties.get('datetime')}, "
        f"cloud {chosen.properties.get('eo:cloud_cover')}) for item {item.id!r}"
    )
    return chosen


def _parse_instant(text: str, end: bool) -> datetime:
    """A date (start or end of that day) or a datetime, in UTC. Naive datetimes are taken as UTC."""
    text = text.strip()
    try:
        day = date.fromisoformat(text)
    except ValueError:
        moment = datetime.fromisoformat(text)
        return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
    moment = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return moment + timedelta(days=1, microseconds=-1) if end else moment


def _parse_range(date_range: str, item_time: datetime) -> Tuple[datetime, datetime]:
    item_date = item_time.date().isoformat()
    problem = f"The baseline range {date_range!r} must be 'start/end' (dates or datetimes)"
    parts = str(date_range).split("/")
    if len(parts) != 2:
        raise ValueError(f"{problem}, ending before the item's date {item_date}")
    try:
        start, end = _parse_instant(parts[0], end=False), _parse_instant(parts[1], end=True)
    except ValueError as error:
        raise ValueError(f"{problem}, ending before the item's date {item_date}: {error}") from None
    if end < start:
        raise ValueError(f"The baseline range {date_range!r} ends before it starts (item date {item_date})")
    if end > item_time:
        hint = ""
        if _is_date(parts[1]):
            hint = "; a date-only end includes that whole day, so end the range on an earlier day"
        raise ValueError(
            f"The baseline range {date_range!r} ends after the item's datetime {_rfc3339(item_time)} "
            f"({item_date}){hint}"
        )
    return start, end


def _is_date(text: str) -> bool:
    try:
        date.fromisoformat(text.strip())
    except ValueError:
        return False
    return True


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _item_datetime(item: dict) -> Optional[datetime]:
    properties = item.get("properties") or {}
    value = properties.get("datetime") or properties.get("start_datetime")
    if not value:
        return None
    try:
        return _parse_instant(str(value), end=False)
    except ValueError:
        return None


def _search_url(item: dict) -> str:
    """`<catalogue>/search`, from the item's `collection` link `<catalogue>/collections/<id>`."""
    for link in item.get("links") or []:
        href = link.get("href") or ""
        if link.get("rel") == "collection" and "/collections/" in href:
            return href.rsplit("/collections/", 1)[0] + "/search"
    raise ValueError(f"Item {item.get('id')!r} has no collection link of the form <catalogue>/collections/<id>")


def _proj_code(properties: dict) -> Optional[str]:
    if properties.get("proj:code"):
        return str(properties["proj:code"])
    if properties.get("proj:epsg"):
        return f"EPSG:{properties['proj:epsg']}"
    return None


def _tile(item: dict) -> dict:
    """What puts an item on a tile: `grid:code`, or else `proj:code` and `proj:transform`."""
    properties = item.get("properties") or {}
    if properties.get("grid:code"):
        return {"grid:code": properties["grid:code"]}
    code, transform = _proj_code(properties), properties.get("proj:transform")
    if code and transform:
        return {"proj:code": code, "proj:transform": transform}
    raise ValueError(
        f"Item {item.get('id')!r} has neither grid:code nor proj:code and proj:transform, so its tile is not known"
    )


def _tile_of(properties: dict, tile: dict) -> Optional[dict]:
    """The values of `tile`'s keys in `properties`, or None if any is missing."""
    if "grid:code" in tile:
        code = properties.get("grid:code")
        return {"grid:code": code} if code else None
    code, transform = _proj_code(properties), properties.get("proj:transform")
    return {"proj:code": code, "proj:transform": transform} if code and transform else None


def _same_tile(found: dict, tile: dict) -> bool:
    if "grid:code" in tile:
        return found["grid:code"] == tile["grid:code"]
    a, b = found["proj:transform"], tile["proj:transform"]
    return (
        found["proj:code"].upper() == tile["proj:code"].upper()
        and len(a) >= 6
        and len(b) >= 6
        and all(math.isclose(float(x), float(y), abs_tol=1e-6) for x, y in zip(a[:6], b[:6]))
    )


def _covers(geometry: Optional[dict], area) -> bool:
    if not geometry:
        return False
    try:
        return shape(geometry).covers(area)
    except (ValueError, TypeError, AttributeError):
        return False


def _item_url(result: dict, search_url: str) -> str:
    for link in result.get("links") or []:
        if link.get("rel") == "self" and link.get("href"):
            return urljoin(search_url, link["href"])
    collections = search_url.rsplit("/", 1)[0] + "/collections"
    return f"{collections}/{result.get('collection')}/items/{result.get('id')}"


def _match(result: dict, tile: dict, area, search_url: str, timeout: float) -> Optional[_Candidate]:
    """The search result as a candidate if it covers the AOI and is on `tile`, fetching the item when
    the result does not say which tile it is on."""
    # A result without geometry is dropped without being fetched, on purpose: Element84 and CEDA
    # search results both carry geometry.
    if not _covers(result.get("geometry"), area):
        return None
    candidate = _Candidate(result)
    found = _tile_of(result.get("properties") or {}, tile)
    if found is None:
        candidate.fetched = fetch_item(_item_url(result, search_url), timeout)
        candidate.item = candidate.fetched.item
        if not _covers(candidate.item.get("geometry"), area):
            return None
        found = _tile_of(candidate.fetched.properties, tile)
        if found is None:
            return None
    return candidate if _same_tile(found, tile) else None


def _sort_key(item: dict, item_time: datetime):
    cloud = (item.get("properties") or {}).get("eo:cloud_cover")
    try:
        cloud = float(cloud)
    except (TypeError, ValueError):
        cloud = None
    if cloud is not None and math.isnan(cloud):
        cloud = None
    when = _item_datetime(item)
    distance = abs((item_time - when).total_seconds()) if when else math.inf
    return (cloud is None, cloud if cloud is not None else 0.0, distance, str(item.get("id")))


def _next_link(page: dict, url: str) -> Optional[str]:
    for link in page.get("links") or []:
        if link.get("rel") == "next" and link.get("href"):
            if (link.get("method") or "GET").upper() != "GET":
                return None
            return urljoin(url, link["href"])
    return None


def _describe(tile: dict, item_id: Optional[str]) -> str:
    if "grid:code" in tile:
        return str(tile["grid:code"])
    found = _ARD_TILE.search(str(item_id or ""))
    if found:
        return f"T{found.group(1)} ({tile['proj:code']}, transform {tile['proj:transform']})"
    return f"{tile['proj:code']} with transform {tile['proj:transform']}"
