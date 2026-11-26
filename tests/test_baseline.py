"""
The baseline search: find the "before" item for a change product in the input item's own catalogue.

A local fixture server plays the catalogue. Its search results copy the two real shapes: Element84
(sentinel-2-c1-l2a) results carry `grid:code` and `eo:cloud_cover`; CEDA (sentinel2_ard) results
carry only `datetime`, and the item itself has `eo:cloud_cover`, `proj:code` and `proj:transform`.
"""

import copy
import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from eodh_workflow import StacItemInput
from golden_data import AOI, GOLDEN, box, item_json_path

BASELINE = "2023-05-01/2023-05-31"
ITEM_DATE = "2023-06-08"
ARD_TRANSFORM = [10.0, 0.0, 399960.0, 0.0, -10.0, 300000.0]
OTHER_ORBIT_TRANSFORM = [10.0, 0.0, 380000.0, 0.0, -10.0, 310000.0]
PART_OF_AOI = box(-1.33, 51.70, -1.0, 51.80)


def golden_item(name: str) -> dict:
    return json.loads(item_json_path(GOLDEN / name / "staged").read_text())


class Catalogue:
    """A STAC catalogue at `<server>/catalogue` holding one input item, and the search results to answer with."""

    def __init__(self, server, kind: str):
        self.server, self.kind = server, kind
        self.template = golden_item(kind)
        self.collection = self.template["collection"]
        self.search_path = "/catalogue/search"
        self.search_url = server.base + self.search_path
        self.input = self.item(self.template["id"], self.template["properties"]["datetime"])
        if kind == "ard":
            self.input["properties"]["proj:transform"] = ARD_TRANSFORM

    def item_path(self, item_id: str) -> str:
        return f"/catalogue/collections/{self.collection}/items/{item_id}"

    def item(self, item_id: str, when: str, cloud=None, geometry=None, **properties) -> dict:
        """A full item of the collection, as its item URL serves it, with relative asset hrefs."""
        item = copy.deepcopy(self.template)
        item["id"] = item_id
        item["properties"]["datetime"] = when
        item["properties"].pop("eo:cloud_cover", None)
        if cloud is not None:
            item["properties"]["eo:cloud_cover"] = cloud
        if self.kind == "ard":
            item["properties"]["proj:transform"] = ARD_TRANSFORM
        item["properties"].update(properties)
        if geometry is not None:
            item["geometry"] = geometry
        item["assets"] = {key: {**asset, "href": f"./{key}.tif"} for key, asset in item["assets"].items()}
        item["links"] = [
            {"rel": "self", "href": self.server.base + self.item_path(item_id)},
            {"rel": "collection", "href": f"{self.server.base}/catalogue/collections/{self.collection}"},
        ]
        return item

    def candidate(self, item_id: str, when: str, cloud=None, geometry=None, **properties) -> dict:
        """Serve the item at its URL; return it as a search result of this collection shows it."""
        item = self.item(item_id, when, cloud, geometry, **properties)
        self.server.serve_page(self.item_path(item_id), json.dumps(item))
        if self.kind == "l2a":
            return item
        return {key: item[key] for key in ("type", "stac_version", "id", "collection", "geometry", "bbox", "links")} | {
            "properties": {"datetime": when}
        }

    def answer(self, *pages: list):
        """Answer the search with `pages` of results, each linked to the next by a `next` link."""
        for number, results in enumerate(pages):
            path = self.search_path if number == 0 else f"{self.search_path}?page={number + 1}"
            links = []
            if number + 1 < len(pages):
                links = [{"rel": "next", "href": f"{self.search_url}?page={number + 2}"}]
            self.server.serve_page(path, json.dumps({"type": "FeatureCollection", "features": results, "links": links}))

    def input_item(self) -> StacItemInput:
        return StacItemInput(item=self.input, item_path=Path(f"{self.input['id']}.json"))

    def searches(self) -> list:
        return [path for path in self.server.requests_with_query if path.startswith(self.search_path)]

    def fetched(self, item_id: str) -> bool:
        return self.item_path(item_id) in self.server.requests


@pytest.fixture
def catalogue(serve, tmp_path):
    """`catalogue("l2a" | "ard")`: a Catalogue on a fresh fixture server."""
    root = tmp_path / "www"
    root.mkdir()
    server = serve(root)
    return lambda kind: Catalogue(server, kind)


def find(catalogue: Catalogue, date_range: str = BASELINE, **kwargs) -> StacItemInput:
    from common import find_baseline_item

    return find_baseline_item(catalogue.input_item(), AOI, date_range, **kwargs)


def test_stage1_l2a_search_returns_least_cloudy_item_fetched_with_absolute_hrefs(catalogue):
    cat = catalogue("l2a")
    cat.answer(
        [
            cat.candidate("cloudy", "2023-05-20T11:00:00Z", cloud=40.0),
            cat.candidate("clear", "2023-05-10T11:00:00Z", cloud=2.0),
            cat.candidate("hazy", "2023-05-25T11:00:00Z", cloud=15.0),
        ]
    )

    baseline = find(cat)

    assert isinstance(baseline, StacItemInput)
    assert baseline.id == "clear"
    assert baseline.assets
    for asset in baseline.assets.values():
        assert asset["href"].startswith(cat.server.base + "/"), asset["href"]
    [search] = cat.searches()
    query = parse_qs(urlparse(search).query)
    assert query["collections"] == [cat.collection]
    assert [float(v) for v in query["bbox"][0].split(",")] == pytest.approx([-1.345, 51.735, -1.315, 51.753])
    assert "2023-05-01" in query["datetime"][0] and "2023-05-31" in query["datetime"][0]
    assert cat.fetched("clear")
    assert not cat.fetched("cloudy") and not cat.fetched("hazy"), "L2A results already carry cloud and tile"


def test_stage1_ard_search_reads_cloud_and_grid_from_each_fetched_item(catalogue):
    cat = catalogue("ard")
    other_orbit = "S2B_20230520_latn518lonw0008_T30UXC_ORB137_20230520120350_utm30n_osgb"
    cat.answer(
        [
            cat.candidate("other-crs", "2023-05-12T11:00:00Z", cloud=0.0, **{"proj:code": "EPSG:32630"}),
            cat.candidate(other_orbit, "2023-05-20T11:00:00Z", cloud=0.0, **{"proj:transform": OTHER_ORBIT_TRANSFORM}),
            cat.candidate("cloudy", "2023-05-15T11:00:00Z", cloud=30.0),
            cat.candidate("clear", "2023-05-05T11:00:00Z", cloud=5.0),
        ]
    )

    baseline = find(cat)

    assert baseline.id == "clear"
    assert baseline.properties["proj:transform"] == ARD_TRANSFORM
    for asset in baseline.assets.values():
        assert asset["href"].startswith(cat.server.base + "/"), asset["href"]


def test_stage1_items_off_the_tile_not_covering_the_aoi_or_the_input_itself_are_dropped(catalogue):
    cat = catalogue("l2a")
    cat.answer(
        [
            cat.candidate(cat.input["id"], cat.input["properties"]["datetime"], cloud=0.0),
            cat.candidate("other-tile", "2023-05-10T11:00:00Z", cloud=0.0, **{"grid:code": "MGRS-30UXD"}),
            cat.candidate("part-of-aoi", "2023-05-12T11:00:00Z", cloud=0.0, geometry=PART_OF_AOI),
            cat.candidate("kept", "2023-05-14T11:00:00Z", cloud=50.0),
        ]
    )

    assert find(cat).id == "kept"


def test_stage1_search_follows_next_links_to_later_pages(catalogue):
    cat = catalogue("l2a")
    cat.answer(
        [cat.candidate("page1", "2023-05-10T11:00:00Z", cloud=30.0)],
        [cat.candidate("page2", "2023-05-12T11:00:00Z", cloud=20.0)],
        [cat.candidate("page3", "2023-05-14T11:00:00Z", cloud=1.0)],
    )

    assert find(cat).id == "page3"
    assert len(cat.searches()) == 3


def test_stage1_search_stops_following_next_links_at_a_limit(catalogue):
    cat = catalogue("l2a")
    results = [cat.candidate(f"item{n}", f"2023-05-{10 + n}T11:00:00Z", cloud=10.0 + n) for n in range(5)]
    page = {"type": "FeatureCollection", "features": results, "links": [{"rel": "next", "href": cat.search_url}]}
    cat.server.serve_page(cat.search_path, json.dumps(page))

    assert find(cat).id == "item0"
    assert 1 < len(cat.searches()) < 100


def test_stage1_cloud_tie_goes_to_the_item_closest_in_time(catalogue):
    cat = catalogue("l2a")
    cat.answer(
        [
            cat.candidate("early", "2023-05-02T11:00:00Z", cloud=5.0),
            cat.candidate("late", "2023-05-30T11:00:00Z", cloud=5.0),
            cat.candidate("middle", "2023-05-16T11:00:00Z", cloud=5.0),
        ]
    )

    assert find(cat).id == "late"


def test_stage1_item_without_cloud_cover_sorts_last(catalogue):
    cat = catalogue("l2a")
    cat.answer(
        [
            cat.candidate("no-cloud-value", "2023-05-30T11:00:00Z"),
            cat.candidate("very-cloudy", "2023-05-02T11:00:00Z", cloud=60.0),
        ]
    )

    assert find(cat).id == "very-cloudy"


@pytest.mark.parametrize(
    "date_range",
    ["2023-05-01/2023-05-31", "2023-05-01T00:00:00Z/2023-05-31T23:59:59Z"],
    ids=["dates", "datetimes"],
)
def test_stage1_baseline_range_is_read_from_dates_or_datetimes(date_range, catalogue):
    cat = catalogue("l2a")
    cat.answer([cat.candidate("clear", "2023-05-10T11:00:00Z", cloud=2.0)])

    assert find(cat, date_range).id == "clear"
    [search] = cat.searches()
    assert "2023-05-01" in parse_qs(urlparse(search).query)["datetime"][0]


@pytest.mark.parametrize(
    "date_range",
    ["not-a-range", "2023-05-01", "2023-05-31/2023-05-01", "2023-13-01/2023-05-31", "2023-05-01/2023-07-01"],
    ids=["not-a-range", "no-end", "end-before-start", "bad-month", "ends-after-the-item"],
)
def test_stage1_bad_baseline_range_fails_naming_the_range_and_the_item_date(date_range, catalogue):
    cat = catalogue("l2a")
    cat.answer([cat.candidate("clear", "2023-05-10T11:00:00Z", cloud=2.0)])

    with pytest.raises(ValueError) as raised:
        find(cat, date_range)
    assert date_range in str(raised.value)
    assert ITEM_DATE in str(raised.value)
    assert cat.searches() == []


def test_stage1_no_matching_item_fails_naming_collection_range_and_tile(catalogue):
    cat = catalogue("l2a")
    cat.answer([cat.candidate("other-tile", "2023-05-10T11:00:00Z", cloud=0.0, **{"grid:code": "MGRS-31UCT"})])

    with pytest.raises(Exception) as raised:
        find(cat)
    message = str(raised.value)
    assert cat.collection in message
    assert BASELINE in message
    assert "30UXC" in message
    assert "31UCT" not in message


def test_stage1_ard_no_matching_item_names_the_tile_from_the_item_id(catalogue):
    cat = catalogue("ard")
    other_tile = "S2B_20230520_latn518lonw0008_T31UCT_ORB137_20230520120350_utm30n_osgb"
    cat.answer(
        [cat.candidate(other_tile, "2023-05-20T11:00:00Z", cloud=0.0, **{"proj:transform": OTHER_ORBIT_TRANSFORM})]
    )

    with pytest.raises(Exception) as raised:
        find(cat)
    message = str(raised.value)
    assert cat.collection in message
    assert BASELINE in message
    assert "30UXC" in message
    assert "31UCT" not in message


@pytest.mark.parametrize("statuses", [[429], [503], [429, 502, 503]], ids=["429", "503", "429-502-503"])
def test_stage1_search_retries_429_and_5xx_with_growing_backoff(statuses, catalogue, sleeps):
    cat = catalogue("l2a")
    cat.answer([cat.candidate("clear", "2023-05-10T11:00:00Z", cloud=2.0)])
    cat.server.fail(cat.search_path, list(statuses))

    assert find(cat).id == "clear"
    assert len(cat.searches()) == len(statuses) + 1
    assert len(sleeps) == len(statuses)
    assert sleeps[0] > 0
    assert all(later > earlier for earlier, later in zip(sleeps, sleeps[1:])), sleeps


@pytest.mark.parametrize(
    "status, retry_after, expected", [(429, 2, 2), (503, 120, 30)], ids=["2s", "120s-capped-at-30s"]
)
def test_stage1_search_waits_as_long_as_retry_after_capped_at_30s(status, retry_after, expected, catalogue, sleeps):
    cat = catalogue("l2a")
    cat.answer([cat.candidate("clear", "2023-05-10T11:00:00Z", cloud=2.0)])
    cat.server.fail(cat.search_path, [status], retry_after=retry_after)

    assert find(cat).id == "clear"
    assert sum(sleeps) == pytest.approx(expected)


def test_stage1_search_that_keeps_failing_names_url_and_status(catalogue, sleeps):
    cat = catalogue("l2a")
    cat.answer([cat.candidate("clear", "2023-05-10T11:00:00Z", cloud=2.0)])
    cat.server.fail(cat.search_path, [503] * 50)

    with pytest.raises(Exception) as raised:
        find(cat)
    assert cat.search_url in str(raised.value)
    assert "503" in str(raised.value)
    assert 1 < len(cat.searches()) < 50


def test_stage1_search_times_out_naming_url(catalogue, sleeps):
    cat = catalogue("l2a")
    cat.answer([cat.candidate("clear", "2023-05-10T11:00:00Z", cloud=2.0)])
    cat.server.hang(cat.search_path)

    started = time.monotonic()
    with pytest.raises(Exception) as raised:
        find(cat, timeout=0.5)
    assert time.monotonic() - started < 20
    assert cat.search_url in str(raised.value)
    assert re.search(r"time", str(raised.value), re.IGNORECASE)
    assert 2 < len(cat.searches()) < 50
