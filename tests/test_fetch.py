"""
Fetch mode: the workflow reads the item from a URL (`--stac_item=<url>`) instead of a staged
directory. The staged golden fixtures are served from a local HTTP server, so the tests run offline.
"""

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pystac
import pytest

import main
import workflow
from eodh_workflow import StacItemInput, WorkflowInputs, output_problems
from golden_data import AOI_JSON, ITEM_IDS, item_json_path, read_output, run_ndvi


def run_fetched(url: str, output_dir: Path, **params):
    """Call workflow.run() as main.py does in fetch mode: no staged item, the URL in params."""
    params = {"workflow": "eodh-ndvi", "aoi": AOI_JSON, "stac_item": url, **params}
    output_dir.mkdir(parents=True, exist_ok=True)
    return workflow.run(WorkflowInputs(stac_item=None, params=params), output_dir)


def source_of(spec) -> dict:
    [item] = spec.items
    return item.source_item.item if isinstance(item.source_item, StacItemInput) else item.source_item


def assert_same_output(fetched, staged):
    data, transform, crs = read_output(fetched)
    expected, expected_transform, expected_crs = read_output(staged)
    assert crs == expected_crs
    assert transform == expected_transform
    assert np.array_equal(data, expected, equal_nan=True)
    assert np.isfinite(data).sum() > 40000


@pytest.mark.parametrize(
    "name, hrefs",
    [("l2a", "absolute"), ("ard", "absolute"), ("l2a", "relative"), ("l2a", "file-url")],
    ids=["l2a-absolute-http", "ard-absolute-http", "l2a-relative-to-http-item", "l2a-file-item-url"],
)
def test_fetched_item_gives_same_ndvi_and_source_item_as_staged(name, hrefs, request, serve, tmp_path):
    staged_dir = request.getfixturevalue(f"{name}_staged")
    staged_item = json.loads(item_json_path(staged_dir).read_text())
    if hrefs == "file-url":
        url = item_json_path(staged_dir).resolve().as_uri()
    else:
        server = serve(staged_dir)
        url = server.with_absolute_hrefs() if hrefs == "absolute" else server.item_url()

    fetched = run_fetched(url, tmp_path / "fetched")
    staged = run_ndvi(staged_dir, tmp_path / "staged")

    assert_same_output(fetched, staged)
    source = source_of(fetched)
    assert source["id"] == ITEM_IDS[name]
    assert source["collection"] == staged_item["collection"]
    assert source["properties"] == staged_item["properties"]
    if hrefs != "file-url":
        assert server.band_requests(), "the bands were not read from the item URL's server"


def test_main_runs_on_item_url_and_carries_source_metadata(l2a_staged, serve, tmp_path):
    url = serve(l2a_staged).with_absolute_hrefs()
    out = tmp_path / "out"
    main.main([f"--output_dir={out}", "--workflow=eodh-ndvi", f"--aoi={AOI_JSON}", f"--stac_item={url}"])

    assert output_problems(out) == []
    collection = next(pystac.Catalog.from_file(str(out / "catalog.json")).get_children())
    [item] = list(collection.get_items())
    source = json.loads(item_json_path(l2a_staged).read_text())["properties"]
    assert item.properties["eodh:source_item_id"] == ITEM_IDS["l2a"]
    assert item.properties["platform"] == source["platform"]
    assert item.properties["eo:cloud_cover"] == source["eo:cloud_cover"]


def test_fetch_in_common_returns_a_stac_item_input_the_product_runs_on(l2a_staged, serve, tmp_path):
    from common import fetch_item

    fetched_input = fetch_item(serve(l2a_staged).with_absolute_hrefs())

    assert isinstance(fetched_input, StacItemInput)
    assert fetched_input.id == ITEM_IDS["l2a"]
    params = {"workflow": "eodh-ndvi", "aoi": AOI_JSON}
    (tmp_path / "fetched").mkdir()
    fetched = workflow.run(WorkflowInputs(stac_item=fetched_input, params=params), tmp_path / "fetched")
    assert_same_output(fetched, run_ndvi(l2a_staged, tmp_path / "staged"))


@pytest.mark.parametrize("given", ["both", "neither"])
def test_both_or_neither_item_input_fails_naming_the_inputs(given, l2a_staged, serve, tmp_path):
    args = [f"--output_dir={tmp_path / 'out'}", "--workflow=eodh-ndvi", f"--aoi={AOI_JSON}"]
    if given == "both":
        args += [f"--stac_item_dir={l2a_staged}", f"--stac_item={serve(l2a_staged).with_absolute_hrefs()}"]

    with pytest.raises(Exception) as raised:
        main.main(args)
    message = str(raised.value)
    assert "stac_item_dir" in message
    assert re.search(r"stac_item(?!_dir)", message)


@pytest.mark.parametrize(
    "href",
    ["s3://some-bucket/T30UXC/B08.tif", "gs://some-bucket/T30UXC/B08.tif", "ftp://example.com/T30UXC/B08.tif"],
    ids=["s3", "gs", "ftp"],
)
def test_href_needing_credentials_fails_before_any_band_is_read(href, l2a_staged, serve, tmp_path):
    server = serve(l2a_staged)
    url = server.with_absolute_hrefs(nir=href)

    with pytest.raises(Exception, match=re.escape(href)):
        run_fetched(url, tmp_path / "out")
    assert server.band_requests() == []


@pytest.mark.parametrize(
    "statuses",
    [[429], [503], [429, 502], [503, 503, 503]],
    ids=["429", "503", "429-then-502", "503-three-times"],
)
def test_fetch_retries_429_and_5xx_with_growing_backoff(statuses, l2a_staged, serve, sleeps, tmp_path):
    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.fail(server.item_path(), list(statuses))

    fetched = run_fetched(url, tmp_path / "out")

    assert source_of(fetched)["id"] == ITEM_IDS["l2a"]
    assert server.requests.count(server.item_path()) == len(statuses) + 1
    assert len(sleeps) == len(statuses)
    assert sleeps[0] > 0
    assert all(later > earlier for earlier, later in zip(sleeps, sleeps[1:])), sleeps


def test_fetch_that_keeps_failing_names_url_and_status(l2a_staged, serve, sleeps, tmp_path):
    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.fail(server.item_path(), [503] * 50)

    with pytest.raises(Exception) as raised:
        run_fetched(url, tmp_path / "out")
    assert url in str(raised.value)
    assert "503" in str(raised.value)
    assert 1 < server.requests.count(server.item_path()) < 50


def test_fetch_times_out_naming_url(l2a_staged, serve, sleeps):
    from common import fetch_item

    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.hang(server.item_path())

    started = time.monotonic()
    with pytest.raises(Exception) as raised:
        fetch_item(url, timeout=0.5)
    assert time.monotonic() - started < 20
    assert url in str(raised.value)
    assert re.search(r"time", str(raised.value), re.IGNORECASE)
    assert 2 < server.requests.count(server.item_path()) < 50


def test_fetch_retries_a_timeout_then_reads_the_item(l2a_staged, serve, sleeps):
    from common import fetch_item

    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.hang(server.item_path(), times=1)

    fetched = fetch_item(url, timeout=0.5)

    assert fetched.id == ITEM_IDS["l2a"]
    assert server.requests.count(server.item_path()) == 2
    assert len(sleeps) >= 1


@pytest.mark.parametrize(
    "status, retry_after, expected",
    [(429, 2, 2), (503, 3, 3), (429, 120, 30)],
    ids=["429-waits-2s", "503-waits-3s", "120s-capped-at-30s"],
)
def test_fetch_waits_as_long_as_retry_after_capped_at_30s(status, retry_after, expected, l2a_staged, serve, sleeps):
    from common import fetch_item

    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.fail(server.item_path(), [status], retry_after=retry_after)

    assert fetch_item(url).id == ITEM_IDS["l2a"]
    assert sum(sleeps) == pytest.approx(expected)


@pytest.mark.parametrize("status, retry_after", [(429, "nan"), (503, "inf"), (429, "-inf")])
def test_fetch_with_non_finite_retry_after_backs_off_and_reads_item(status, retry_after, l2a_staged, serve, sleeps):
    from common import fetch_item

    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.fail(server.item_path(), [status], retry_after=retry_after)

    assert fetch_item(url).id == ITEM_IDS["l2a"]
    assert sleeps
    assert all(math.isfinite(seconds) and seconds > 0 for seconds in sleeps), sleeps


@pytest.mark.parametrize("failure", ["not-json", "dropped-connection"])
def test_fetch_of_a_bad_body_fails_naming_url(failure, l2a_staged, serve, sleeps):
    from common import fetch_item

    server = serve(l2a_staged)
    if failure == "not-json":
        url = server.serve_page(server.item_path(), "<html><body>Bad gateway</body></html>")
    else:
        url = server.with_absolute_hrefs()
        server.drop(server.item_path())

    with pytest.raises(Exception) as raised:
        fetch_item(url, timeout=5)
    assert url in str(raised.value)


def test_fetch_rejects_file_url_with_a_host_but_accepts_localhost(l2a_staged):
    from common import fetch_item

    path = item_json_path(l2a_staged).resolve().as_posix()

    url = f"file://example.com{path}"
    with pytest.raises(ValueError) as raised:
        fetch_item(url)
    assert url in str(raised.value)
    assert fetch_item(f"file://localhost{path}").id == ITEM_IDS["l2a"]
