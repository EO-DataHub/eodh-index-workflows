"""
Fetch mode: the workflow reads the item from a URL (`--stac_item=<url>`) instead of a staged
directory. The staged golden fixtures are served from a local HTTP server, so the tests run offline.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pystac
import pytest

import main
import workflow
from eodh_workflow import StacItemInput, WorkflowInputs, output_problems
from golden_data import AOI_JSON, ITEM_IDS, item_json_path, read_output, run_ndvi

SERVER_SCRIPT = Path(__file__).parent / "fixture_server.py"


class FixtureServer:
    """A fixture_server.py process serving a staged directory, and the controls for it."""

    def __init__(self, root: Path, workdir: Path):
        self.root = root
        self.control_path = workdir / "control.json"
        self.log_path = workdir / "requests.log"
        self.control = {"pages": {}, "failures": {}, "hang": []}
        self._save()
        self.process = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT), str(root), str(self.control_path), str(self.log_path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.base = f"http://127.0.0.1:{int(self.process.stdout.readline())}"

    def _save(self):
        self.control_path.write_text(json.dumps(self.control))

    def stop(self):
        self.process.kill()
        self.process.wait()

    def item_path(self) -> str:
        """URL path of the staged item JSON."""
        return "/" + item_json_path(self.root).relative_to(self.root).as_posix()

    def item_url(self) -> str:
        """URL of the staged item as it is, with relative asset hrefs."""
        return self.base + self.item_path()

    def with_absolute_hrefs(self, **overrides) -> str:
        """Serve the item with absolute http:// asset hrefs (or `overrides[key]`); return its URL."""
        path = self.item_path()
        item = json.loads(item_json_path(self.root).read_text())
        folder = path.rsplit("/", 1)[0]
        for key, asset in item["assets"].items():
            asset["href"] = overrides.get(key, f"{self.base}{folder}/{asset['href'].removeprefix('./')}")
        self.control["pages"][path] = json.dumps(item)
        self._save()
        return self.base + path

    def fail(self, path: str, statuses: list):
        """Answer the first requests for `path` with `statuses`, then serve it."""
        self.control["failures"][path] = statuses
        self._save()

    def hang(self, path: str):
        self.control["hang"].append(path)
        self._save()

    @property
    def requests(self) -> list:
        return self.log_path.read_text().splitlines() if self.log_path.exists() else []

    def band_requests(self) -> list:
        return [path for path in self.requests if path.lower().endswith((".tif", ".tiff"))]


@pytest.fixture
def serve(monkeypatch, tmp_path_factory):
    """Start a FixtureServer on a staged directory: `serve(staged_dir)`. Stopped after the test."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    servers = []

    def start(root: Path) -> FixtureServer:
        server = FixtureServer(root, tmp_path_factory.mktemp("server"))
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


@pytest.fixture
def sleeps(monkeypatch):
    """Delays passed to time.sleep, which returns at once."""
    calls = []
    monkeypatch.setattr(time, "sleep", lambda seconds: calls.append(seconds))
    return calls


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
def test_stage1_fetched_item_gives_same_ndvi_and_source_item_as_staged(name, hrefs, request, serve, tmp_path):
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


def test_stage1_main_runs_on_item_url_and_carries_source_metadata(l2a_staged, serve, tmp_path):
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


def test_stage1_fetch_in_common_returns_a_stac_item_input_the_product_runs_on(l2a_staged, serve, tmp_path):
    from common import fetch_item

    fetched_input = fetch_item(serve(l2a_staged).with_absolute_hrefs())

    assert isinstance(fetched_input, StacItemInput)
    assert fetched_input.id == ITEM_IDS["l2a"]
    params = {"workflow": "eodh-ndvi", "aoi": AOI_JSON}
    (tmp_path / "fetched").mkdir()
    fetched = workflow.run(WorkflowInputs(stac_item=fetched_input, params=params), tmp_path / "fetched")
    assert_same_output(fetched, run_ndvi(l2a_staged, tmp_path / "staged"))


@pytest.mark.parametrize("given", ["both", "neither"])
def test_stage1_both_or_neither_item_input_fails_naming_the_inputs(given, l2a_staged, serve, tmp_path):
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
def test_stage1_href_needing_credentials_fails_before_any_band_is_read(href, l2a_staged, serve, tmp_path):
    server = serve(l2a_staged)
    url = server.with_absolute_hrefs(nir=href)

    with pytest.raises(Exception, match=re.escape(href)):
        run_fetched(url, tmp_path / "out")
    assert server.band_requests() == []


@pytest.mark.parametrize("statuses", [[429], [503], [429, 502]], ids=["429", "503", "429-then-502"])
def test_stage1_fetch_retries_429_and_5xx_with_backoff(statuses, l2a_staged, serve, sleeps, tmp_path):
    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.fail(server.item_path(), list(statuses))

    fetched = run_fetched(url, tmp_path / "out")

    assert source_of(fetched)["id"] == ITEM_IDS["l2a"]
    assert server.requests.count(server.item_path()) == len(statuses) + 1
    assert len(sleeps) >= len(statuses)
    assert all(delay > 0 for delay in sleeps)


def test_stage1_fetch_that_keeps_failing_names_url_and_status(l2a_staged, serve, sleeps, tmp_path):
    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.fail(server.item_path(), [503] * 50)

    with pytest.raises(Exception) as raised:
        run_fetched(url, tmp_path / "out")
    assert url in str(raised.value)
    assert "503" in str(raised.value)
    assert 1 < server.requests.count(server.item_path()) < 50


def test_stage1_fetch_times_out_naming_url(l2a_staged, serve, sleeps):
    from common import fetch_item

    server = serve(l2a_staged)
    url = server.with_absolute_hrefs()
    server.hang(server.item_path())

    started = time.monotonic()
    with pytest.raises(Exception) as raised:
        fetch_item(url, timeout=0.5)
    assert time.monotonic() - started < 10
    assert url in str(raised.value)
    assert re.search(r"time", str(raised.value), re.IGNORECASE)
