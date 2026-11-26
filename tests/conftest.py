import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from eodh_workflow.testing import stage_item
from golden_data import copy_staged, item_json_path

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def input_item() -> dict:
    """One real item from the input collection. Its assets are replaced by synthetic rasters when staged.
    tests/helpers/ uses it."""
    return json.loads((FIXTURES / "input_item.json").read_text())


@pytest.fixture
def staged(tmp_path, input_item) -> Path:
    """A directory laid out the way ADES stage-in writes it, for `input_item`, with synthetic rasters."""
    return stage_item(input_item, tmp_path / "staged")


@pytest.fixture
def l2a_staged(tmp_path) -> Path:
    """A writable copy of the staged golden sentinel-2-c1-l2a item (real data over the Oxford AOI)."""
    return copy_staged("l2a", tmp_path / "l2a_staged")


@pytest.fixture
def ard_staged(tmp_path) -> Path:
    """A writable copy of the staged golden sentinel2_ard item (real data over the Oxford AOI)."""
    return copy_staged("ard", tmp_path / "ard_staged")


SERVER_SCRIPT = Path(__file__).parent / "fixture_server.py"


class FixtureServer:
    """A fixture_server.py process serving a staged directory, and the controls for it."""

    def __init__(self, root: Path, workdir: Path):
        self.root = root
        self.control_path = workdir / "control.json"
        self.log_path = workdir / "requests.log"
        self.control = {"pages": {}, "failures": {}, "retry_after": {}, "hang": [], "hang_first": {}, "drop": []}
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

    def fail(self, path: str, statuses: list, retry_after=None):
        """Answer the first requests for `path` with `statuses` (with a Retry-After header if given), then serve it."""
        self.control["failures"][path] = statuses
        if retry_after is not None:
            self.control["retry_after"][path] = retry_after
        self._save()

    def hang(self, path: str, times=None):
        """Never answer requests for `path`, or only the first `times` of them."""
        if times is None:
            self.control["hang"].append(path)
        else:
            self.control["hang_first"][path] = times
        self._save()

    def serve_page(self, path: str, body: str) -> str:
        """Serve `body` at `path`; return its URL."""
        self.control["pages"][path] = body
        self._save()
        return self.base + path

    def drop(self, path: str):
        """Close the connection part way through every response body for `path`."""
        self.control["drop"].append(path)
        self._save()

    @property
    def requests(self) -> list:
        return self.log_path.read_text().splitlines() if self.log_path.exists() else []

    @property
    def requests_with_query(self) -> list:
        full = self.log_path.with_name(self.log_path.name + ".full")
        return full.read_text().splitlines() if full.exists() else []

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
