"""
eodh-dnbr: NBR(baseline) - NBR(input) over the AOI, and its burn-severity classes.

A local fixture server plays the input item's catalogue. It serves the staged golden L2A item as the
input, and a baseline item on the same tile whose swir22 band is changed, so dNBR has a spread of values.
"""

import importlib
import inspect
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
import yaml

import main
import workflow
from eodh_workflow import output_problems
from golden_data import AOI_JSON, copy_staged, item_json_path, polygon_mask, read_output, run_product

ROOT = Path(workflow.__file__).resolve().parent.parent
NAME = "eodh-dnbr"
BASELINE = "2023-05-01/2023-05-31"
BASELINE_ID = "S2A_T30UXC_20230510T112121_L2A"
ITEM_DATE = date(2023, 6, 8)
DIVERGING = {
    "brbg", "bwr", "coolwarm", "piyg", "prgn", "puor", "rdbu", "rdgy", "rdylbu", "rdylgn", "seismic", "spectral",
}  # fmt: skip


def _scale_swir22(path: Path):
    """Rewrite swir22 so its reflectance is multiplied by a factor that runs from 2.0 to 0.15 across the columns."""
    with rasterio.open(path) as src:
        profile = {**src.profile, "driver": "GTiff"}
        raw = src.read(1)
    scale, offset = 0.0001, -0.1
    reflectance = raw.astype("float64") * scale + offset
    factor = np.linspace(2.0, 0.15, raw.shape[1])[np.newaxis, :]
    changed = np.clip(np.round((reflectance * factor - offset) / scale), 1, 65535).astype(raw.dtype)
    changed = np.where((raw == 0) | (reflectance <= 0), raw, changed)
    path.unlink()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(changed, 1)


@dataclass
class DnbrCatalogue:
    input_staged: Path
    baseline_staged: Path
    input_url: str
    baseline_url: str
    output_dir: Path

    def run(self) -> Path:
        """Run eodh-dnbr through main.py on the input item URL with the baseline range; return the output dir."""
        main.main(
            [
                f"--output_dir={self.output_dir}",
                f"--workflow={NAME}",
                f"--stac_item={self.input_url}",
                f"--aoi={AOI_JSON}",
                f"--baseline={BASELINE}",
            ]
        )
        return self.output_dir


@pytest.fixture
def dnbr_catalogue(serve, tmp_path) -> DnbrCatalogue:
    """The input item and a baseline item on the same tile, in a catalogue on the fixture server."""
    www = tmp_path / "www"
    input_staged = copy_staged("l2a", www / "input")
    baseline_staged = copy_staged("l2a", www / "baseline")
    _scale_swir22(item_json_path(baseline_staged).parent / "swir22.tif")
    server = serve(www)

    template = json.loads(item_json_path(input_staged).read_text())
    collection = template["collection"]
    collection_url = f"{server.base}/catalogue/collections/{collection}"

    def publish(staged: Path, item_id: str, when: str, cloud: float) -> tuple:
        item = json.loads(json.dumps(template))
        item["id"] = item_id
        item["properties"]["datetime"] = when
        item["properties"]["eo:cloud_cover"] = cloud
        folder = item_json_path(staged).parent.relative_to(www).as_posix()
        for key, asset in item["assets"].items():
            asset["href"] = f"{server.base}/{folder}/{asset['href'].removeprefix('./')}"
        path = f"/catalogue/collections/{collection}/items/{item_id}"
        item["links"] = [
            {"rel": "self", "href": server.base + path},
            {"rel": "collection", "href": collection_url},
        ]
        return server.serve_page(path, json.dumps(item)), item

    input_url, _ = publish(input_staged, template["id"], template["properties"]["datetime"], 0.08)
    baseline_url, baseline_item = publish(baseline_staged, BASELINE_ID, "2023-05-10T11:26:49.000000Z", 1.5)
    page = {"type": "FeatureCollection", "features": [baseline_item], "links": []}
    server.serve_page("/catalogue/search", json.dumps(page))

    return DnbrCatalogue(input_staged, baseline_staged, input_url, baseline_url, tmp_path / "out")


def _output_item(output_dir: Path) -> pystac.Item:
    collection = next(pystac.Catalog.from_file(str(output_dir / "catalog.json")).get_children())
    [item] = list(collection.get_items())
    return item


def _rasters(item: pystac.Item) -> dict:
    """{"dnbr" | "severity": (asset key, array, nodata, transform, crs)}: the float asset and the integer asset."""
    found = {}
    for key, asset in item.assets.items():
        with rasterio.open(asset.get_absolute_href()) as src:
            kind = "dnbr" if np.issubdtype(np.dtype(src.dtypes[0]), np.floating) else "severity"
            assert kind not in found, f"two {kind} rasters: {found[kind][0]} and {key}"
            found[kind] = (key, src.read(1), src.nodata, src.transform, src.crs)
    assert set(found) == {"dnbr", "severity"}
    return found


def _render_for(item: pystac.Item, asset_key: str) -> dict:
    [render] = [render for render in item.properties["renders"].values() if render["assets"] == [asset_key]]
    return render


def test_stage2_dnbr_is_baseline_nbr_minus_input_nbr_on_the_input_nir_grid(dnbr_catalogue, tmp_path):
    cat = dnbr_catalogue
    before, before_transform, before_crs = read_output(run_product("eodh-nbr", cat.baseline_staged, tmp_path / "b"))
    after, after_transform, after_crs = read_output(run_product("eodh-nbr", cat.input_staged, tmp_path / "a"))
    assert (before_transform, before_crs) == (after_transform, after_crs)
    expected = before.astype("float64") - after.astype("float64")

    _, dnbr, _, transform, crs = _rasters(_output_item(cat.run()))["dnbr"]
    assert (transform, crs, dnbr.shape) == (after_transform, after_crs, after.shape)
    inside = polygon_mask(dnbr.shape, transform, crs)
    assert inside.sum() > 40000
    assert np.nanmax(expected[inside]) - np.nanmin(expected[inside]) > 0.5, "the baseline fixture gives no spread"
    np.testing.assert_allclose(dnbr[inside], expected[inside], rtol=1e-5, atol=1e-6, equal_nan=True)
    assert np.isnan(dnbr[~inside]).all()


def test_stage2_severity_classes_rise_with_dnbr_with_usgs_bounds_in_the_module(dnbr_catalogue):
    rasters = _rasters(_output_item(dnbr_catalogue.run()))
    _, dnbr, _, _, _ = rasters["dnbr"]
    _, severity, nodata, _, _ = rasters["severity"]
    assert np.issubdtype(severity.dtype, np.integer)
    assert nodata is not None

    finite = np.isfinite(dnbr)
    assert (severity[~finite] == nodata).all()
    assert (severity[finite] != nodata).all()
    order = np.argsort(dnbr[finite], kind="stable")
    assert (np.diff(severity[finite][order].astype("int64")) >= 0).all(), "a higher dNBR got a lower class"
    assert len(np.unique(severity[finite])) >= 3

    module = importlib.import_module("products.dnbr")
    assert module.NAME == NAME
    assert re.search(r"usgs", inspect.getsource(module), re.IGNORECASE)


def test_stage2_output_names_the_baseline_item_and_renders_dnbr_and_severity(dnbr_catalogue):
    output_dir = dnbr_catalogue.run()
    assert output_problems(output_dir) == []
    item = _output_item(output_dir)

    values = [value for value in item.properties.values() if isinstance(value, str)]
    assert BASELINE_ID in values
    assert dnbr_catalogue.baseline_url in values
    derived = {link.get_absolute_href() for link in item.links if link.rel == "derived_from"}
    assert derived == {dnbr_catalogue.input_url, dnbr_catalogue.baseline_url}

    rasters = _rasters(item)
    dnbr_render = _render_for(item, rasters["dnbr"][0])
    [[low, high]] = dnbr_render["rescale"]
    assert high > 0 and low == -high
    assert dnbr_render["colormap_name"].removesuffix("_r") in DIVERGING

    severity_key, severity, nodata, _, _ = rasters["severity"]
    colormap = _render_for(item, severity_key)["colormap"]
    colormap = json.loads(colormap) if isinstance(colormap, str) else colormap
    classes = {int(value) for value in np.unique(severity) if value != nodata}
    assert classes <= {int(key) for key in colormap}


def _cwl_nodes(name: str) -> dict:
    return {node["class"]: node for node in yaml.safe_load((ROOT / "cwl" / f"{name}.cwl").read_text())["$graph"]}


def _valid_baseline(value: str):
    start, end = (date.fromisoformat(part) for part in value.split("/"))
    assert start < end < ITEM_DATE, value


def test_stage2_dnbr_cwl_job_examples_and_record_follow_ndvi_with_a_baseline_date_range():
    ndvi, cwl = _cwl_nodes("eodh-ndvi"), _cwl_nodes(NAME)
    assert cwl["Workflow"]["id"] == NAME
    assert cwl["CommandLineTool"]["arguments"] == [f"--workflow={NAME}"]
    assert set(cwl["Workflow"]["inputs"]) == set(ndvi["Workflow"]["inputs"]) | {"baseline"}
    assert cwl["Workflow"]["inputs"]["baseline"]["type"] == "string"
    assert cwl["Workflow"]["steps"]["main"]["in"]["baseline"] == "baseline"
    tool_inputs = dict(cwl["CommandLineTool"]["inputs"])
    assert tool_inputs.pop("baseline") == {
        "type": "string",
        "inputBinding": {"prefix": "--baseline=", "separate": False},
    }
    assert tool_inputs == ndvi["CommandLineTool"]["inputs"]
    for key in ("requirements", "hints", "baseCommand", "outputs"):
        assert cwl["CommandLineTool"][key] == ndvi["CommandLineTool"][key], key

    for path in ("cwl/{}.job.json", "examples/{}.l2a.execute.json", "examples/{}.ard.execute.json"):
        ours = json.loads((ROOT / path.format(NAME)).read_text())
        theirs = json.loads((ROOT / path.format("eodh-ndvi")).read_text())
        _valid_baseline(ours.get("inputs", ours).pop("baseline"))
        assert ours == theirs, path

    ndvi_record = json.loads((ROOT / "records" / "eodh-ndvi.json").read_text())
    record = json.loads((ROOT / "records" / f"{NAME}.json").read_text())
    assert record["id"] == NAME
    properties = record["properties"]
    assert properties["applicableCollections"] == ndvi_record["properties"]["applicableCollections"]
    hub_types = {key: value["hubType"] for key, value in properties["inputParameters"].items()}
    assert hub_types == {"stac_item": "stac-item", "aoi": "aoi", "baseline": "date-range"}
    assert re.search(r"dnbr", properties["title"] + properties["description"], re.IGNORECASE)
    links = {link["rel"]: link["href"] for link in record["links"]}
    assert links["application"].endswith(f"/cwl/{NAME}.cwl")
    assert links["self"].endswith(f"/items/{NAME}")
    assert links["vcs"] == {link["rel"]: link["href"] for link in ndvi_record["links"]}["vcs"]


def test_stage2_readme_lists_dnbr_and_explains_the_baseline():
    readme = (ROOT / "README.md").read_text()
    [row] = [line for line in readme.splitlines() if line.startswith(f"| `{NAME}`")]
    assert f"cwl/{NAME}.cwl" in row and f"records/{NAME}.json" in row
    assert "`baseline`" in readme
    for words in (r"date range", r"same tile", r"cloud", r"closest"):
        assert re.search(words, readme, re.IGNORECASE), words


def test_stage2_version_is_bumped_past_the_nbr_release():
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
    assert tuple(int(part) for part in version.split(".")) > (0, 3, 0)
