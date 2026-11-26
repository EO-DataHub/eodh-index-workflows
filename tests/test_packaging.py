"""The eodh-ndvi CWL and catalogue record, versions over every CWL, the job fixture and the README."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

from golden_data import AOI, ITEM_IDS

ROOT = Path(__file__).resolve().parent.parent
CWL = ROOT / "cwl" / "eodh-ndvi.cwl"
RECORD = ROOT / "records" / "eodh-ndvi.json"
JOB = ROOT / "cwl" / "eodh-ndvi.job.json"


def _graph(cwl_path: Path) -> dict:
    graph = yaml.safe_load(cwl_path.read_text())["$graph"]
    return {node["class"]: node for node in graph}, graph


def _names(section) -> dict:
    """CWL inputs/requirements as a dict, whichever of the list or map forms is used."""
    if isinstance(section, dict):
        return section
    return {entry.get("id", entry.get("class")): entry for entry in section}


def test_stage3_cwl_runs_the_ndvi_product_from_the_image():
    nodes, graph = _graph(CWL)
    assert sorted(node["class"] for node in graph) == ["CommandLineTool", "Workflow"]
    workflow, tool = nodes["Workflow"], nodes["CommandLineTool"]
    assert workflow["id"] == "eodh-ndvi"

    inputs = _names(workflow["inputs"])
    assert set(inputs) == {"stac_item", "aoi"}
    assert inputs["aoi"]["type"] == "string"

    assert tool["arguments"] == ["--workflow=eodh-ndvi"]
    requirements = {**_names(tool.get("requirements", [])), **_names(tool.get("hints", []))}
    network = requirements["NetworkAccess"]
    assert network.get("networkAccess") is True
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
    assert requirements["DockerRequirement"]["dockerPull"] == f"ghcr.io/eo-datahub/eodh-index-workflows:{version}"


def test_stage3_template_cwl_is_removed():
    assert not (ROOT / "cwl" / "workflow.cwl").exists()
    assert "cwl/workflow.cwl" not in (ROOT / "Makefile").read_text()


def test_stage3_record_offers_ndvi_for_both_collections_with_item_and_aoi_inputs():
    record = json.loads(RECORD.read_text())
    properties = record["properties"]
    assert record["id"] == "eodh-ndvi"
    assert sorted(properties["applicableCollections"]) == ["sentinel-2-c1-l2a", "sentinel2_ard"]
    inputs = properties["inputParameters"]
    assert inputs["stac_item"]["hubType"] == "stac-item"
    assert inputs["aoi"]["hubType"] == "aoi"
    links = {link["rel"]: link["href"] for link in record["links"]}
    assert links["application"].endswith("eodh-ndvi.cwl")
    assert "eo-datahub/eodh-index-workflows" in links["vcs"].lower()


def _second_cwl(text: str) -> str:
    """A copy of the NDVI CWL for a made-up second product."""
    return text.replace("eodh-ndvi", "eodh-other")


@pytest.fixture
def repo_copy(tmp_path) -> Path:
    """The parts of the repo that scripts/versions.py reads, with a second product's CWL added."""
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(ROOT / "cwl", tmp_path / "cwl")
    (tmp_path / "cwl" / "eodh-other.cwl").write_text(_second_cwl(CWL.read_text()))
    return tmp_path


def _versions(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "versions.py"), *args], capture_output=True, text=True, cwd=repo
    )


def test_stage3_versions_check_and_bump_cover_every_cwl(repo_copy):
    shown = _versions(repo_copy, "check")
    assert shown.returncode == 0, shown.stderr
    assert "eodh-ndvi.cwl" in shown.stdout and "eodh-other.cwl" in shown.stdout

    other = repo_copy / "cwl" / "eodh-other.cwl"
    other.write_text(re.sub(r"s:softwareVersion: *\S+", "s:softwareVersion: 9.9.9", other.read_text()))
    assert _versions(repo_copy, "check").returncode != 0

    bumped = _versions(repo_copy, "bump", "0.4.2")
    assert bumped.returncode == 0, bumped.stderr
    for cwl in (repo_copy / "cwl").glob("*.cwl"):
        text = cwl.read_text()
        assert "s:softwareVersion: 0.4.2" in text, cwl.name
        assert "eodh-index-workflows:0.4.2" in text, cwl.name


def test_stage3_readme_lists_products_and_how_to_add_one():
    readme = (ROOT / "README.md").read_text()
    assert "eodh-ndvi" in readme
    assert re.search(r"^#+ .*add.*product", readme, re.IGNORECASE | re.MULTILINE)
    for part in ("src/products/", "cwl/", "records/"):
        assert part in readme, part


def _all_types(node) -> list:
    """Every `type` value of every input in a CWL process, whichever form the inputs use."""
    return [entry if isinstance(entry, str) else entry.get("type") for entry in _names(node["inputs"]).values()]


def test_cwl_takes_the_item_url_as_a_string_passed_as_stac_item():
    nodes, _ = _graph(CWL)
    workflow, tool = nodes["Workflow"], nodes["CommandLineTool"]
    assert _names(workflow["inputs"])["stac_item"]["type"] == "string"

    bound = [
        entry
        for entry in _names(tool["inputs"]).values()
        if isinstance(entry, dict) and (entry.get("inputBinding") or {}).get("prefix") == "--stac_item="
    ]
    assert len(bound) == 1
    assert bound[0]["type"] == "string"
    assert bound[0]["inputBinding"].get("separate") is False

    for node in (workflow, tool):
        assert not [t for t in _all_types(node) if "Directory" in str(t)], node["id"]
    assert "stac_item_dir" not in CWL.read_text()
    requirements = {**_names(tool.get("requirements", [])), **_names(tool.get("hints", []))}
    assert requirements["NetworkAccess"].get("networkAccess") is True


def test_job_fixture_gives_the_hub_l2a_item_url_and_the_fixture_aoi():
    job = json.loads(JOB.read_text())
    assert set(job) == {"stac_item", "aoi"}
    url = job["stac_item"]
    assert isinstance(url, str)
    host = urlparse(url).hostname
    assert url.startswith("https://") and (
        host == "eodatahub.org.uk" or (host is not None and host.endswith(".eodatahub.org.uk"))
    )
    assert "sentinel-2-c1-l2a" in url
    assert url.rstrip("/").endswith(ITEM_IDS["l2a"])
    assert json.loads(job["aoi"]) == AOI


def test_record_is_unchanged_and_keeps_raster_stac_item():
    record = json.loads(RECORD.read_text())
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    assert digest == "6d879ffab7aed11715f809b51fa5b6aae68a830e8111eecb68aa20667b621b0d"
    assert record["properties"]["inputParameters"]["stac_item"]["type"] == "raster"


def test_stage1_version_is_0_4_0_in_pyproject_and_every_cwl():
    assert re.search(r'^version = "0\.4\.0"$', (ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    cwls = sorted((ROOT / "cwl").glob("*.cwl"))
    assert [cwl.stem for cwl in cwls] == ["eodh-evi", "eodh-nbr", "eodh-ndvi", "eodh-ndwi", "eodh-savi"]
    for cwl in cwls:
        text = cwl.read_text()
        assert yaml.safe_load(text)["s:softwareVersion"] == "0.4.0", cwl.name
        assert re.findall(r"eodh-index-workflows:(\S+)", text) == ["0.4.0"], cwl.name


def test_stage1_readme_product_table_lists_ndwi():
    [row] = [line for line in (ROOT / "README.md").read_text().splitlines() if line.startswith("| `eodh-ndwi`")]
    assert re.search(r"ndwi", row, re.IGNORECASE) and "rdbu" in row
    assert "cwl/eodh-ndwi.cwl" in row and "records/eodh-ndwi.json" in row
