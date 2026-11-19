"""Stage 3: the eodh-ndvi CWL and catalogue record, versions over every CWL, and the README."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CWL = ROOT / "cwl" / "eodh-ndvi.cwl"
RECORD = ROOT / "records" / "eodh-ndvi.json"


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
    assert inputs["stac_item"]["type"] == "Directory"
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
