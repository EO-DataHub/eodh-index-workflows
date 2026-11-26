"""The single-index products (NDVI, EVI, SAVI, NBR, NDWI) on the shared index function, and bands on mixed grids."""

import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
import yaml
from affine import Affine

import main
import workflow
from eodh_workflow import Band, output_problems
from golden_data import (
    AOI,
    AOI_JSON,
    ITEM_IDS,
    item_json_path,
    on_grid,
    polygon_mask,
    read_golden,
    read_output,
    run_ndvi,
    run_product,
)

ROOT = Path(workflow.__file__).resolve().parent.parent
SHARED_FUNCTION = "run_index_product"

# sha256 of the NDVI output pixels (float32 bytes) per fixture, recorded on the code before this change.
NDVI_PIXELS_SHA256 = {
    "l2a": "7e7398a20945d322ef1387b9520b0b1779f89c45e83fd3635925bb935ec6fd16",
    "ard": "3a5a7e87eaec1b1b3a1fd55408a37dad8efbb42e6cc01ca581494ef0b5fced78",
}
NDVI_GRID = {
    "l2a": ((10.0, 0.0, 614230.0, 0.0, -10.0, 5734920.0), "EPSG:32630", (206, 213)),
    "ard": ((10.0, 0.0, 445310.0, 0.0, -10.0, 206260.0), "EPSG:27700", (203, 210)),
}
# sha256 of cwl/eodh-ndvi.cwl with its version replaced by VERSION, recorded before this change.
NDVI_CWL_SHA256 = "fdbc8fe11706babbe8e0ed8ad4cfab25db405034245a204652c10115a4f853f9"


# sha256 of each earlier product's CWL with its version replaced by VERSION, and of its record, recorded at 0.3.0.
EARLIER_CWL_SHA256 = {
    "evi": "f880e0780429cdc091bee898670c3af449f5ae3bb75b833494b1e0ccb77289d1",
    "savi": "75e7c2dae614711b32f04fea28b44baf5a42d9e321db8acdcaf90cd6d39cc135",
    "nbr": "48313630b0989ddce8634fe4b342ee85d7b32450c4e8bb7d89f15067dc2ca3a4",
}
EARLIER_RECORD_SHA256 = {
    "evi": "08817828c5d27801dccb8a6d03eeeb5ac2985b54c2285f536ec3f7f1ab3291e8",
    "savi": "fb11aff06c7386c0abe2a4fe0049bab5accd67bfdece78e7e255398d8f5b2602",
    "nbr": "ddc4dfb94274cc8eba3272db47126fd177ce580dfaa1d4a53a50f2a4fcf66ebf",
}


def _without_version(text: str) -> str:
    text = re.sub(r"(s:softwareVersion: *)\S+", r"\1VERSION", text)
    return re.sub(r"(eodh-index-workflows:)\S+", r"\1VERSION", text)


def _cwl_nodes(path: Path) -> dict:
    return {node["class"]: node for node in yaml.safe_load(path.read_text())["$graph"]}


def _staged_bands(staged: Path) -> dict:
    item_dir = item_json_path(staged).parent
    return {key: item_dir / f"{key}.tif" for key in ("nir", "swir22")}


@pytest.mark.parametrize("name", ["l2a", "ard"])
def test_ndvi_output_is_byte_for_byte_unchanged(name, request, tmp_path):
    spec = run_ndvi(request.getfixturevalue(f"{name}_staged"), tmp_path / "out")
    with rasterio.open(spec.items[0].assets["data"].path) as src:
        data = src.read(1)
        grid = (tuple(src.transform)[:6], src.crs.to_string(), data.shape)
    assert data.dtype == np.float32
    assert grid == NDVI_GRID[name]
    assert hashlib.sha256(data.tobytes()).hexdigest() == NDVI_PIXELS_SHA256[name]


def test_ndvi_cwl_is_unchanged_apart_from_the_version():
    text = _without_version((ROOT / "cwl" / "eodh-ndvi.cwl").read_text())
    assert hashlib.sha256(text.encode()).hexdigest() == NDVI_CWL_SHA256


@pytest.mark.parametrize("product", ["ndvi", "evi", "savi", "nbr", "ndwi"])
def test_product_module_leaves_the_shared_steps_to_the_shared_function(product):
    import common

    assert callable(getattr(common, SHARED_FUNCTION))
    module = importlib.import_module(f"products.{product}")
    assert module.NAME == f"eodh-{product}"
    source = inspect.getsource(module)
    assert SHARED_FUNCTION in source
    for shared_step in ("item_input(", "brightness(", "write_aoi_cog(", "OutputSpec(", "parse_aoi("):
        assert shared_step not in source, f"{product}.py calls {shared_step}"


@pytest.mark.parametrize("product", ["evi", "savi", "nbr"])
def test_earlier_product_cwl_and_record_are_unchanged_apart_from_the_version(product):
    cwl = _without_version((ROOT / "cwl" / f"eodh-{product}.cwl").read_text())
    assert hashlib.sha256(cwl.encode()).hexdigest() == EARLIER_CWL_SHA256[product]
    record = json.loads((ROOT / "records" / f"eodh-{product}.json").read_text())
    assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == EARLIER_RECORD_SHA256[product]


NIR = np.array([0.5, 0.3, 0.05])
RED = np.array([0.1, 0.2, 0.04])
BLUE = np.array([0.05, 0.1, 0.03])
SWIR22 = np.array([0.2, 0.25, 0.01])
GREEN = np.array([0.08, 0.1, 0.06])


@pytest.mark.parametrize(
    "product, bands, expected",
    [
        pytest.param(
            "evi",
            {"nir": NIR, "red": RED, "blue": BLUE},
            2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0),
            id="evi-eopro-coefficients",
        ),
        pytest.param("savi", {"nir": NIR, "red": RED}, (NIR - RED) / ((NIR + RED + 0.5) * (1 + 0.5)), id="savi-l-half"),
        pytest.param("nbr", {"nir": NIR, "swir22": SWIR22}, (NIR - SWIR22) / (NIR + SWIR22), id="nbr-nir-swir22"),
    ],
)
def test_index_formula_matches_eopro(product, bands, expected):
    module = importlib.import_module(f"products.{product}")
    np.testing.assert_allclose(getattr(module, product)(**bands), expected, rtol=1e-6)


def test_ndwi_formula_is_green_minus_nir_over_their_sum_and_high_for_water():
    from products import ndwi

    np.testing.assert_allclose(ndwi.ndwi(green=GREEN, nir=NIR), (GREEN - NIR) / (GREEN + NIR), rtol=1e-6)
    water, vegetation = ndwi.ndwi(green=np.array([0.08, 0.05]), nir=np.array([0.02, 0.4]))
    assert water > 0 > vegetation


def test_evi_writes_eopro_coefficients_without_xarray_spatial():
    from products import evi

    source = inspect.getsource(evi)
    assert "xrspatial" not in source and "xarray_spatial" not in source
    for value in ("2.5", "6", "7.5", "1"):
        assert re.search(rf"\b{re.escape(value)}\b", source), value


@pytest.mark.parametrize("name", ["l2a", "ard"])
@pytest.mark.parametrize("product", ["evi", "savi", "nbr", "ndwi"])
def test_index_matches_eopro_golden_inside_aoi_on_the_10m_grid(product, name, request, tmp_path):
    golden, golden_transform, golden_crs = read_golden(name, product)
    staged = request.getfixturevalue(f"{name}_staged")
    data, transform, crs = read_output(run_product(f"eodh-{product}", staged, tmp_path / "out"))
    assert (transform.a, transform.e) == (10.0, -10.0)
    assert crs.to_string() == golden_crs
    ours = on_grid(data, transform, golden.shape, golden_transform)
    inside = polygon_mask(golden.shape, golden_transform, golden_crs)
    assert inside.sum() > 40000
    assert np.allclose(ours[inside], golden[inside], equal_nan=True)


@pytest.mark.parametrize(
    "product, colormap",
    [
        pytest.param("evi", "ylgn", id="evi-ylgn"),
        pytest.param("savi", "ylgn", id="savi-ylgn"),
        pytest.param("nbr", "rdylgn", id="nbr-rdylgn"),
        pytest.param("ndwi", "rdbu", id="ndwi-rdbu"),
    ],
)
def test_index_output_is_valid_and_renders_minus_one_to_one(product, colormap, l2a_staged, tmp_path):
    output_dir = tmp_path / "out"
    main.main(
        [
            f"--stac_item_dir={l2a_staged}",
            f"--output_dir={output_dir}",
            f"--workflow=eodh-{product}",
            f"--aoi={AOI_JSON}",
        ]
    )
    assert output_problems(output_dir) == []
    collection = next(pystac.Catalog.from_file(str(output_dir / "catalog.json")).get_children())
    [item] = list(collection.get_items())
    assert item.id == f"{ITEM_IDS['l2a']}_{product}"
    assert item.properties["renders"] == collection.extra_fields["renders"]
    [render] = item.properties["renders"].values()
    assert render["assets"] == ["data"]
    assert render["rescale"] == [[-1, 1]]
    assert render["colormap_name"] == colormap


@pytest.mark.parametrize("product", ["evi", "savi", "nbr", "ndwi"])
def test_index_is_packaged_like_ndvi(product):
    name = f"eodh-{product}"
    ndvi, cwl = _cwl_nodes(ROOT / "cwl" / "eodh-ndvi.cwl"), _cwl_nodes(ROOT / "cwl" / f"{name}.cwl")
    assert cwl["Workflow"]["id"] == name
    assert cwl["CommandLineTool"]["arguments"] == [f"--workflow={name}"]
    assert set(cwl["Workflow"]["inputs"]) == set(ndvi["Workflow"]["inputs"])
    for key in ("inputs", "requirements", "hints", "baseCommand", "outputs"):
        assert cwl["CommandLineTool"][key] == ndvi["CommandLineTool"][key], key
    ndvi_version = yaml.safe_load((ROOT / "cwl" / "eodh-ndvi.cwl").read_text())["s:softwareVersion"]
    assert yaml.safe_load((ROOT / "cwl" / f"{name}.cwl").read_text())["s:softwareVersion"] == ndvi_version

    for path in ("cwl/{}.job.json", "examples/{}.l2a.execute.json", "examples/{}.ard.execute.json"):
        ours = json.loads((ROOT / path.format(name)).read_text())
        assert ours == json.loads((ROOT / path.format("eodh-ndvi")).read_text()), path

    ndvi_record = json.loads((ROOT / "records" / "eodh-ndvi.json").read_text())
    record = json.loads((ROOT / "records" / f"{name}.json").read_text())
    assert record["id"] == name
    properties = record["properties"]
    assert properties["applicableCollections"] == ndvi_record["properties"]["applicableCollections"]
    hub_types = {key: value["hubType"] for key, value in properties["inputParameters"].items()}
    assert hub_types == {"stac_item": "stac-item", "aoi": "aoi"}
    assert re.search(product, properties["title"] + properties["description"], re.IGNORECASE)
    links = {link["rel"]: link["href"] for link in record["links"]}
    assert links["application"].endswith(f"/cwl/{name}.cwl")
    assert links["self"].endswith(f"/items/{name}")
    assert links["vcs"] == {link["rel"]: link["href"] for link in ndvi_record["links"]}["vcs"]


def test_write_aoi_cog_reads_a_20m_band_onto_the_10m_grid_by_nearest(l2a_staged, tmp_path):
    from common import write_aoi_cog

    paths = _staged_bands(l2a_staged)
    with rasterio.open(paths["nir"]) as nir, rasterio.open(paths["swir22"]) as swir22:
        assert (nir.res, swir22.res) == ((10.0, 10.0), (20.0, 20.0))
        nir_transform, raw, swir22_transform = nir.transform, swir22.read(1).astype("float64"), swir22.transform

    out = tmp_path / "out.tif"
    write_aoi_cog("item", [Band(paths["nir"], 1), Band(paths["swir22"], 1)], AOI, out, lambda n, s: np.asarray(s, "f4"))
    with rasterio.open(out) as src:
        data, transform, crs = src.read(1), src.transform, src.crs

    assert (transform.a, transform.e) == (nir_transform.a, nir_transform.e)
    assert ((transform.c - nir_transform.c) / 10.0).is_integer() and (
        (transform.f - nir_transform.f) / 10.0
    ).is_integer()
    raw[raw == 0] = np.nan
    nearest = np.repeat(np.repeat(raw, 2, axis=0), 2, axis=1)
    nearest_transform = Affine(10.0, 0.0, swir22_transform.c, 0.0, -10.0, swir22_transform.f)
    expected = on_grid(nearest, nearest_transform, data.shape, transform)
    inside = polygon_mask(data.shape, transform, crs)
    assert inside.sum() > 40000
    np.testing.assert_array_equal(data[inside], expected[inside].astype("float32"))


def test_write_aoi_cog_rejects_a_band_in_another_crs(l2a_staged, tmp_path):
    from common import write_aoi_cog

    paths = _staged_bands(l2a_staged)
    other = tmp_path / "swir22_relabelled.tif"
    with rasterio.open(paths["swir22"]) as src:
        profile = {**src.profile, "driver": "GTiff", "crs": "EPSG:32631"}
        with rasterio.open(other, "w", **profile) as dst:
            dst.write(src.read())

    with pytest.raises(ValueError, match=r"(?i)crs") as error:
        write_aoi_cog("item", [Band(paths["nir"], 1), Band(other, 1)], AOI, tmp_path / "out.tif", lambda n, s: n)
    assert str(other) in str(error.value)
