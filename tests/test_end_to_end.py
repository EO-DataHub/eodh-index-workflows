"""
Level 3: run the real entrypoint (src/main.py) on the staged golden L2A item with the eodh-ndvi
workflow and an AOI, then check the output directory against every rule the Hub depends on.
"""

import pystac

import main
from eodh_workflow import output_problems
from golden_data import AOI_JSON, ITEM_IDS


def _run(staged, output_dir):
    main.main([f"--stac_item_dir={staged}", f"--output_dir={output_dir}", "--workflow=eodh-ndvi", f"--aoi={AOI_JSON}"])
    assert output_problems(output_dir) == []
    catalog = pystac.Catalog.from_file(str(output_dir / "catalog.json"))
    return catalog, next(catalog.get_children())


def test_stage1_output_is_valid_and_renderable(l2a_staged, tmp_path):
    _, collection = _run(l2a_staged, tmp_path / "out")
    [item] = list(collection.get_items())
    assert item.id == f"{ITEM_IDS['l2a']}_ndvi"
    assert item.properties["eodh:source_item_id"] == ITEM_IDS["l2a"]
    assert item.properties["renders"] == collection.extra_fields["renders"]
    [render] = item.properties["renders"].values()
    assert render["assets"] == ["data"]
    assert render["rescale"] == [[-0.2, 1]]
    assert render["colormap_name"] == "ylgn"
    assert item.assets["data"].media_type.startswith("image/tiff")


def test_stage1_each_run_gets_its_own_collection(l2a_staged, tmp_path):
    first_catalog, first = _run(l2a_staged, tmp_path / "first")
    second_catalog, second = _run(l2a_staged, tmp_path / "second")
    assert first_catalog.id == second_catalog.id
    assert first.id != second.id
