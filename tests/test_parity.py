"""Stage 3: the workflow's NDVI equals EOPro's golden arrays inside the AOI polygon, on the same grid."""

import numpy as np
import pytest

from golden_data import on_grid, polygon_mask, read_golden, read_output, run_ndvi


@pytest.mark.parametrize("name", ["l2a", "ard"])
def test_stage3_ndvi_matches_eopro_golden_inside_aoi(name, request, tmp_path):
    golden, golden_transform, golden_crs = read_golden(name)
    data, transform, crs = read_output(run_ndvi(request.getfixturevalue(f"{name}_staged"), tmp_path / "out"))

    assert crs.to_string() == golden_crs
    ours = on_grid(data, transform, golden.shape, golden_transform)
    inside = polygon_mask(golden.shape, golden_transform, golden_crs)
    assert inside.sum() > 40000
    assert np.allclose(ours[inside], golden[inside], equal_nan=True)
