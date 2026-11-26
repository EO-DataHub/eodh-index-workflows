# EODH index workflows

Spectral index workflows for the EO DataHub Workflow Runner. Each product rebuilds one of EOPro's products as a Hub workflow: it runs on one STAC item over a drawn area of interest, and writes a Cloud Optimised GeoTIFF (COG) with STAC that the Hub's map viewer can render.

All products share one Docker image and one version. Each product has its own CWL file and its own workflow catalogue record.

The repository started from the `eodh-workflow-starter` template. [AGENTS.md](AGENTS.md) describes the template's rules, checks and release steps, which still apply.

## Products

| Workflow ID | What it calculates | Collections | CWL | Record |
|---|---|---|---|---|
| `eodh-ndvi` | NDVI, `(nir - red) / (nir + red)`, the same values as EOPro. Shown with `ylgn` from -0.2 to 1. | `sentinel-2-c1-l2a`, `sentinel2_ard` | [cwl/eodh-ndvi.cwl](cwl/eodh-ndvi.cwl) | [records/eodh-ndvi.json](records/eodh-ndvi.json) |

Every product takes two inputs:

- `stac_item`: the URL of the STAC item, as a string.
- `aoi`: the area of interest, a GeoJSON Polygon, MultiPolygon, or a Feature with one of those, as a string. Pixels outside it are NaN.

The workflows fetch their own data, so ADES runs no stage-in step for them. A workflow fetches the item JSON from the URL, then reads only the area of interest from each band it needs, with HTTP range requests. It reads `http://`, `https://` and `file://` hrefs, without credentials. Relative hrefs are resolved against the item URL. Any other href, such as `s3://`, fails with a message that names it.

The CWL passes the product name to the image as the fixed argument `--workflow=<workflow ID>`. Users and the catalogue never see it.

## Repository layout

| Path | Contents |
|---|---|
| `src/workflow.py` | The dispatcher. It finds the products in `src/products/` and runs the one named by `--workflow`. |
| `src/products/` | One module per product. Each sets `NAME` (the workflow ID) and `run()`. |
| `src/common/` | Code the products share: AOI handling, band lookup and the brightness rule. |
| `src/main.py` | The entrypoint that the CWL calls. Do not edit it. |
| `src/eodh_workflow/` | Helpers from the template. Do not edit them. |
| `cwl/<workflow ID>.cwl` | The workflow definition that the Hub runs, one per product |
| `cwl/<workflow ID>.job.json` | The fixture that `make check-docker` runs the CWL on |
| `records/<workflow ID>.json` | The product's `workflow-catalogue` record |
| `scripts/make_golden.py` | Writes the EOPro golden arrays in `tests/fixtures/golden/`. Run by hand in an `eodh-workflows` environment. |
| `tests/` | Tests, including the EOPro parity tests |

## Checks

Prerequisites: [uv](https://docs.astral.sh/uv/), and Docker with `buildx` for `make check-docker`.

```bash
make install       # dependencies and pre-commit hooks
make check         # lint, tests, cwltool --validate on every CWL, versions
make check-docker  # build the image and run every CWL on its fixture through cwltool (needs the network)
make run           # run one product without Docker: make run WORKFLOW=eodh-ndvi
```

`make help` lists every command.

## How to add a product

Adding a product needs no change to shared code: no edit to `src/workflow.py`, the `Makefile` or `scripts/`. It needs no stage-in either: the product gets its bands through `band()`, which reads them from the fetched item.

1. **Product module.** Add `src/products/<name>.py` with `NAME = "eodh-<name>"` and `run(inputs, output_dir) -> OutputSpec`. Use `src/common/` for the AOI, the bands (`band()`) and the brightness rule. Set the `CollectionSpec` id to the workflow ID.
2. **CWL.** Copy `cwl/eodh-ndvi.cwl` to `cwl/eodh-<name>.cwl`. Change the `Workflow` `id`, `label` and `doc`, and the tool's `arguments` to `["--workflow=eodh-<name>"]`. Keep the image and version.
3. **Fixture.** Add `cwl/eodh-<name>.job.json` with the item URL and AOI that `make check-docker` runs it on.
4. **Record.** Copy `records/eodh-ndvi.json` to `records/eodh-<name>.json`. Change the `id`, title, description, keywords, `applicableCollections` and the `application` link.
5. **Tests.** Add tests for the product's values, and parity tests against EOPro golden arrays where EOPro has the product.
6. Run `make check` and `make check-docker`, and add the product to the table above.

## Versions and release

One version covers the image and every product. It is in `pyproject.toml`, and in `s:softwareVersion` and the `dockerPull` tag of every CWL in `cwl/`.

- `make versions` shows the version in each place.
- `make bump VERSION=x.y.z` sets it everywhere.

Release by pushing a `vX.Y.Z` tag; see [AGENTS.md](AGENTS.md#7-release). The image is `ghcr.io/eo-datahub/eodh-index-workflows`.

## Helpers version

The helpers' version is in `src/eodh_workflow/__init__.py`, and changes are in [CHANGELOG.md](CHANGELOG.md).
