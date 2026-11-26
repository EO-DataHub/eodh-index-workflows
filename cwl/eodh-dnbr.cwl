cwlVersion: v1.2
$namespaces:
  s: https://schema.org/
# Must equal the pyproject.toml version and the dockerPull tag. Change all of them with `make bump VERSION=x.y.z`.
s:softwareVersion: 0.4.1
schemas:
  - http://schema.org/version/9.0/schemaorg-current-http.rdf

$graph:
  # The Workflow's id is the workflow ID on the Hub.
  - class: Workflow
    id: eodh-dnbr
    label: dNBR
    doc: >-
      Differenced Normalised Burn Ratio, NBR(baseline) - NBR(item), for one post-fire Sentinel-2 L2A or
      Sentinel-2 ARD item over an area of interest, and the USGS burn-severity classes from it. The baseline
      item is found in the item's own collection, on the same tile, inside the baseline date range: the item
      with the least cloud, ties going to the date closest to the input item. Writes a dNBR COG and a
      burn-severity COG clipped to the area, with renderable STAC.
    requirements:
      # The runner sizes the job's pods from the largest ResourceRequirement in the file.
      - class: ResourceRequirement
        coresMax: 2
        ramMax: 4096
    inputs:
      stac_item:
        label: STAC item
        doc: >-
          The URL of a sentinel-2-c1-l2a or sentinel2_ard item with nir and swir22 bands. The workflow
          fetches the item and reads only the area of interest from its bands.
        type: string
      aoi:
        label: Area of interest
        doc: A GeoJSON Polygon, MultiPolygon, or a Feature with one of those, in EPSG:4326.
        type: string
      baseline:
        label: Baseline date range
        doc: >-
          The dates to find the pre-fire image in, as start/end (for example 2023-05-01/2023-05-31). The
          range must end before the item's date. The workflow uses the item from the same collection and
          tile with the least cloud, ties going to the date closest to the input item.
        type: string
    outputs:
      - id: results
        type: Directory
        outputSource:
          - main/results
    steps:
      main:
        run: "#main-tool"
        in:
          stac_item: stac_item
          aoi: aoi
          baseline: baseline
        out: [results]

  - class: CommandLineTool
    id: main-tool
    requirements:
      - class: ResourceRequirement
        coresMax: 2
        ramMax: 4096
      - class: NetworkAccess
        networkAccess: true
    hints:
      DockerRequirement:
        dockerPull: ghcr.io/eo-datahub/eodh-index-workflows:0.4.1
    baseCommand: ["python3", "/app/main.py"]
    arguments: ["--workflow=eodh-dnbr"]
    inputs:
      stac_item:
        type: string
        inputBinding:
          prefix: --stac_item=
          separate: false
      aoi:
        type: string
        inputBinding:
          prefix: --aoi=
          separate: false
      baseline:
        type: string
        inputBinding:
          prefix: --baseline=
          separate: false
    outputs:
      results:
        type: Directory
        outputBinding:
          glob: "."
