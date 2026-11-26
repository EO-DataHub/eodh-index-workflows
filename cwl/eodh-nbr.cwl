cwlVersion: v1.2
$namespaces:
  s: https://schema.org/
# Must equal the pyproject.toml version and the dockerPull tag. Change all of them with `make bump VERSION=x.y.z`.
s:softwareVersion: 0.3.0
schemas:
  - http://schema.org/version/9.0/schemaorg-current-http.rdf

$graph:
  # The Workflow's id is the workflow ID on the Hub.
  - class: Workflow
    id: eodh-nbr
    label: NBR
    doc: >-
      Normalised Burn Ratio, (nir - swir22) / (nir + swir22), for one Sentinel-2 L2A or Sentinel-2 ARD
      item over an area of interest. NBR needs the nir and swir22 bands; on L2A the 20 m swir22 band is read
      onto the 10 m nir grid. Writes a COG clipped to the area, with renderable STAC.
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
        dockerPull: ghcr.io/eo-datahub/eodh-index-workflows:0.3.0
    baseCommand: ["python3", "/app/main.py"]
    arguments: ["--workflow=eodh-nbr"]
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
    outputs:
      results:
        type: Directory
        outputBinding:
          glob: "."
