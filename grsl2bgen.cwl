# Based on OGC best practises (8.4.  Application)
cwlVersion: v1.2
$namespaces:
  s: https://schema.org/
  js: https://json-schema.org/
$schemas:
- http://schema.org/version/latest/schemaorg-current-http.rdf
$graph:
- class: Workflow
  id: main
  label: GRSl2bgen CWL
  doc: Process a GRS L2A netCDF reflectance file into L2B water quality parameters
  requirements:
    InlineJavascriptRequirement: {}
  inputs:
    bucket_name:
      type: string
      doc: S3 bucket holding input_file
    input_file:
      type: string
      doc: Path (key) of the L2A netCDF file (GRS output) within bucket_name
    no_clobber:
      type: boolean
      doc: Do not process <input_file> if the output file already exists.
      default: True
  steps:
    retrieve_input:
      run: '#download_file_from_s3'
      in:
        bucket_name: bucket_name
        file_path: input_file
      out:
        [data]
    l2bgen_run:
      run: '#l2bgen_chain'
      in:
        input_file: retrieve_input/data
        no_clobber: no_clobber
      out:
        [nc]
  outputs:
    nc:
      type: File[]
      outputSource: l2bgen_run/nc

- class: CommandLineTool
  id: download_file_from_s3
  doc: |
    Download a single object from bucket_name/file_path as a staged local File.
  baseCommand: download_from_s3
  arguments: ["--out", $(runtime.outdir)/staged]
  hints:
    DockerRequirement:
      dockerPull: artifactory.cnes.fr/platform-incub-docker-prod-local/cwl-processing/cwl_helpers:v1.3.3
    ResourceRequirement:
      ramMin: 200
  requirements:
    NetworkAccess:
      networkAccess: true
    ToolTimeLimit:
      timelimit: 3600
  inputs:
    bucket_name:
      type: string
      inputBinding:
        prefix: "--bucket"
        position: 1
    file_path:
      type: string
      inputBinding:
        prefix: "--path"
        position: 2
  outputs:
    data:
      type: File
      outputBinding:
        glob: $(runtime.outdir)/staged/*

- class: CommandLineTool
  id: l2bgen_chain
  baseCommand: "GRSl2bgen"
  arguments: ["--odir", $(runtime.outdir)]
  hints:
    DockerRequirement:
      # TODO: replace with the published Docker Hub image once l2bgen is pushed
      # (see .github/workflows/main.yml, job podman-build, image "l2bgen").
      dockerPull: TODO_REPLACE_WITH_DOCKERHUB_IMAGE/l2bgen:1.2.0dev0
    ResourceRequirement:
      ramMax: 4000
  requirements:
    ToolTimeLimit:
      timelimit: 3600
  inputs:
    input_file:
      type: File
      inputBinding:
        position: 1
    no_clobber:
      type: boolean
      inputBinding:
        prefix: "--no_clobber"
        position: 2
  outputs:
    nc:
      type: File[]
      outputBinding:
        glob: $(runtime.outdir)/*.nc
