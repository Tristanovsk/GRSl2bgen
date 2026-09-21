Processing Chain Description
=============================

This page documents GRSl2bgen following the
`Processing Chain Documentation Template <https://processing-chain-guidelines.readthedocs.io/en/latest/Data_processing_chain_template/>`_.

Process description
--------------------

Description
~~~~~~~~~~~

GRSl2bgen is a scientific processor that derives water-quality parameters from a GRS
Level-2A (Rrs, water-leaving reflectance) product. Given one L2A product, it successively
computes an Optical Water Type (OWT) classification, chlorophyll-a concentration, suspended
particulate matter (SPM), colored dissolved organic matter (CDOM), and water transparency,
then merges all of these into a single Level-2B netCDF product.

The processor is invoked through the ``GRSl2bgen`` command-line executable
(entry point ``GRSl2bgen.run:main``, see :py:mod:`GRSl2bgen.run`), which wraps the core
:py:class:`GRSl2bgen.process.Process` class.

Application Domain (Granule)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One processing run (one granule) corresponds to **one GRS Level-2A product for one
acquisition date and one tile/footprint**, read as a single ``.nc``/``.zarr`` file or as a
directory containing a main file and its ``_anc.nc`` ancillary file (see
:py:class:`GRSl2bgen.product.Product`).

The output granule is a single Level-2B water-quality product (netCDF) covering the same
footprint and resolution as the input L2A product.

Scheduling and Triggers
~~~~~~~~~~~~~~~~~~~~~~~~

GRSl2bgen has no built-in scheduler; it is triggered externally, granule by granule:

- **Interactively / single granule**: manual call to ``GRSl2bgen <input_file>`` (see
  :doc:`index` "Example" section for a CLI example).
- **Containerized**: via the Docker image built from
  `Dockerfile <https://github.com/CNES/GRSl2bgen/blob/main/Dockerfile>`_ and run with
  `run_docker.sh <https://github.com/CNES/GRSl2bgen/blob/main/run_docker.sh>`_ (see
  :doc:`index` "Compile Docker image locally"). The image is built and published by the CI
  pipeline (GitLab ``kaniko_build`` job in ``.gitlab-ci.yml``, and the GitHub
  ``podman-build``/``pypi`` jobs in ``.github/workflows/main.yml``).

There is no periodicity of its own; scheduling (e.g. reprocessing on new GRS L2A outputs)
is delegated to the calling scripts/workflow.

Dataflow
--------

.. mermaid:: _diagrams/dataflow.mmd

Inputs
------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Data Type Name
     - Cardinality
     - Selection Criteria
   * - GRS Level-2A product (``<input_file>``)
     - 1..1 (mandatory)
     - Water-leaving reflectance (Rrs) product produced by
       `GRSprocessor <https://github.com/CNES/GRSprocessor>`_: a ``.nc`` file, a ``.zarr``
       store, or a directory containing ``<name>.nc`` plus its ``<name>_anc.nc`` ancillary
       file, passed as the positional CLI argument.

Outputs
-------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Data Type Name
     - Cardinality
     - Description
   * - Level-2B water quality product (netCDF)
     - 1..1
     - Merged product (see :py:class:`GRSl2bgen.output.L2bProduct`) containing the OWT
       indices/distances for three classifications (Spyrakos2018, Bi2024, Tarasenko2025),
       chlorophyll-a, SPM, CDOM and transparency variables, plus the ``flags``/``mask``
       variables carried over from the input product (see `Data Types`_ below for naming).

Return Codes
------------

GRSl2bgen is a Python CLI (``docopt``-based). Unlike GRSprocessor, it does not configure a
dedicated file logger: log records go through the standard Python ``logging`` module at
``INFO`` level (see ``GRSl2bgen/__init__.py``), with no ``log_file.log``/``error.log`` written
to disk by default.

- ``0``: normal process exit, including the case where ``--no_clobber`` is set and the
  output file already exists (the granule is skipped after printing a message).
- Non-zero interpreter exit codes may occur for unhandled exceptions raised during
  processing (no top-level exception handler currently wraps :py:meth:`GRSl2bgen.process.Process.execute`),
  or for invalid CLI arguments rejected by ``docopt``.

Log Format
----------

No structured/file-based log format is currently defined; messages are emitted through the
root ``logging`` logger (``INFO`` level and above) and go to the console (stderr) by
default.

Required Resources
-------------------

No SLURM/HPC job scripts or documented resource budgets ship with this repository (unlike
GRSprocessor). Actual CPU/RAM/runtime needs depend on the input product's spatial resolution
and tile size; the OWT classification step is the main configurable cost driver via the
``chunk``/``Nproc`` parameters of :py:class:`GRSl2bgen.owt.OWT_process` (multiprocessing via
``dask.delayed``).

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Resource Type
     - Quantity
     - Source / notes
   * - Disk — input
     - size of one GRS L2A product (netCDF/Zarr, plus ancillary file if any)
     - depends on sensor/resolution
   * - Disk — output
     - one Level-2B netCDF product, ``int16``-encoded with zlib compression (complevel 5)
     - written to ``-o``/``--odir`` (see :py:meth:`GRSl2bgen.output.L2bProduct.export_to_netcdf`)

Data Types
----------

.. list-table::
   :header-rows: 1
   :widths: 18 32 15 35

   * - Data Name
     - Description
     - Granule
     - Nomenclature
   * - GRS Level-2A input product
     - Water-leaving reflectance (Rrs) product from GRSprocessor
     - one tile, one date
     - e.g. ``S2B_MSIL2Agrs_<datetime>_N<baseline>_R<orbit>_T<tile>_<datetime>.nc``
   * - Level-2B output product
     - netCDF water-quality product (OWT, chlorophyll-a, SPM, CDOM, transparency)
     - one tile, one date
     - input basename with ``L2Agrs`` replaced by ``L2B``, e.g.
       ``S2B_MSIL2B_<datetime>_N<baseline>_R<orbit>_T<tile>_<datetime>.nc`` (see
       :py:func:`GRSl2bgen.run.main`)
