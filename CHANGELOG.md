# Changelog

All notable changes to GRSl2bgen are listed here, the most recent first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
[Semantic Versioning](https://semver.org/). Dates are those of the version change in the
repository.

## [1.2.0] - 2026-11-17

### Added

- Diffuse attenuation at 490 nm: NASA KD2 band ratio with the MERIS KD2E coefficients
  (`Kd490_KD2`) and Lee et al. (2013) from the QAA v6 IOPs (`Kd490_Lee13`).
- Kd(PAR) over the first optical depth of Morel et al. (2007), from `Kd490_Lee13`
  (`Kd_par_Mo07`).
- Euphotic depth (1 % of the PAR) and Kd(PAR) over the euphotic zone of Lee et al. (2005),
  from the QAA v6 IOPs (`Zeu_Lee05`, `Kd_par_Lee05`).
- Batch processing notebook: GRS NetCDF L2A products (folder `<name>/` holding `<name>.nc`
  and `<name>_anc.nc`), detected product by product next to the Zarr ones.
- Case study notebook: light attenuation and transparency in the Étang de Berre.
- Tests of the NetCDF L2A input and of the KD2 coefficients.
- This changelog, also rendered as the History page of the documentation.
- Quick look of the L2B product, `<name>_ql_l2b.png` (Chl-a, SPM and Kd(PAR) on fixed
  logarithmic scales, read from a coarse pyramid level), written with the STAC Item
  (`--stac`, `write_stac_item`) and added to it as its `thumbnail` asset
  (`GRSl2bgen.quicklook`).
- Processing restricted to the water pixels: the retrievals are computed on the valid
  pixels (`mask == 0`) packed side by side on a compact grid, block by block, then put back
  on the image grid (`GRSl2bgen.compact.CompactGrid`, option `compact` of `Process`, True
  by default). Same product; on full Sentinel-2 tiles at 20 m, the run takes 13 % less time
  with 34 % of water pixels, 24 % less with 3 % and 37 % less without water, with the
  same peak memory.

### Changed

- Python 3.11 to 3.14, all tested by CI (test matrix); build, lint and documentation on 3.12.
- `develop` is the default branch on GitHub; the coverage badge is updated on each push to
  `develop` instead of `main`.
- Logging: importing `GRSl2bgen` no longer sets the root logger to `INFO`; the package logs
  through the loggers of its modules (`GRSl2bgen.process`, `GRSl2bgen.output`, ...), named in
  the log records (`GRSl2bgen.output::export_to_netcdf` instead of `root::export_to_netcdf`).
  The command line and `RunLogger` still log at `INFO`; from Python, configure the logging
  (e.g. `logging.basicConfig(level=logging.INFO)`) to see the processing steps.
- Logging level names passed to `setLevel` directly instead of `logging.getLevelName`.
- Default L2B name: the `L2A` level tag of the GRS NetCDF products is replaced by `L2B` as
  well (`MSIL2A` -> `MSIL2B`), not only `L2Agrs` (any case).
- Batch processing notebook: a Zarr product is complete only when its levels hold the
  `Rrs` and `mask` arrays (a GRS run that failed while writing the store leaves them out).
- Faster NetCDF output: with `persist=True` (the default of `Process`), the L2B product,
  already computed in memory, is written from numpy arrays instead of chunk by chunk through
  dask; the file is unchanged and a full Sentinel-2 tile at 20 m is processed in about 70 s
  instead of 90 s, with the same peak memory (the memory freed by the dask threads is
  returned to the system, `malloc_trim` of glibc).
- NetCDF output: with `persist=True`, the variables are packed to `int16` (and the OWT class
  IDs to `uint8`) in parallel by dask, with the CF encoder of xarray, before the file is
  written, instead of on one thread by the writer. Same file (values, attributes and their
  order); on a full Sentinel-2 tile at 20 m the run is about 5 s faster with 34 % of water
  pixels and 16 s faster without water. The packed copy is kept beside the product during
  the write: up to 0.7 GB more peak memory on a tile with much water.

### Fixed

- STAC Item: the `derived_from` link of an L2A product given by URL (e.g. `s3://`) keeps the
  URL; it was made a local absolute path (`<cwd>/s3:/bucket/...`).

## [1.1.1] - 2026-10-23

### Fixed

- `scipy` added to the dependencies: it is needed by the interpolations of xarray and was only
  installed by chance.
- Python 3.11 or later (`requires-python`, as zarr 3), instead of 3.9.
- Docker image on Ubuntu 24.04 (Python 3.12), in a virtual environment: the image could not
  install zarr 3 with the Python 3.10 of Ubuntu 22.04.
- Packing tests through a temporary file (in-memory NetCDF needs `h5netcdf`, not a dependency).
- CI: the coverage badge is updated on each push to `main` (it was stuck at 23 %), and the
  tests fail below 65 % coverage.

## [1.1.0] - 2026-10-21

### Added

- OWT-blended SPM from the Cordeiro (2022) SPM optical water types.
- CDOM and detritus absorption of QAA v6 for MSI (`adg443_QAAv6`), with the OWT-blended
  Chl-a as first guess, and a(443) of QAA v6 minus the Bricaud (1998) aph(443) of the Chl-a
  (`adg443_QAAv6Chl`).
- Secchi disk depth of Lee et al. (2015) from the QAA v6 IOPs (`Zsd_Lee15`).
- `utils`: pure water IOPs with temperature and salinity correction (`iopw`), Bricaud (1998)
  aph, QAA v6 IOPs.
- GRS L2A Zarr pyramids as input, with a level option (`Product`, `Process`, CLI `--level`).
- `config.SEL_XARRAY_METHOD` for the selection of the wavelengths.
- CLI: log file next to the output (`<name>.log`) in the GRS processor format,
  `--product_dir` (one folder per product) and `--stac` (STAC Item, written last).
- Batch processing tutorial for full tiles.

### Changed

- OWT blends (Chl-a, SPM) normalised by the weights of the valid estimates only.
- Band ratios masked band by band (no value from two negative reflectances).
- NetCDF inputs read with their stored chunks.
- Atomic export: the product is written under a temporary name and renamed once complete.
- The output keeps only the `time` and `spatial_ref` scalar coordinates (no more `band`,
  `wl`, `central_wavelength`, `forecast_period` and `forecast_reference_time`).

### Fixed

- Default L2B output name.
- Units of the Euclidean OWT distances.

## [1.0.2] - 2026-10-13

### Fixed

- Log of negative band ratios (CDOM, Chl-a OCx ratios).
- `All-NaN slice` and `xr.merge` FutureWarning warnings removed.

### Changed

- Version read from `GRSl2bgen.__version__` in `pyproject.toml`.
- Docker image built from `pyproject.toml` (`setup.py` removed).

## [1.0.1] - 2026-10-13

### Fixed

- Fractional power of a negative number.

## [1.0.0] - 2026-10-01

### Added

- First stable version for CNES.
- Documentation on Read the Docs.

## [0.1.0] - 2026-09-25

### Added

- Xtrem quality blended algorithm and CI environment.

## [0.0.2] - 2025-07-02

### Added

- OWT retrieval and L2A inputs from multiple sources.

## [0.0.1] - 2024-05-03

### Added

- First version, adapted from obs2co_l2bgen.

[Unreleased]: https://github.com/Tristanovsk/GRSl2bgen/compare/v1.1.1...develop
[1.1.1]: https://github.com/Tristanovsk/GRSl2bgen/releases/tag/v1.1.1
[1.1.0]: https://github.com/Tristanovsk/GRSl2bgen/releases/tag/v1.1.0
