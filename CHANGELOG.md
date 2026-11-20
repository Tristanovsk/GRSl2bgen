# Changelog

All notable changes to GRSl2bgen are listed here, the most recent first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
[Semantic Versioning](https://semver.org/). Dates are those of the version change in the
repository.

## [Unreleased]

### Added

- **Classes left out of an OWT classification**: option `exclude_classes` of `OWT` (labels of
  the `owt` coordinate of the database, e.g. `['5a']`, or 1-based positions). The excluded
  classes are removed before the spectral distance and the other classes keep their number,
  so that `owt_index_<name>` has the same numbering as with all the classes; the excluded
  classes are listed in the attribute `excluded_classes` of `owt_index_<name>`. For the
  Tarasenko et al. (2025) classification of the L2B chain: option `ta2025_exclude` of
  `OWT_process` and `Process`, command-line option `--ta2025_exclude` (comma-separated),
  default `GRSl2bgen.config.TA2025_EXCLUDE_CLASSES` (empty: the 16 classes, products
  unchanged), which can be set with the environment variable `GRSL2BGEN_TA2025_EXCLUDE`
  (e.g. `5a`, also read by the child processes of a batch). Without the eutrophic type 5a of Bi et al. (2024), the closest water type to
  the bloom types, the diluted edges of the *L. chlorophorum* blooms are classified as a
  bloom type instead of 5a (30TWT, 2017-2022: bloom extent of the time series +34 %, same
  REPHY matchups); identical to a classification with the 5a spectrum made unreachable.

### Changed

- **OWT of the SPM blending: the Cordeiro (2022) classes are replaced by 14 water types
  simulated with invRrs** (`owt_database='invRrsWT'`, `data/owt_invrrs_water_types.nc`),
  from oligotrophic to extremely turbid waters along the biomass, CDOM and mineral
  particle axes, classified with the spectral angle (2 best classes). Recipe
  (`GRSl2bgen.suspended_particulate_matter.SPM_OWT_RECIPE`): Jiang et al. (2021) green
  band for the biomass and CDOM types, Jiang et al. (2021) red band for the
  hypereutrophic and moderately turbid types, Zhang et al. (2014) for the turbid to
  extremely turbid types; where no blended estimate is in its validity range, the
  algorithm of the best class is used without it (within 0-2000 mg L-1, option
  `fallback`). On GLORIA (3038 samples with TSS, MSI bands): median symmetric accuracy
  70.5 % and bias -12 %, against 72.5 % and -18 % with Cordeiro, every sample with a
  value; better in lakes, coastal waters and estuaries, worse in rivers (141 % against
  126 %, 193 samples); recipe cross-validated by data set (70.5-72.2 %). On a Loire
  image (30TWT, 2017-04-12), the clear waters are unchanged and the turbid ones 1.3-1.8
  times higher. Output: `owt_index_invRrsWT` and `owt_dist_invRrsWT` (radians) replace
  `owt_index_Cordeiro2022SPM` and `owt_dist_Cordeiro2022SPM`; `SPM_OWTblend_Ta2025`
  keeps its name. `Spm.spm_blending` takes the options `recipe` and `fallback`.

### Removed

- The Cordeiro (2022) OWT database (`owt_database='Cordeiro2022SPM'`,
  `data/owt_cordeiro2022_spm.nc`, `data/Means_OWT_Cordeiro_S2A_SPM.json`) and the
  commented-out former implementation of the SPM blending.

### Fixed

- OWT classification with negative L2A reflectance: the spectral angle and the Euclidean
  distance used the negative values as they were. The spectrum of a pixel with a negative
  band in the classification range (350-800 nm) is now raised by a spectrally flat offset,
  `-min(Rrs)`, so that its minimum is 0; every pixel stays classified
  (`GRSl2bgen.owt.handle_negative_rrs`, option `negative_handling` of `OWT`: `'offset'` by
  default, `'clip'`, `'mask'` with `negative_tolerance`, or `None` for the previous
  behaviour).
- Documentation of the low threshold of the SWIR module (`swir_module_low`, `QC_L2A`):
  0.0005 sr-1, the value used by the code (0.002 sr-1 was written in the docs).

### Added

- Flags of the level of negativity of the L2A reflectance, `-min(Rrs)` over the bands of
  the OWT classification, in `flags_l2b` (`QC_L2A`, options `negative_thresholds` and
  `negative_wl_range`): `neg_rrs_low`, `neg_rrs_medium`, `neg_rrs_high` (bits 4-6,
  above 1e-4, 5e-4 and 2e-3 sr-1, cumulative), and `neg_rrs_swir` (bit 7):
  `neg_rrs_medium` and SWIR module above the low threshold. The GRS flag `neg_rrs` only
  tests 490 and 565 nm. On three 30TWT images, small negative values (red and near
  infrared) are noise of clear waters, whereas the large ones (mostly at 443 nm) go with a
  non-black SWIR (median SWIR module 10-50 times that of the water): over-correction of
  mixed land/water, adjacency, floating or emerged vegetation. `neg_rrs_swir` flags
  0.7-1.9 % of the water pixels but 41-65 % of the pixels classified as a Ta2025 bloom
  type (mostly cyanobacteria and 5b). Tests.

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
