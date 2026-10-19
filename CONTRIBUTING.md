# Contributing to GRSl2bgen

Thanks for your interest in contributing! This document explains how to report issues, set up
a development environment, and submit changes.

## Reporting issues

Please use [GitHub Issues](https://github.com/CNES/GRSl2bgen/issues) to report bugs or
request features. Include:

- the `GRSl2bgen` version (see `GRSl2bgen --version`) and how you installed it (pip, conda,
  Docker),
- the exact command line used,
- the relevant console output (GRSl2bgen currently logs to stderr, see the
  [processing chain documentation](https://cnes.github.io/GRSl2bgen/processing_chain.html#log-format)
  for details — there is no `log_file.log`/`error.log` on disk yet),
- if possible, a minimal way to reproduce (input L2A product, resolution...).

## Development setup

```bash
git clone https://github.com/CNES/GRSl2bgen.git
cd GRSl2bgen
conda create -n l2bgen_dev python=3.12
conda activate l2bgen_dev
conda install -c conda-forge rasterio xarray dask rioxarray
pip install -r requirements.txt
pip install -e .[dev]
```

The `dev` extra (see `pyproject.toml`) installs `black`, `isort`, `bumpver`, `pip-tools` and
`pytest`.

## Code style

- Format code with `black` and `isort` before committing.
- CI also runs `ruff` and `mypy` on pull requests (see
  [`.github/workflows/main.yml`](.github/workflows/main.yml), job `lint`); please fix warnings
  they raise on the lines you touch.

## Running tests

```bash
pytest tests/
```

This is the same test suite executed by CI (job `python-tests` in
[`.github/workflows/main.yml`](.github/workflows/main.yml)). Run it from the environment of the
development setup above; the whole suite takes about 20 s. If you add a feature or fix a bug,
please add tests in `tests/` as part of your change.

The end-to-end tests (`tests/test_process.py`) run the full processing chain on a small real
L2A product stored in `tests/data/`: a 128 x 128 pixel crop (2.56 km) of a Sentinel-2 GRS scene
over the Camargue lagoons (tile 31TFJ, 2021-02-11), kept as a two-level Zarr pyramid with the
variables, attributes and packing of the full product. It was made with
`tests/data/make_test_l2a.py`, which can rebuild it from the source product or crop another
L2A for a new test case:

```bash
python tests/data/make_test_l2a.py <L2A product dir or .zarr> [y0 x0 size]
```

`tests/conftest.py` provides session-scoped fixtures, so the chain runs only once per test
session whatever the number of tests using them:

- `l2a_path`: path of the test L2A product;
- `l2b_nc`: L2B NetCDF produced from it by `Process(...).run()`;
- `l2b_pyramid`: L2B multiscale Zarr store (levels 0 to 2) produced from it.

## Submitting changes

1. Create a branch off `develop` named `feature/<short-description>`.
2. Make your changes, with tests where relevant.
3. Open a pull request targeting `develop` (CI runs automatically on PRs to `main` and
   `develop`). Make sure the `python-tests` job passes.
4. One of the maintainers will review your PR.

## Documentation

The documentation lives under [`docs/`](docs/) (Sphinx) and is published to
[cnes.github.io/GRSl2bgen](https://cnes.github.io/GRSl2bgen/) on every push to `main`.
See `docs/source/index.rst` for the entry point. You can build it locally with:

```bash
pip install sphinx furo myst-parser sphinx-autoapi sphinxcontrib-mermaid
cd docs
sphinx-build -b html source build/html
```

## Code of conduct

Please be respectful and constructive in issues, pull requests, and discussions. For anything
sensitive, contact the maintainers directly.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE), the license of this project.
