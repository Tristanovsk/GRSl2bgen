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
[`.github/workflows/main.yml`](.github/workflows/main.yml)). There is no `tests/` directory in
the repository yet — if you add a feature or fix a bug, please add tests there as part of your
change.

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
