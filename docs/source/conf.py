# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Make the package importable without installation (local builds);
# on Read the Docs the package is also pip-installed (see .readthedocs.yaml).
DOCS_SOURCE = Path(__file__).resolve().parent
REPO_ROOT = DOCS_SOURCE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import GRSl2bgen  # noqa: E402

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GRSl2bgen'
copyright = '2026, CNES/Magellium'
author = 'Tristan Harmel, Soham Mukherjee'
release = GRSl2bgen.__version__
version = '.'.join(release.split('.')[:2])
today_fmt = '%Y-%m-%d'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
    'myst_nb',
    'IPython.sphinxext.ipython_console_highlighting',
]

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_readme.md', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autoclass_content = 'class'
autodoc_typehints = 'description'
# 'members' is set in the autosummary templates (_templates/) to avoid
# documenting objects twice
autodoc_default_options = {
    'member-order': 'bysource',
    'show-inheritance': True,
}

# NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

todo_include_todos = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'dask': ('https://docs.dask.org/en/stable', None),
    'zarr': ('https://zarr.readthedocs.io/en/stable', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
pygments_style = 'sphinx'

html_theme_options = {
    'repository_url': 'https://github.com/Tristanovsk/GRSl2bgen',
    'repository_branch': 'master',
    'path_to_docs': 'docs/source',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'use_download_button': True,
    'navigation_with_keys': True,
    'show_toc_level': 2,
    'secondary_sidebar_items': ['page-toc', 'edit-this-page'],
}

html_title = ''
html_logo = '_static/GRSl2bgen.png'
html_favicon = '_static/grsl2bgen_logo_v0.svg'

html_static_path = ['_static']
html_show_sourcelink = False
html_last_updated_fmt = today_fmt

htmlhelp_basename = 'grsl2bgen_doc'

# -- MyST / notebook rendering -----------------------------------------------

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
]

# Notebooks are executed only when their content changed; outputs are cached
# in docs/build/.jupyter_cache, which is committed so that Read the Docs can
# reuse it instead of re-running the (data-dependent) processing.
nb_execution_mode = 'cache'
nb_execution_cache_path = str(DOCS_SOURCE.parent / 'build' / '.jupyter_cache')
nb_execution_timeout = -1
nb_execution_allow_errors = True
nb_execution_raise_on_error = False
nb_merge_streams = True
