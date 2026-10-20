'''
v0.0.1: version adapted from obs2co_l2bgen
v0.0.2: add OWT retrieval and multiple source L2A input
v0.1.0: add Xtrem quality blended algorithm + CI env
v1.0.0:
    - first stable version for CNES
    - documentation with readthedocs
v1.0.1:
    - fix for a fractional power of a negative number
v1.0.2:
    - fix for the log of negative band ratios (CDOM, Chl-a OCx ratios)
    - remove the 'All-NaN slice' and xr.merge FutureWarning warnings
    - version read from GRSl2bgen.__version__ in pyproject.toml
    - Docker image built from pyproject.toml (setup.py removed)
v1.1.0dev0:
    - OWT-blended SPM from the Cordeiro (2022) SPM OWT (merge of feature/SPM)
    - QAA v6 adg443 for MSI (adg443_QAAv6), with the OWT-blended Chl-a as first guess
    - adg443 from QAA v6 a(443) minus Bricaud (1998) aph(443) from the Chl-a (adg443_QAAv6Chl)
    - Secchi disk depth of Lee et al. (2015) from QAA v6 IOPs (Zsd_Lee15)
    - utils: pure water IOPs with T/S correction (iopw), Bricaud (1998) aphi, QAA v6 IOPs
    - read GRS L2A Zarr pyramids, with a level option (Product, Process, CLI --level)
    - OWT blends (Chl-a, SPM) normalised by the weights of the valid estimates only
    - band ratios masked band by band (no value from two negative reflectances)
    - NetCDF inputs read with their stored chunks
    - config.SEL_XARRAY_METHOD for the selection of the wavelengths
    - fix default L2B output name and units of the Euclidean OWT distances
    - CLI log file next to the output (<name>.log), in the GRS processor format
    - atomic export: the product is renamed to its final name once complete
'''


__package__ = 'GRSl2bgen'
__version__ = '1.1.0dev0'

from .product import Product
from .output import L2bProduct

from .chlorophyll_a import Chl
from .suspended_particulate_matter import Spm
from .cdom import Cdom
from .transparency import Transparency
from .owt import OWT, OWT_process
from .utils import iopw

from .process import Process



import logging

#init logger
logger = logging.getLogger()

level = logging.getLevelName("INFO")
logger.setLevel(level)

