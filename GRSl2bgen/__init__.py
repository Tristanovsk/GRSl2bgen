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
'''


__package__ = 'GRSl2bgen'
__version__ = '1.0.2'

from .product import Product
from .output import L2bProduct

from .chlorophyll_a import Chl
from .suspended_particulate_matter import Spm
from .cdom import Cdom
from .transparency import Transparency
from .owt import OWT, OWT_process

from .process import Process



import logging

#init logger
logger = logging.getLogger()

level = logging.getLevelName("INFO")
logger.setLevel(level)

