'''
v0.0.1: version adapted from obs2co_l2bgen

'''


__package__ = 'GRSl2bgen'
__version__ = '0.0.1'

from .product import product
from .output import l2b_product
from .chlorophyll_a import chl
from .suspended_particulate_matter import spm
from .cdom import cdom
from .transparency import transparency
from .process import process


import logging

#init logger
logger = logging.getLogger()

level = logging.getLevelName("INFO")
logger.setLevel(level)

