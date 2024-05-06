'''
v0.0.1: version adapted from obs2co_l2bgen

'''


__package__ = 'GRSl2bgen'
__version__ = '0.0.1'

from .product import Product
from .output import L2bProduct
from .chlorophyll_a import Chl
from .suspended_particulate_matter import Spm
from .cdom import Cdom
from .transparency import Transparency
from .process import Process
from .owt import OWT


import logging

#init logger
logger = logging.getLogger()

level = logging.getLevelName("INFO")
logger.setLevel(level)

