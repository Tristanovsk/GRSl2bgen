'''
v0.0.1: version adapted from obs2co_l2bgen
v0.0.2: add OWT retrieval and mulitple source L2A input
'''


__package__ = 'GRSl2bgen'
__version__ = '0.0.2'

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

