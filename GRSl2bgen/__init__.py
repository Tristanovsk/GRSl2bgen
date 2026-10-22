'''GRSl2bgen: Level-2B water quality products from GRS L2A images.

Release history: CHANGELOG.md at the root of the repository.
'''


__package__ = 'GRSl2bgen'
__version__ = '1.2.0dev0'

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

