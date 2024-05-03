
from pathlib import Path
import os, shutil
import zipfile
import tarfile
import glob

import numpy as np
import xarray as xr
import logging

from . import product, l2b_product
from . import chl,spm,cdom,transparency


opj = os.path.join

class process():
    def __init__(self):
        pass

    def execute(self,l2a_path,l2b_path):
        logging.info('import l2a product')
        prod = product(l2a_path)

        # ----------------------
        # get Chl-a parameters
        # ----------------------
        logging.info('get Chl-a parameters')
        chl_prod= chl(prod.raster)
        chl_prod.process()

        # ----------------------
        # get SPM parameters
        # ----------------------
        logging.info('get SPM parameters')
        spm_prod= spm(prod.raster)
        spm_prod.process()

        # ----------------------
        # get CDOM parameters
        # ----------------------
        logging.info('get CDOM parameters')
        cdom_prod= cdom(prod.raster)
        cdom_prod.process()


        # ----------------------
        # get transparency parameters
        # ----------------------
        logging.info('get transparency parameters')
        trans_prod= transparency(prod.raster)
        trans_prod.process()

        logging.info('construct l2b product')
        l2_raster_list = [chl_prod.output,spm_prod.output,cdom_prod.output,trans_prod.output]
        l2b = l2b_product(prod,l2_raster_list)
        logging.info('export l2b product into netcdf')
        l2b.to_netcdf(l2b_path)
