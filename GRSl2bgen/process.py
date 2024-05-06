
from pathlib import Path
import os, shutil
import zipfile
import tarfile
import glob

import numpy as np
import xarray as xr
import logging

from . import Product, L2bProduct
from . import Chl,Spm,Cdom,Transparency


opj = os.path.join

class Process():
    def __init__(self):
        pass

    def execute(self,l2a_path,l2b_path):
        logging.info('import l2a product')
        prod = Product(l2a_path)

        #  ----------------------
        # get OWT parameters
        # ----------------------

        # ----------------------
        # get Chl-a parameters
        # ----------------------
        logging.info('get Chl-a parameters')
        chl_prod= Chl(prod.raster)
        chl_prod.process()

        # ----------------------
        # get SPM parameters
        # ----------------------
        logging.info('get SPM parameters')
        spm_prod= Spm(prod.raster)
        spm_prod.process()

        # ----------------------
        # get CDOM parameters
        # ----------------------
        logging.info('get CDOM parameters')
        cdom_prod= Cdom(prod.raster)
        cdom_prod.process()


        # ----------------------
        # get transparency parameters
        # ----------------------
        logging.info('get transparency parameters')
        trans_prod= Transparency(prod.raster)
        trans_prod.process()

        logging.info('construct l2b product')
        l2_raster_list = [chl_prod.output,spm_prod.output,cdom_prod.output,trans_prod.output]
        l2b = L2bProduct(prod, l2_raster_list)
        logging.info('export l2b product into netcdf')
        l2b.to_netcdf(l2b_path)
