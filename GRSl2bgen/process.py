from pathlib import Path
import os, shutil
import zipfile
import tarfile
import glob

import numpy as np
import xarray as xr
import logging
from dask.distributed import Client

client = Client(processes=False)  # this yields a LocalCluster that doesn't have multiprocessing capabilities (doc is very brief and not very helpful: http://distributed.dask.org/en/stable/api.html#distributed.LocalCluster)
from . import Product, L2bProduct
from . import Chl, Spm, Cdom, Transparency, OWT_process

opj = os.path.join


class Process():
    def __init__(self,
                 l2a_obj,
                 l2b_path='./l2b_product.nc'):
        self.l2a_obj = l2a_obj
        self.l2b_path = l2b_path
        self.successful = False

    def execute(self, ):
        logging.info('import l2a product')
        l2a_obj = self.l2a_obj

        prod = Product(l2a_obj)

        #  ----------------------
        # get OWT parameters
        # ----------------------
        logging.info('get OWT classification')
        owt_process = OWT_process(prod.raster)
        owt_process.execute()

        # ----------------------
        # get SPM parameters
        # ----------------------
        logging.info('get SPM parameters')
        spm_prod = Spm(prod.raster)
        spm_prod.process()

        # ----------------------
        # get Chl-a parameters
        # ----------------------
        logging.info('get Chl-a parameters')
        chl_prod = Chl(prod.raster)
        chl_prod.process()

        # ----------------------
        # get CDOM parameters
        # ----------------------
        logging.info('get CDOM parameters')
        cdom_prod = Cdom(prod.raster)
        cdom_prod.process()

        # ----------------------
        # get transparency parameters
        # ----------------------
        logging.info('get transparency parameters')
        trans_prod = Transparency(prod.raster)
        trans_prod.process()

        logging.info('construct l2b product')
        l2_raster_list = [
            owt_process.output,
            chl_prod.output,
            spm_prod.output,
            cdom_prod.output,
            trans_prod.output]
        self.l2b = L2bProduct(prod, l2_raster_list)
        self.successful = True

    def write_output(self):
        logging.info('export final l2b product into netcdf')
        self.l2b.export_to_netcdf(self.l2b_path)
