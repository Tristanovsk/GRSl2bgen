import os, sys, re, glob

import numpy as np
import xarray as xr
import rioxarray
import datetime

from dateutil import parser
import logging
from pkg_resources import resource_filename

from . import __package__, __version__

opj = os.path.join


class Product():
    '''

    '''

    def __init__(self, l2a_obj):
        '''
        Get the L2A product object
        :param l2a_obj: path or xarray of the L2A input image
        '''
        self.processor = __package__ + '_' + __version__

        ##################################
        # Get image data
        ##################################
        if isinstance(l2a_obj, str):
            logging.info('Load L2A from files')
            if os.path.isdir(l2a_obj):

                basename = os.path.basename(l2a_obj)
                main_file = opj(l2a_obj, basename + '.nc')
                ancillary_file = opj(l2a_obj, basename + '_anc.nc')

                self.raster = xr.open_dataset(main_file, decode_coords='all', chunks={'wl': -1})
                self.ancillary = xr.open_dataset(ancillary_file, decode_coords='all')
            else:
                # get extension
                extension = l2a_obj.split('.')[-1]

                if extension == 'nc':
                    self.raster = xr.open_dataset(l2a_obj, decode_coords='all', chunks={'wl': -1})
                    self.ancillary = None
                elif 'zarr' in extension:
                    self.raster =xr.open_zarr(l2a_obj,decode_coords='all')
                    self.ancillary = None
                else:
                    logging.info('input file format not recognized, stop')
                    return
        elif isinstance(l2a_obj, xr.Dataset):
            self.raster = l2a_obj
            self.ancillary = None
            return

        # TODO deprecate and remove this part
        if self.raster.attrs['metadata_profile'] != 'beam':
            return

        # reshape into datacube:
        wls =  self.raster.wl

        if self.raster.dims.__contains__('wl'):
            self.raster=self.raster.drop_dims('wl')

        Rrs_vars = []
        Rrs_g_vars = []
        for wl in wls:
            Rrs_vars.append('Rrs_{:d}'.format(wl))
            Rrs_g_vars.append('Rrs_g_{:d}'.format(wl))

        Rrs = self.raster[Rrs_vars].to_array(dim='wl', name='Rrs').chunk({'wl': 1})
        Rrs = Rrs.assign_coords({'wl': wls})
        raster = self.raster.drop_vars(Rrs_vars)
        Rrs_g = self.raster[Rrs_g_vars].to_array(dim='wl', name='Rrs_g').chunk({'wl': 1})
        Rrs_g = Rrs_g.assign_coords({'wl': wls})
        raster = raster.drop_vars(Rrs_g_vars)
        self.raster = xr.merge([raster, Rrs, Rrs_g])




