import os
import numpy as np
import xarray as xr
import rioxarray as rio
import logging
import datetime
from . import __package__, __version__

class L2bProduct():
    def __init__(self, prod, l2_raster_list):
        self.processor = __package__ + '_' + __version__
        self.prod = prod
        self.l2b_raster_list = l2_raster_list
        self.variables = None
        self.l2b_prod = None
        self.complevel = 5
        self.construct_l2b()

    def construct_l2b(self):
        logging.info('construct l2b')

        l2b_prod = xr.merge(self.l2b_raster_list,compat='override' )

        if 'flags' in self.prod.raster.keys():
            l2b_prod['flags'] = self.prod.raster['flags']
        else:
            l2b_prod['flags'] = xr.zeros_like(self.prod.raster.Rrs.isel(wl=0, drop=True).squeeze().astype(np.uint8))
        if 'mask' in self.prod.raster.keys():
            l2b_prod['mask'] = self.prod.raster['mask']
        else:
            l2b_prod['mask'] = xr.zeros_like(self.prod.raster.Rrs.isel(wl=0, drop=True).squeeze().astype(np.uint8))

        l2b_prod.attrs = self.prod.raster.attrs
        l2b_prod.attrs['processing_time'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        l2b_prod.attrs['processor'] = self.processor
        self.variables = list(l2b_prod.keys())
        self.l2b_prod = l2b_prod

    @staticmethod
    def compute_scale_and_offset(array, nbit=16):
        max_, min_ = np.nanmax(array), np.nanmin(array)
        # stretch/compress data to the available packed range
        scale_factor = (max_ - min_) / (2 ** nbit - 1)
        # translate the range to be symmetric about zero
        add_offset = min_ + 2 ** (nbit - 1) * scale_factor
        return scale_factor, add_offset

    def export_to_netcdf(self, ofile):
        '''
        Create output product dimensions, variables, attributes, flags....
        :return:
        '''
        logging.info('export into encoded netcdf')
        complevel = self.complevel
        encoding={}
        for variable in self.variables:

            if variable in ['mask','flags']:
                encoding[variable] = {
                    "zlib": True,
                    "complevel": complevel,
                    "grid_mapping": "spatial_ref"
                    }
            else:
                p = self.l2b_prod[variable]
                scale_factor, add_offset = self.compute_scale_and_offset(p.values, nbit=16)
                #offset = np.mean(p.range)
                #range = float(np.diff(p.range))
                #scale_factor = round(range / 60000, 6)
                encoding[variable] = {
                    'dtype': 'int16',
                    'scale_factor': scale_factor,
                    'add_offset': add_offset,
                    '_FillValue': -32768,
                    "zlib": True,
                    "complevel": complevel,
                    "grid_mapping": "spatial_ref"
                    }



        # write file
        if os.path.exists(ofile):
            os.remove(ofile)

        odir = os.path.dirname(ofile)
        if odir == '':
            odir='./'
        if not os.path.exists(odir):
            os.makedirs(odir)

        self.l2b_prod.to_netcdf(ofile, encoding=encoding)
        self.l2b_prod.close()

        return