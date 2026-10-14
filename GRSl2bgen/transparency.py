import os
import numpy as np
import xarray as xr
from .config import SEL_XARRAY_METHOD


class Transparency():
    def __init__(self,
                 raster,
                 param='Rrs'):

        self.raster = raster
        self.Rrs = raster[param]
        self.output = None



    def process(self):
        # Kd(PAR) Roy and Das, 2022
        acoef = [3.09, -90.17]
        self.Kd_par = self.Kd_par_RD22(acoef=acoef)
        self.Kd_par = self.set_range(self.Kd_par)
        self.Kd_par.name = 'Kd_par'
        self.Kd_par.attrs = {
            'description': 'Coefficient of diffuse attenuation of PAR',
            'applicability': 'general',
            'coef': str(acoef),
            'reference': 'Roy and Das, 2022, Journal of Hydrology, table 3, eq. 13',
            'units': 'm-1',
            'range': [0, 200]
        }

        # where to add new parameters
        self.output = xr.merge([self.Kd_par], compat='no_conflicts', join='outer')
        # proceed with dask computing
        self.output = self.output.compute()

        return

    def set_range(self,param,minval=0,maxval=3000):
        return param.where((param>minval)&(param<maxval) )

    def zsd(self, acoef = [0]):
        # TODO implement secchi disk depth
        return acoef[0]*self.Rrs.sel(wl=490, method=SEL_XARRAY_METHOD)


    def Kd_par_RD22(self,acoef=[3.09,-90.17]):
        '''
        Kd_par from Roy and Das, 2022, Journal of Hydrology, table 3, eq. 13
        :return:
        '''
        R490 = self.Rrs.sel(wl=490, method=SEL_XARRAY_METHOD)
        R665 = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD)
        return acoef[0] * np.exp(acoef[1] * (R490 - R665))


