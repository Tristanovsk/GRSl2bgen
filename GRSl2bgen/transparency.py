import os
import numpy as np
import xarray as xr


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

        self.output = xr.merge([self.Kd_par])

        return

    def set_range(self,param,minval=0,maxval=3000):
        return param.where((param>minval)&(param<maxval) )

    def zsd(self, acoef = [0]):
        # TODO implement secchi disk depth
        return acoef[0]*self.Rrs.sel(wl=490)


    def Kd_par_RD22(self,acoef=[3.09,-90.17]):
        '''
        Kd_par from Roy and Das, 2022, Journal of Hydrology, table 3, eq. 13
        :return:
        '''
        return acoef[0] * np.exp(acoef[1] * (self.Rrs.sel(wl=490) - self.Rrs.sel(wl=665)))
