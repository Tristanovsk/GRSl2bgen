import os
import numpy as np
import xarray as xr


class cdom():
    def __init__(self,
                 raster,
                 param='Rrs'):

        self.raster = raster
        self.Rrs = raster[param]
        self.output = None



    def process(self):

        # Brezonik et al, 2015
        acoef = [1.872, -0.83]
        acdom_B15 = self.brezonik15(acoef=acoef)
        acdom_B15 = self.set_range(acdom_B15)
        acdom_B15.name = 'acdom_B15'
        acdom_B15.attrs = {
            'description': 'CDOM absorption at 440 nm-1 665, 705, 740 nm',
            'appicability': 'general',
            'coef':str(acoef),
            'reference': 'Brezonik et al., 2015, http://dx.doi.org/10.1016/j.rse.2014.04.033',
            'units': 'm-1',
            'range':[0,60]}

        self.output = xr.merge([acdom_B15])

    def set_range(self,param,minval=0,maxval=30):
        return param.where((param>minval)&(param<maxval) )

    def brezonik15(self, acoef = [1.872, -0.83]):
        return np.exp(acoef[0] + acoef[1] * np.log(self.Rrs.sel(wl=490) / self.Rrs.sel(wl=740)))
