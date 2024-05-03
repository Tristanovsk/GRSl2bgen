import os
import numpy as np
import xarray as xr


class Chl():
    def __init__(self,
                 raster,
                 param='Rrs'):

        self.raster = raster
        self.Rrs = raster[param]
        self.OC2ratio = self.OC2_ratio()
        self.OC3ratio = self.OC3_ratio()
        self.output=None

    def process(self):
        # NASA OC2 for OCTS; bands 490, 565 nm
        acoef = [0.2236, -1.8296, 1.9094, -2.9481, -0.1718]
        self.chl_nasa_oc2 = self.OC2(acoef)
        self.chl_nasa_oc2 = self.set_range(self.chl_nasa_oc2)
        self.chl_nasa_oc2.name = 'Chla_OC2nasa'
        self.chl_nasa_oc2.attrs = {
            'description': 'Chl-a concentration from NASA OC2 with OCTS parameterization, bands 490, 565 nm',
            'applicability': 'oligotrophic waters',
            'coef':str(acoef),
            'reference': 'NASA OCx site',
            'units': 'mg m-3',
            'range':[0,2000]}

        # Gitelson-like Red-edge
        acoef = [232.329, 23.174]
        self.chl_M09B = self.M09B(acoef=acoef)
        self.chl_M09B = self.set_range(self.chl_M09B)
        self.chl_M09B.name = 'Chla_M09B'
        self.chl_M09B.attrs = {
            'description': 'Chl-a concentration from 705nm peak, bands 665, 705, 740 nm',
            'applicability': 'turbid and eutrophic waters',
            'coef':str(acoef),
            'reference': 'Moses, W.J.; Gitelson, A.A.; Berdnikov, S.; Povazhnyy, V. ' + \
                         'Estimation of chlorophyll-a concentration in case II waters using' + \
                         'MODIS and MERIS data—Successes and challenges. Environ. Res. Lett. 2009, 4, 1–8.',
            'units': 'mg m-3',
            'range':[0,2000]
            }

        self.output = xr.merge([self.chl_nasa_oc2,self.chl_M09B]).drop_vars('wl').compute()

    def set_range(self,param,minval=0,maxval=1200):
        return param.where((param>minval)&(param<maxval) )

    def OCX_chl(self, ratio, acoef):
        logchl = 0
        for i in range(len(acoef)):
            logchl += acoef[i] * ratio ** i
        chl = 10 ** (logchl)
        return chl

    def OC2_ratio(self):
        return np.log10(self.Rrs.sel(wl=490) / self.Rrs.sel(wl=560))

    def OC3_ratio(self):
        blue = self.Rrs.sel(wl=[443, 490]).max(dim='wl')
        return np.log10(blue / self.Rrs.sel(wl=560))

    def OC2(self, acoef):
        return self.OCX_chl(self.OC2ratio, acoef)

    def OC3(self, acoef):
        blue = np.max(self.Rrs.sel(wl=[443, 490]))
        ratio = np.log10(blue / self.Rrs.sel(wl=560))
        return self.OCX_chl(ratio, acoef)

    def RED2(self):
        return self.Rrs.sel(wl=705) / self.Rrs.sel(wl=665)

    def RED3(self):
        return (1 / self.Rrs.sel(wl=665) - 1 / self.Rrs.sel(wl=705)) * self.Rrs.sel(wl=740)

    # following Ogashawara et al. 2021
    def M09B(self, acoef=np.array([232.329, 23.174])):
        index = self.RED3()
        return (acoef[0] * index + acoef[1])

    def G10B(self, acoef=np.array([113.36, -16.45, 1.124])):
        index = self.RED3(self.Rrs)
        return ((acoef[0] * index + acoef[1]) ** acoef[2])

    def G11B(self, Rrs, acoef=np.array([315.5, 215.95, -25.66])):
        index = self.RED3(Rrs)
        return (acoef[0] * index ** 2 + acoef[1] * index + acoef[2])

    def A14B(self, Rrs, acoef=np.array([581.1, 25.5])):
        return self.M09B(Rrs, acoef=acoef)

    def B16B(self, Rrs, acoef=np.array([98.773, 34.763])):
        return self.M09B(Rrs, acoef=acoef)
