import os
import numpy as np
import xarray as xr





class Spm():
    def __init__(self,
                 raster,
                 param='Rrs'):
        self.raster = raster
        self.Rrs = raster[param]
        self.output = None

    def process(self):
        valid_limit = [0, 2000]
        self.spm_obs2co = self.obs2co(valid_limit=valid_limit)
        #self.spm_obs2co = self.set_range(self.spm_obs2co)
        self.spm_obs2co.name = 'SPM_obs2co'
        self.spm_obs2co.attrs = {
            'description': 'Concentration of suspended particulate matter from bands 665 and 865 nm',
            'applicability': 'general',
            'coef': '',
            'reference': 'OBS2CO',
            'units': 'mg/L',
            'range': valid_limit
        }
        valid_limit = [0, 2000]
        self.turbi_dogliotti = self.turbi_D15(valid_limit=valid_limit)
        #self.turbi_dogliotti = self.set_range(self.turbi_dogliotti)
        self.turbi_dogliotti.name = 'TURB_dogliotti'
        self.turbi_dogliotti.attrs = {
            'description': 'Turbidity in FNU from band 665 nm',
            'applicability': 'general',
            'coef': '',
            'reference': 'Dogliotti et al., 2015',
            'units': 'FNU',
            'range': valid_limit
        }
        valid_limit = [0, 2000]
        self.spm_nechad = self.spm_N10(valid_limit=valid_limit)
        #self.spm_nechad = self.set_range(self.spm_nechad)
        self.spm_nechad.name = 'SPM_nechad'
        self.spm_nechad.attrs = {
            'description': 'Concentration of suspended particulate matter from band 665 nm',
            'applicability': 'low to moderately turbid',
            'coef': '',
            'reference': 'Nechad et al., 2010',
            'units': 'mg/l',
            'range': valid_limit
        }
        self.output = xr.merge([self.spm_obs2co, self.turbi_dogliotti, self.spm_nechad]).drop_vars('wl')
        self.output = self.output.compute(scheduler='processes')

    def set_range(self, param, minval=0, maxval=2000):
        return param.where((param > minval) & (param < maxval))

    def obs2co(self, switch=[0.07, 0.14], coef0=[610.94, 0.2324], coef1=[691.13, 2.5411], valid_limit=[None, None]):
        """ Switching Semi-analytical algorithm to retrieve suspended particulate
        matter (in mg/l) from remote sensing reflectances (Rrs, in sr-1).
        This algorithm was calibrated on GET radiometric database (Morin 2019)
        """
        red, nir = self.Rrs.sel(wl=665), self.Rrs.sel(wl=865)
        spm_high = coef1[0] * (nir / red) ** coef1[1]
        spm_low = self.nechad_relationship(red, coef0)
        w = (red - switch[0]) / (switch[1] - switch[0])
        spm_mixing = (1 - w) * spm_low + w * spm_high

        spm = red.where(red > switch[0], spm_low)
        spm = spm.where(red < switch[1], spm_high)
        spm = spm.where((red <= switch[0]) | (red >= switch[1]), spm_mixing)
        return spm.where((spm >= valid_limit[0]),0).where(spm <= valid_limit[1])

    def turbi_D15(self, switch=[0.05, 0.07],
                  coef_l=[228.1, 0.1641],
                  coef_h=[3078.9, 0.2112],
                  valid_limit=[0, 2000]):
        ''' Switching Semi-analytical algorithm to retrieve Turbidity (in FNU)
        from remote sensing reflectances (Rrs, in sr-1).
        This algorithm was published in Dogliotti et al., 2015
        '''
        red = self.Rrs.sel(wl=665)
        t_low = self.nechad_relationship(red, coef_l)
        t_high = self.nechad_relationship(red, coef_h)
        w = (red - switch[0]) / (switch[1] - switch[0])
        t_mixing = (1 - w) * t_low + w * t_high

        t = red.where(red > switch[0], t_low)
        t = t.where(red < switch[1], t_high)
        t = t.where((red <= switch[0]) | (red >= switch[1]), t_mixing)
        return t.where(t >= valid_limit[0],0).where(t <= valid_limit[1])

    def spm_N10(self, coefs=[342.1, 0.19563], valid_limit=[0, 2000]):
        ''' Semi-analytical algorithm to retrieve suspended particulate matter (in mg/l)
        from remote sensing reflectances (Rrs, in sr-1).
        This algorithm was published in Nechad et al., 2010
        '''
        red = self.Rrs.sel(wl=665)
        spm = self.nechad_relationship(red, coefs)
        return spm.where((spm >= valid_limit[0]),0).where(spm <= valid_limit[1])

    @staticmethod
    def nechad_relationship(Rrs, coefs):
        rho_wl = np.pi * Rrs
        return coefs[0] * rho_wl / (1 - (rho_wl / coefs[1]))