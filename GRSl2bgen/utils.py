''' Utilities: pure water and phytoplankton inherent optical properties. '''

import pandas as pd
import xarray as xr

from importlib_resources import files

from . import __package__

data_dir = files(__package__ + '.data')
water_scat_file = data_dir.joinpath('water_coef.txt')
water_abs_file = data_dir.joinpath('pope_fry_1997_water_absorption.csv')
water_abs_temperature_corr_file = data_dir.joinpath('Rottgers_etal_2014_temperature_water_abs_corr.csv')
water_abs_salinity_corr_file = data_dir.joinpath('Rottgers_etal_2014_salinity_water_abs_corr.csv')
aphi_bricaud1998_file = data_dir.joinpath('aph_bricaud_1998.txt')


class iopw:
    '''
    Pure (sea)water inherent optical properties: absorption and
    backscattering coefficients, with temperature and salinity corrections
    of absorption from Röttgers et al. (2014).
    '''
    def __init__(self):
        self.water_abs_file = water_abs_file
        self.water_scat_file = water_scat_file
        self.load_iopw()

    def load_iopw(self, ):
        self.iop_w = pd.read_csv(self.water_scat_file, skiprows=30, sep=' ', header=None,
                                 names=('wl', 'a', 'b')).set_index('wl').to_xarray()

        self.water_PF1997 = pd.read_csv(self.water_abs_file,
                                        skiprows=1).set_index('wl').to_xarray()  # .interp(wl=wl)

        self.water_abs_corr = pd.read_csv(water_abs_temperature_corr_file,
                            skiprows=5, sep=';', index_col=0).to_xarray().rename({'wavelength': 'wl'})
        self.water_abs_corr = xr.merge([self.water_abs_corr,
                                        pd.read_csv(water_abs_salinity_corr_file,
                                        skiprows=5, sep=';', index_col=0).to_xarray().rename({'wavelength': 'wl'})],
                                       compat='override', join='outer')

    def get_iopw(self,
                 wl,
                 Twater=20,
                 Salinity=0,
                 method='linear',
                 mute=False):
        r'''
        Interpolate and return absorption and backscattering coefficients (m-1)
        of pure water, with absorption corrected for temperature and salinity
        (Röttgers et al., 2014):

        .. math::

            a_w(\lambda, T, S) &= a_w(\lambda, T_{ref}) + (T - T_{ref})\, \Psi_T(\lambda)
                                  + S\, \Psi_S(\lambda), \quad T_{ref} = 22^\circ C \\
            b_{bw}(\lambda) &= \frac{b_w(\lambda)}{2}

        :param wl: wavelength in nm, scalar or np.array
        :param Twater: water temperature in deg C
        :param Salinity: salinity (PSU)
        :param method: interpolation method (see xarray.DataArray.interp)
        :param mute: if true values are not returned (only saved in object)
        :return: aw, bbw (m-1)
        '''
        Twater_ref=22

        self.wl = wl

        self.aw = self.iop_w.a.interp(wl=wl,method=method) \
                  + (Twater - Twater_ref)*self.water_abs_corr.PsiT.interp(wl=wl,method=method).fillna(0) \
                  + Salinity * self.water_abs_corr.PsiS_ASW.interp(wl=wl,method=method).fillna(0)
        self.aw.name = 'aw'

        self.bbw = self.iop_w.b.interp(wl=wl,kwargs={"fill_value": 0.0})/ 2.
        self.bbw.name = 'bbw'

        if not mute:
            return self.aw, self.bbw


def aphi_bricaud1998(wl, method='linear'):
    r'''
    Coefficients of the phytoplankton absorption model of Bricaud et al.
    (1998), interpolated on `wl`:

    .. math::

        a_{\phi}(\lambda) = A_{\phi}(\lambda)\, \mathrm{Chl}^{E_{\phi}(\lambda)}

    Bricaud, A.; Morel, A.; Babin, M.; Allali, K.; Claustre, H. Variations of
    light absorption by suspended particles with chlorophyll a concentration in
    oceanic (case 1) waters: analysis and implications for bio-optical models.
    J. Geophys. Res. 1998, 103, 31033-31044. Table from the Polymer auxiliary
    data (400-700 nm, 2 nm step).

    :param wl: wavelength in nm, scalar or np.array
    :param method: interpolation method (see xarray.DataArray.interp)
    :return: Aphi (m-1), Ephi (unitless)
    '''
    table = pd.read_csv(aphi_bricaud1998_file, comment='!', skiprows=12, header=None,
                        names=('wl', 'Ap', 'Ep', 'Aphi', 'Ephi')).set_index('wl').to_xarray()
    table = table.interp(wl=wl, method=method)
    return table.Aphi, table.Ephi
