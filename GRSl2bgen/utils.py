''' Utilities: pure water and phytoplankton inherent optical properties, QAA v6. '''

import numpy as np
import pandas as pd
import xarray as xr

from importlib_resources import files

from . import __package__
from .config import SEL_XARRAY_METHOD

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


def qaa_v6_iops(Rrs,
                g0=0.089,
                g1=0.1245,
                Twater=20,
                Salinity=0,
                wls=[443, 490, 560, 665],
                method=SEL_XARRAY_METHOD):
    """QAA v6 steps 0-5: total absorption and backscattering at the `wls` bands.

    Lee, Z.P.; Carder, K.L.; Arnone, R.A. Deriving inherent optical
    properties from water color: a multiband quasi-analytical algorithm for
    optically deep waters. Appl. Opt. 2002, 41, 5755-5772, with the v6
    updates (IOCCG, 2014).

    Below-water reflectance ``rrs = Rrs / (0.52 + 1.7 Rrs)``,
    ``u = (-g0 + sqrt(g0**2 + 4 g1 rrs)) / (2 g1)``, total absorption at the
    reference wavelength (560 nm if ``Rrs665 < 0.0015``, 665 nm otherwise),
    ``bbp`` at the reference wavelength extrapolated with
    ``eta = 2 (1 - 1.2 exp(-0.9 rrs443 / rrs560))`` and
    ``a = (1 - u) (bbw + bbp) / u`` at every band.

    Parameters
    ----------
    Rrs : xarray.DataArray
        Reflectance (sr-1) with a ``wl`` dimension.
    g0, g1 : float, optional
        Coefficients of the ``rrs`` to ``u`` relationship (default 0.089 and
        0.1245).
    Twater : float, optional
        Water temperature (deg C) for the pure water absorption (default 20).
    Salinity : float, optional
        Salinity (PSU) for the pure water absorption (default 0).
    wls : list of float, optional
        Blue, blue-green, green and red wavelengths (nm), default
        ``[443, 490, 560, 665]``; the QAA formulas use them in this order.
    method : str, optional
        Selection method passed to ``Rrs.sel`` (default
        ``config.SEL_XARRAY_METHOD``).

    Returns
    -------
    xarray.Dataset
        With a ``wl`` dimension holding the selected band centres: ``Rrs``
        (non-positive values masked), ``rrs``, ``u``, ``a``, ``bbp`` (m-1),
        and the pure water ``aw`` and ``bbw`` (m-1).
    """

    wl = Rrs.wl.sel(wl=wls, method=method).values
    Rrs = Rrs.sel(wl=wls, method=method).assign_coords(wl=wl)
    # drop the per-band coordinates (e.g. central_wavelength): they would
    # conflict between the bands selected below
    Rrs = Rrs.drop_vars([c for c in Rrs.coords if c != 'wl' and 'wl' in Rrs[c].dims])
    # mask non-positive bands: rrs enters ratios and logarithms below
    Rrs = Rrs.where(Rrs > 0)
    R443, R490, R560, R665 = [Rrs.sel(wl=w) for w in wl]

    aw, bbw = iopw().get_iopw(wl, Twater=Twater, Salinity=Salinity)
    aw = aw.assign_coords(wl=wl)
    bbw = bbw.assign_coords(wl=wl)
    aw560, aw665 = aw.values[2:]
    bbw560, bbw665 = bbw.values[2:]

    # Step 0: above -> below water
    rrs = Rrs / (0.52 + 1.7 * Rrs)
    rrs443, rrs490, rrs560, rrs665 = [rrs.sel(wl=w) for w in wl]

    # Step 1
    u = (-g0 + np.sqrt(g0 ** 2 + 4 * g1 * rrs)) / (2 * g1)
    u560, u665 = u.sel(wl=wl[2]), u.sel(wl=wl[3])

    # Step 2: total absorption at the reference wavelength
    chi = np.log10((rrs443 + rrs490) / (rrs560 + 5 * rrs665 / rrs490 * rrs665))
    a560 = aw560 + 10 ** (-1.146 - 1.366 * chi - 0.469 * chi ** 2)
    a665 = aw665 + 0.39 * (R665 / (R443 + R490)) ** 1.14
    green_ref = R665 < 0.0015

    # Step 3: particulate backscattering at the reference wavelength
    bbp560 = u560 * a560 / (1 - u560) - bbw560
    bbp665 = u665 * a665 / (1 - u665) - bbw665
    bbp0 = xr.where(green_ref, bbp560, bbp665)
    wl0 = xr.where(green_ref, wl[2], wl[3])

    # Steps 4-5: spectral bbp and total absorption
    eta = 2.0 * (1 - 1.2 * np.exp(-0.9 * rrs443 / rrs560))
    bbp = bbp0 * (wl0 / rrs.wl) ** eta
    a = (1 - u) * (bbw + bbp) / u

    return xr.Dataset({'Rrs': Rrs, 'rrs': rrs, 'u': u, 'a': a, 'bbp': bbp,
                       'aw': aw.drop_vars([c for c in aw.coords if c != 'wl']),
                       'bbw': bbw.drop_vars([c for c in bbw.coords if c != 'wl'])})
