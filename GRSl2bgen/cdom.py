"""Coloured dissolved organic matter (CDOM) retrieval algorithms."""

import numpy as np
import xarray as xr

from .utils import iopw, aphi_bricaud1998


class Cdom():
    """CDOM absorption retrieval from remote-sensing reflectance.

    Groups the Brezonik et al. (2015) band-ratio model and a QAA (v6)
    decomposition of the total absorption into detrital plus CDOM absorption
    (``adg``) and phytoplankton absorption (``aph``), adapted to the MSI
    bands.

    Parameters
    ----------
    raster : xarray.Dataset
        L2A raster product containing the variable `param`, with dimensions
        ``[y, x, wl]`` (``wl`` = wavelength in nm).
    param : str, optional
        Name of the reflectance variable used as input (default ``'Rrs'``,
        in sr-1).
    chl : xarray.DataArray, optional
        First-guess Chl-a (mg m-3), dimensions ``[y, x]``, e.g. the
        OWT-blended ``Chla_OWTblend_Ta2025``. Used for the QAA product, which
        is only computed when it is given.

    Attributes
    ----------
    raster : xarray.Dataset
        Input raster.
    Rrs : xarray.DataArray
        Reflectance variable ``raster[param]``.
    chl : xarray.DataArray or None
        First-guess Chl-a.
    output : xarray.Dataset or None
        Merged CDOM products, set by `process`.
    """

    def __init__(self,
                 raster,
                 param='Rrs',
                 chl=None):

        self.raster = raster
        self.Rrs = raster[param]
        self.chl = chl
        self.output = None

    def process(self):
        """Compute the CDOM products and store them in ``self.output``.

        Computes ``acdom_B15`` (Brezonik et al. 2015, 490/740 ratio) and, if
        a first-guess Chl-a is available (``self.chl``), ``adg443_QAAv6``
        (QAA v6 adapted to MSI, 443/490 decomposition), masks values outside
        0-30 m-1 with `set_range` and attaches metadata.
        """

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

        merge_list = [acdom_B15]
        if self.chl is None:
            self.output = xr.merge(merge_list, compat='no_conflicts', join='outer')
            return

        # QAA v6 adapted to MSI
        adg443 = self.a_dg443_QAA_v6(self.Rrs, self.chl)['adg443']
        adg443 = self.set_range(adg443)
        adg443.name = 'adg443_QAAv6'
        adg443.attrs = {
            'description': 'Detrital and CDOM absorption at 443 nm from QAA v6 adapted to MSI, '
                           'bands 443, 490, 560, 665 nm; a(443) and a(490) decomposed with '
                           'zeta = aph(443)/aph(490) from Bricaud et al. (1998) with a first-guess '
                           'Chl-a (' + str(self.chl.name) + '), and xi = adg(443)/adg(490)',
            'applicability': 'general',
            'reference': 'Lee, Z.P.; Carder, K.L.; Arnone, R.A. Deriving inherent optical properties '
                         'from water color: a multiband quasi-analytical algorithm for optically deep '
                         'waters. Appl. Opt. 2002, 41, 5755-5772; QAA v6 (IOCCG, 2014); '
                         'Bricaud, A.; Morel, A.; Babin, M.; Allali, K.; Claustre, H. '
                         'J. Geophys. Res. 1998, 103, 31033-31044',
            'units': 'm-1',
            'range': [0, 30]}

        merge_list.append(adg443)
        self.output = xr.merge(merge_list, compat='no_conflicts', join='outer')

    def set_range(self,param,minval=0,maxval=30):
        return param.where((param>minval)&(param<maxval) )

    def brezonik15(self, acoef = [1.872, -0.83]):
        R490, R740 = self.Rrs.sel(wl=490), self.Rrs.sel(wl=740)
        # mask non-positive bands, not the ratio: two negative bands give a
        # positive but meaningless ratio
        ratio = R490.where(R490 > 0) / R740.where(R740 > 0)
        return np.exp(acoef[0] + acoef[1] * np.log(ratio))

    def a_dg443_QAA_v6(self,
                       Rrs,
                       chl,
                       g0=0.089,
                       g1=0.1245,
                       Twater=20,
                       Salinity=0,
                       wls=[443, 490, 560, 665],
                       method='nearest',
                       eps=1e-3):
        """QAA v6 adapted to MSI: ``adg`` and ``aph`` at 443 nm.

        Lee, Z.P.; Carder, K.L.; Arnone, R.A. Deriving inherent optical
        properties from water color: a multiband quasi-analytical algorithm
        for optically deep waters. Appl. Opt. 2002, 41, 5755-5772, with the
        v6 updates (IOCCG, 2014).

        Steps 0-5 follow QAA v6: below-water reflectance
        ``rrs = Rrs / (0.52 + 1.7 Rrs)``,
        ``u = (-g0 + sqrt(g0**2 + 4 g1 rrs)) / (2 g1)``, total absorption at
        the reference wavelength (560 nm if ``Rrs665 < 0.0015``, 665 nm
        otherwise), ``bbp`` at the reference wavelength extrapolated with
        ``eta = 2 (1 - 1.2 exp(-0.9 rrs443 / rrs560))`` and
        ``a = (1 - u) (bbw + bbp) / u`` at every band.

        Step 6 is modified for MSI: ``a - aw`` at 443 and 490 nm (instead of
        412 and 443 nm) is decomposed with
        ``xi = adg(443) / adg(490) = exp(S (490 - 443))``,
        ``S = 0.015 + 0.002 / (0.6 + rrs443 / rrs560)``, and
        ``zeta = aph(443) / aph(490)`` from the Bricaud et al. (1998) model
        ``aph = Aphi Chl**Ephi``, i.e.
        ``zeta = Aphi443 / Aphi490 * chl**(Ephi443 - Ephi490)``:
        ``adg443 = ((a443 - aw443) - zeta (a490 - aw490)) / (1 - zeta / xi)``
        and ``aph443 = (a443 - aw443) - adg443``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance (sr-1) with a ``wl`` dimension.
        chl : xarray.DataArray or float
            First-guess Chl-a (mg m-3) for ``zeta``, e.g. from a red/NIR
            algorithm; pixels with ``chl <= 0`` or NaN are NaN.
        g0, g1 : float, optional
            QAA v6 coefficients of the ``rrs`` to ``u`` relationship (default
            0.089 and 0.1245).
        Twater : float, optional
            Water temperature (deg C) for the pure water absorption
            (default 20).
        Salinity : float, optional
            Salinity (PSU) for the pure water absorption (default 0).
        wls : list of float, optional
            Blue, blue-green, green and red wavelengths (nm), default
            ``[443, 490, 560, 665]``.
        method : str, optional
            Selection method passed to ``Rrs.sel`` (default ``'nearest'``).
        eps : float, optional
            Pixels with ``|1 - zeta / xi| < eps`` (ill-conditioned
            decomposition) are set to NaN.

        Returns
        -------
        xarray.Dataset
            ``adg443`` and ``aph443`` (m-1), NaN where ``adg443 <= 0`` or
            ``aph443 < 0``, and ``bbp`` (m-1) at the ``wls`` bands.
        """

        wl = Rrs.wl.sel(wl=wls, method=method).values
        Rrs = Rrs.sel(wl=wls, method=method).assign_coords(wl=wl)
        # mask non-positive bands: rrs enters ratios and logarithms below
        Rrs = Rrs.where(Rrs > 0)
        R443, R490, R560, R665 = [Rrs.sel(wl=w) for w in wl]

        aw, bbw = iopw().get_iopw(wl, Twater=Twater, Salinity=Salinity)
        aw = aw.assign_coords(wl=wl)
        bbw = bbw.assign_coords(wl=wl)
        aw443, aw490, aw560, aw665 = aw.values
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

        # Step 6 (modified): decompose using 443 / 490 instead of 412 / 443
        Aphi, Ephi = aphi_bricaud1998(wl[:2])
        chl = xr.DataArray(chl) if np.isscalar(chl) else chl
        zeta = Aphi.values[0] / Aphi.values[1] * chl.where(chl > 0) ** (Ephi.values[0] - Ephi.values[1])
        S = 0.015 + 0.002 / (0.6 + rrs443 / rrs560)
        xi = np.exp(S * (wl[1] - wl[0]))
        denom = 1 - zeta / xi
        a443, a490 = a.sel(wl=wl[0]), a.sel(wl=wl[1])
        adg443 = ((a443 - aw443) - zeta * (a490 - aw490)) / denom.where(np.abs(denom) >= eps)
        aph443 = (a443 - aw443) - adg443

        # Step 7: QC
        valid = (adg443 > 0) & (aph443 >= 0)
        return xr.Dataset({'adg443': adg443.where(valid),
                           'aph443': aph443.where(valid),
                           'bbp': bbp.transpose('wl', ...)})
