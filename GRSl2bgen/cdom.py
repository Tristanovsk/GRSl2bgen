"""Coloured dissolved organic matter (CDOM) retrieval algorithms."""

import numpy as np
import xarray as xr

from .utils import aphi_bricaud1998, qaa_v6_iops
from .config import SEL_XARRAY_METHOD


class Cdom():
    """CDOM absorption retrieval from remote-sensing reflectance.

    Groups the Brezonik et al. (2015) band-ratio model and a QAA (v6)
    decomposition of the total absorption into detrital plus CDOM absorption
    (``adg``) and phytoplankton absorption (``aph``), adapted to the MSI
    bands, either by the QAA 443/490 decomposition or with ``aph`` computed
    from the Chl-a.

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
        OWT-blended ``Chla_OWTblend_Ta2025``. Used for the QAA products, which
        are only computed when it is given.

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
        (QAA v6 adapted to MSI, 443/490 decomposition) and ``adg443_QAAv6Chl``
        (QAA v6 minus the phytoplankton absorption from the Chl-a), masks
        values outside 0-30 m-1 with `set_range` and attaches metadata.
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
            'range': [0, 30]}

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

        # QAA v6 total absorption minus Bricaud (1998) phytoplankton absorption
        adg443_chl = self.a_dg443_QAA_v6_chl(self.Rrs, self.chl)['adg443']
        adg443_chl = self.set_range(adg443_chl)
        adg443_chl.name = 'adg443_QAAv6Chl'
        adg443_chl.attrs = {
            'description': 'Detrital and CDOM absorption at 443 nm: QAA v6 a(443) - aw(443) minus the '
                           'phytoplankton absorption aph(443) of Bricaud et al. (1998) computed from '
                           'the Chl-a (' + str(self.chl.name) + ')',
            'applicability': 'general',
            'reference': 'Lee, Z.P.; Carder, K.L.; Arnone, R.A. Deriving inherent optical properties '
                         'from water color: a multiband quasi-analytical algorithm for optically deep '
                         'waters. Appl. Opt. 2002, 41, 5755-5772; QAA v6 (IOCCG, 2014); '
                         'Bricaud, A.; Morel, A.; Babin, M.; Allali, K.; Claustre, H. '
                         'J. Geophys. Res. 1998, 103, 31033-31044',
            'units': 'm-1',
            'range': [0, 30]}

        merge_list.append(adg443_chl)
        self.output = xr.merge(merge_list, compat='no_conflicts', join='outer')

    def set_range(self,param,minval=0,maxval=30):
        return param.where((param>minval)&(param<maxval) )

    def brezonik15(self, acoef = [1.872, -0.83]):
        R490 = self.Rrs.sel(wl=490, method=SEL_XARRAY_METHOD)
        R740 = self.Rrs.sel(wl=740, method=SEL_XARRAY_METHOD)
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
                       method=SEL_XARRAY_METHOD,
                       eps=1e-3):
        """QAA v6 adapted to MSI: ``adg`` and ``aph`` at 443 nm.

        Lee, Z.P.; Carder, K.L.; Arnone, R.A. Deriving inherent optical
        properties from water color: a multiband quasi-analytical algorithm
        for optically deep waters. Appl. Opt. 2002, 41, 5755-5772, with the
        v6 updates (IOCCG, 2014).

        Steps 0-5 follow QAA v6 (see `GRSl2bgen.utils.qaa_v6_iops`): total
        absorption ``a`` and particulate backscattering ``bbp`` at the
        ``wls`` bands.

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
            Selection method passed to ``Rrs.sel`` (default
            ``config.SEL_XARRAY_METHOD``).
        eps : float, optional
            Pixels with ``|1 - zeta / xi| < eps`` (ill-conditioned
            decomposition) are set to NaN.

        Returns
        -------
        xarray.Dataset
            ``adg443`` and ``aph443`` (m-1), NaN where ``adg443 <= 0`` or
            ``aph443 < 0``, and ``bbp`` (m-1) at the ``wls`` bands.
        """

        # Steps 0-5: QAA v6 total absorption and backscattering
        iops = qaa_v6_iops(Rrs, g0=g0, g1=g1, Twater=Twater, Salinity=Salinity,
                           wls=wls, method=method)
        wl = iops.wl.values
        a, aw, bbp = iops.a, iops.aw, iops.bbp
        aw443, aw490 = aw.values[:2]
        rrs443, rrs560 = iops.rrs.sel(wl=wl[0]), iops.rrs.sel(wl=wl[2])

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

    def a_dg443_QAA_v6_chl(self,
                           Rrs,
                           chl,
                           g0=0.089,
                           g1=0.1245,
                           Twater=20,
                           Salinity=0,
                           wls=[443, 490, 560, 665],
                           method=SEL_XARRAY_METHOD):
        """``adg`` at 443 nm from the QAA v6 total absorption and the Chl-a.

        Steps 0-5 follow QAA v6 (see `GRSl2bgen.utils.qaa_v6_iops`): total
        absorption ``a`` at 443 nm. The phytoplankton absorption is not
        obtained by the 443/490 decomposition of `a_dg443_QAA_v6`, very
        sensitive to errors in Rrs(443) (amplified by ``1 / (1 - zeta / xi)``,
        about 3), but from the Chl-a with the Bricaud et al. (1998) model:
        ``aph443 = Aphi443 * chl**Ephi443`` and
        ``adg443 = (a443 - aw443) - aph443``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance (sr-1) with a ``wl`` dimension.
        chl : xarray.DataArray or float
            Chl-a (mg m-3); pixels with ``chl <= 0`` or NaN are NaN.
        g0, g1 : float, optional
            QAA v6 coefficients of the ``rrs`` to ``u`` relationship (default
            0.089 and 0.1245).
        Twater : float, optional
            Water temperature (deg C) for the pure water absorption
            (default 20).
        Salinity : float, optional
            Salinity (PSU) for the pure water absorption (default 0).
        wls : list of float, optional
            Blue, blue-green, green and red wavelengths (nm) used by QAA,
            default ``[443, 490, 560, 665]``.
        method : str, optional
            Selection method passed to ``Rrs.sel`` (default
            ``config.SEL_XARRAY_METHOD``).

        Returns
        -------
        xarray.Dataset
            ``adg443`` and ``aph443`` (m-1), NaN where ``adg443 <= 0``.
        """
        iops = qaa_v6_iops(Rrs, g0=g0, g1=g1, Twater=Twater, Salinity=Salinity,
                           wls=wls, method=method)
        wl0 = iops.wl.values[0]
        Aphi, Ephi = aphi_bricaud1998(wl0)
        chl = xr.DataArray(chl) if np.isscalar(chl) else chl
        aph443 = float(Aphi) * chl.where(chl > 0) ** float(Ephi)
        adg443 = iops.a.sel(wl=wl0) - iops.aw.sel(wl=wl0) - aph443

        valid = adg443 > 0
        return xr.Dataset({'adg443': adg443.where(valid).drop_vars('wl', errors='ignore'),
                           'aph443': aph443.where(valid)})
