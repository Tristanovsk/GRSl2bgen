"""Water transparency retrieval algorithms: Kd(PAR) and Secchi disk depth."""

import numpy as np
import xarray as xr

from .config import SEL_XARRAY_METHOD
from .utils import qaa_v6_iops


class Transparency():
    """Water transparency retrieval from remote-sensing reflectance.

    Groups the Roy and Das (2022) Kd(PAR) model and the Secchi disk depth of
    Lee et al. (2015), computed from the QAA v6 inherent optical properties
    and the Kd model of Lee et al. (2013).

    Parameters
    ----------
    raster : xarray.Dataset
        L2A raster product containing the variable `param`, with dimensions
        ``[y, x, wl]`` (``wl`` = wavelength in nm).
    param : str, optional
        Name of the reflectance variable used as input (default ``'Rrs'``,
        in sr-1).
    sza : str, optional
        Name of the solar zenith angle variable (degrees) of `raster`, used
        for the Secchi disk depth, which is only computed when it is present
        (default ``'sza'``).

    Attributes
    ----------
    raster : xarray.Dataset
        Input raster.
    Rrs : xarray.DataArray
        Reflectance variable ``raster[param]``.
    sza : xarray.DataArray or None
        Solar zenith angle (degrees).
    output : xarray.Dataset or None
        Merged transparency products, set by `process`.
    """

    def __init__(self,
                 raster,
                 param='Rrs',
                 sza='sza'):

        self.raster = raster
        self.Rrs = raster[param]
        self.sza = raster.get(sza)
        self.output = None
        self.flags = {}

    def process(self):
        """Compute the transparency products and store them in ``self.output``.

        Computes ``Kd_par`` (Roy and Das 2022) and, if the solar zenith angle
        is available (``self.sza``), ``Zsd_Lee15`` (Secchi disk depth, Lee et
        al. 2015), masks values outside their valid range with `set_range`
        and attaches metadata (lazy for dask input). The boolean
        ``zsd_a_below_aw`` (``a`` set to ``aw`` at the transparent band) is
        stored in ``self.flags`` for the ``flags_l2b`` bit field.
        """
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
        merge_list = [self.Kd_par]

        # Secchi disk depth, Lee et al., 2015
        if self.sza is not None:
            zsd = self.zsd_Lee15(self.Rrs, self.sza)
            self.Zsd = zsd['Zsd']
            self.Zsd = self.set_range(self.Zsd, maxval=100)
            self.Zsd.name = 'Zsd_Lee15'
            self.Zsd.attrs = {
                'description': 'Secchi disk depth from the diffuse attenuation coefficient and Rrs '
                               'at the most transparent band (443-665 nm); Kd from QAA v6 inherent '
                               'optical properties and the model of Lee et al. (2013)',
                'applicability': 'optically deep waters',
                'reference': 'Lee, Z.; Shang, S.; Hu, C.; Du, K.; Weidemann, A.; Hou, W.; Lin, J.; '
                             'Lin, G. Secchi disk depth: a new theory and mechanistic model for '
                             'underwater visibility. Remote Sens. Environ. 2015, 169, 139-149; '
                             'Lee, Z.; Hu, C.; Shang, S.; Du, K.; Lewis, M.; Arnone, R.; Brewin, R. '
                             'Penetration of UV-visible solar radiation in the global oceans: '
                             'insights from ocean color remote sensing. J. Geophys. Res. Oceans '
                             '2013, 118, 4241-4255',
                'units': 'm',
                'range': [0, 100]
            }
            merge_list.append(self.Zsd)
            merge_list.append(zsd['a_below_aw'].rename('zsd_a_below_aw'))

        self.output = xr.merge(merge_list, compat='no_conflicts', join='outer')
        # boolean flags go to the ``flags_l2b`` bit field (see mask.L2B_FLAGS)
        self.flags = {name: self.output[name] for name in ['zsd_a_below_aw'] if name in self.output}
        self.output = self.output.drop_vars(list(self.flags))

        return

    def set_range(self,param,minval=0,maxval=3000):
        return param.where((param>minval)&(param<maxval) )

    def zsd_Lee15(self,
                  Rrs,
                  sza,
                  bathymetry=None,
                  Twater=20,
                  Salinity=0,
                  wls=[443, 490, 560, 665],
                  method=SEL_XARRAY_METHOD,
                  mcoef=[0.005, 4.259, 0.52, 10.8],
                  gamma=0.265):
        """Secchi disk depth of Lee et al. (2015) from QAA v6 IOPs.

        Lee, Z.; Shang, S.; Hu, C.; Du, K.; Weidemann, A.; Hou, W.; Lin, J.;
        Lin, G. Secchi disk depth: a new theory and mechanistic model for
        underwater visibility. Remote Sens. Environ. 2015, 169, 139-149.

        Step A: total absorption ``a`` and backscattering
        ``bb = bbw + bbp`` from QAA v6 (see `GRSl2bgen.utils.qaa_v6_iops`);
        ``a`` is set to the pure water absorption ``aw`` where ``a < aw`` (in
        clear water, with the 560 nm QAA reference, ``a665`` is extrapolated
        and often falls just below ``aw665``).

        Step B: diffuse attenuation at each band (Lee et al. 2013),
        ``Kd = (1 + m0 sza) a + (1 - gamma bbw / bb) m1 (1 - m2 exp(-m3 a)) bb``
        with the solar zenith angle ``sza`` in degrees.

        Step C: transparent window, the band of minimum ``Kd`` among `wls`,
        with ``Kd_tr`` and ``Rrs_tr`` at that band.

        Step D: ``Zsd = 1 / (2.5 Kd_tr) ln(|0.14 - Rrs_tr| / 0.013)``.

        Step E: NaN where ``Kd_tr <= 0`` or ``Zsd <= 0``; ``a_below_aw``
        marks the valid pixels where ``a`` was set to ``aw`` at the
        transparent band.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance (sr-1) with a ``wl`` dimension.
        sza : xarray.DataArray or float
            Solar zenith angle (degrees).
        bathymetry : xarray.DataArray or float, optional
            Water depth (m, positive). If given, ``bottom_limited`` flags the
            pixels where ``Zsd`` exceeds it (optically shallow water).
        Twater : float, optional
            Water temperature (deg C) for the pure water IOPs (default 20).
        Salinity : float, optional
            Salinity (PSU) for the pure water IOPs (default 0).
        wls : list of float, optional
            Blue, blue-green, green and red wavelengths (nm) used by QAA and
            searched for the transparent window, default
            ``[443, 490, 560, 665]``.
        method : str, optional
            Selection method passed to ``Rrs.sel`` (default
            ``config.SEL_XARRAY_METHOD``).
        mcoef : list of float, optional
            ``[m0, m1, m2, m3]`` of the Kd model, default
            ``[0.005, 4.259, 0.52, 10.8]``.
        gamma : float, optional
            ``gamma`` of the Kd model (default 0.265).

        Returns
        -------
        xarray.Dataset
            ``Zsd`` (m), ``wl_tr`` (nm), ``Kd_tr`` (m-1), ``a_below_aw``
            (bool) and ``Kd`` (m-1) at the `wls` bands; plus ``bottom_limited`` (bool) if `bathymetry`
            is given.
        """

        m0, m1, m2, m3 = mcoef

        # Step A: QAA v6 IOPs
        iops = qaa_v6_iops(Rrs, Twater=Twater, Salinity=Salinity, wls=wls, method=method)
        a, bbw = iops.a, iops.bbw
        below_aw = a < iops.aw
        a = a.where(~below_aw, iops.aw)
        bb = bbw + iops.bbp

        # Step B: Kd at each band (Lee et al. 2013)
        Kd = (1 + m0 * sza) * a \
             + (1 - gamma * bbw / bb) * m1 * (1 - m2 * np.exp(-m3 * a)) * bb

        # Step C: transparent window; NaN filled with +/-inf so that all-NaN
        # pixels do not raise warnings in the min/max reductions
        Kd_tr = Kd.fillna(np.inf).min('wl')
        Kd_tr = Kd_tr.where(np.isfinite(Kd_tr))
        is_tr = Kd == Kd_tr
        wl_tr = iops.wl.where(is_tr, -np.inf).max('wl')
        wl_tr = wl_tr.where(np.isfinite(wl_tr))
        Rrs_tr = iops.Rrs.where(is_tr).fillna(-np.inf).max('wl')
        Rrs_tr = Rrs_tr.where(np.isfinite(Rrs_tr))

        # Step D: Secchi depth
        arg = np.abs(0.14 - Rrs_tr) / 0.013
        Zsd = 1 / (2.5 * Kd_tr) * np.log(arg.where(arg > 0))

        # Step E: QC
        valid = (Kd_tr > 0) & (Zsd > 0)
        out = xr.Dataset({'Zsd': Zsd.where(valid),
                          'wl_tr': wl_tr.where(valid),
                          'Kd_tr': Kd_tr.where(valid),
                          'a_below_aw': below_aw.where(is_tr, False).any('wl') & valid,
                          'Kd': Kd.transpose('wl', ...)})
        if bathymetry is not None:
            out['bottom_limited'] = out.Zsd > bathymetry
        return out

    def Kd_par_RD22(self,acoef=[3.09,-90.17]):
        '''
        Kd_par from Roy and Das, 2022, Journal of Hydrology, table 3, eq. 13
        :return:
        '''
        R490 = self.Rrs.sel(wl=490, method=SEL_XARRAY_METHOD)
        R665 = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD)
        return acoef[0] * np.exp(acoef[1] * (R490 - R665))
