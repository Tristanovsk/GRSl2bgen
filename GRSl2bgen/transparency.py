"""Water transparency retrieval algorithms: Kd(490), Kd(PAR), euphotic depth and Secchi disk depth."""

import numpy as np
import xarray as xr

from .config import SEL_XARRAY_METHOD
from .utils import qaa_v6_iops

# NASA OBPG KD2 coefficients of log10(Kd490 - Kw490) as a polynomial of
# log10(Rrs_blue / Rrs_green); no MSI set is published, MERIS KD2E (490/560 nm)
# is the closest band pair
KD2E = {'wls': [490, 560],
        'coef': [-0.8641, -1.6549, 2.0112, -2.5174, -1.1035],
        'Kw490': 0.0166}


class Transparency():
    """Water transparency retrieval from remote-sensing reflectance.

    Groups the Roy and Das (2022) Kd(PAR) model, the empirical Kd(490) of
    NASA (KD2), and the products of the QAA v6 inherent optical properties:
    Kd(490) of Lee et al. (2013), Kd(PAR) of Morel et al. (2007) from it,
    euphotic depth and Kd(PAR) of Lee et al. (2005) and the Secchi disk depth
    of Lee et al. (2015).

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
        for the QAA-based products, which are only computed when it is present
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

        Computes ``Kd_par`` (Roy and Das 2022), ``Kd490_KD2`` (NASA KD2) and,
        if the solar zenith angle is available (``self.sza``), the QAA v6
        products: ``Zsd_Lee15`` (Secchi disk depth, Lee et al. 2015),
        ``Kd490_Lee13`` (Lee et al. 2013), ``Kd_par_Mo07`` (Morel et al.
        2007, from ``Kd490_Lee13``), ``Zeu_Lee05`` and ``Kd_par_Lee05``
        (euphotic depth and Kd(PAR) over it, Lee et al. 2005). Masks values
        outside their valid range with `set_range` and attaches metadata
        (lazy for dask input). The boolean
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
            'range': [0, 3000]
        }
        merge_list = [self.Kd_par]

        # Kd(490), NASA KD2
        self.Kd490_KD2 = self.set_range(self.kd490_KD2(), maxval=50)
        self.Kd490_KD2.name = 'Kd490_KD2'
        self.Kd490_KD2.attrs = {
            'description': 'Diffuse attenuation coefficient at 490 nm, empirical blue/green '
                           'band ratio polynomial (NASA KD2, MERIS KD2E coefficients, '
                           '490/560 nm)',
            'applicability': 'open ocean (Case 1) waters',
            'coef': str(KD2E['coef']),
            'reference': 'Mueller, J.L. SeaWiFS algorithm for the diffuse attenuation '
                         'coefficient, K(490), using water-leaving radiances at 490 and 555 nm. '
                         'NASA Tech. Memo. 2000-206892, vol. 11, 2000, 24-27; NASA OBPG Kd '
                         'ATBD, https://oceancolor.gsfc.nasa.gov/resources/atbd/kd/',
            'units': 'm-1',
            'range': [0, 50]
        }
        merge_list.append(self.Kd490_KD2)

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

            # Kd(490), Lee et al. 2013, from the IOPs of the Secchi depth
            self.Kd490 = zsd['Kd'].sel(wl=490, method=SEL_XARRAY_METHOD, drop=True)
            self.Kd490 = self.set_range(self.Kd490, maxval=50)
            self.Kd490.name = 'Kd490_Lee13'
            self.Kd490.attrs = {
                'description': 'Diffuse attenuation coefficient at 490 nm from QAA v6 inherent '
                               'optical properties and the model of Lee et al. (2013), average '
                               'from the surface to the 10 % Ed(490) depth',
                'applicability': 'optically deep waters',
                'reference': 'Lee, Z.; Hu, C.; Shang, S.; Du, K.; Lewis, M.; Arnone, R.; '
                             'Brewin, R. Penetration of UV-visible solar radiation in the global '
                             'oceans: insights from ocean color remote sensing. J. Geophys. Res. '
                             'Oceans 2013, 118, 4241-4255',
                'units': 'm-1',
                'range': [0, 50]
            }
            merge_list.append(self.Kd490)

            # Kd(PAR), Morel et al. 2007, from Kd490_Lee13
            self.Kd_par_Mo07 = self.set_range(self.kd_par_Mo07(self.Kd490), maxval=50)
            self.Kd_par_Mo07.name = 'Kd_par_Mo07'
            self.Kd_par_Mo07.attrs = {
                'description': 'Diffuse attenuation coefficient of PAR over the first optical '
                               'depth, from Kd490_Lee13',
                'applicability': 'open ocean (Case 1) waters',
                'reference': 'Morel, A.; Huot, Y.; Gentili, B.; Werdell, P.J.; Hooker, S.B.; '
                             'Franz, B.A. Examining the consistency of products derived from '
                             'various ocean color sensors in open ocean (Case 1) waters in the '
                             'perspective of a multi-sensor approach. Remote Sens. Environ. '
                             '2007, 111, 69-88',
                'units': 'm-1',
                'range': [0, 50]
            }
            merge_list.append(self.Kd_par_Mo07)

            # euphotic depth and Kd(PAR), Lee et al. 2005
            lee05 = self.zeu_Lee05(zsd['a'].sel(wl=490, method=SEL_XARRAY_METHOD, drop=True),
                                   zsd['bb'].sel(wl=490, method=SEL_XARRAY_METHOD, drop=True),
                                   self.sza)
            reference_lee05 = ('Lee, Z.; Du, K.; Arnone, R.; Liew, S.; Penta, B. Penetration of '
                               'solar radiation in the upper ocean: a numerical model for oceanic '
                               'and coastal waters. J. Geophys. Res. 2005, 110, C09019')
            self.Zeu = self.set_range(lee05['Zeu'], maxval=200)
            self.Zeu.name = 'Zeu_Lee05'
            self.Zeu.attrs = {
                'description': 'Euphotic depth, depth of 1 % of the surface PAR, from QAA v6 '
                               'a(490) and bb(490)',
                'applicability': 'optically deep waters',
                'reference': reference_lee05,
                'units': 'm',
                'range': [0, 200]
            }
            merge_list.append(self.Zeu)
            self.Kd_par_Lee05 = self.set_range(lee05['Kd_par'], maxval=50)
            self.Kd_par_Lee05.name = 'Kd_par_Lee05'
            self.Kd_par_Lee05.attrs = {
                'description': 'Diffuse attenuation coefficient of PAR averaged over the '
                               'euphotic zone, ln(100) / Zeu_Lee05',
                'applicability': 'optically deep waters',
                'reference': reference_lee05,
                'units': 'm-1',
                'range': [0, 50]
            }
            merge_list.append(self.Kd_par_Lee05)

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
            (bool), and ``Kd``, ``a`` (set to ``aw`` where below) and ``bb``
            (m-1) at the `wls` bands; plus ``bottom_limited`` (bool) if
            `bathymetry` is given.
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
                          'Kd': Kd.transpose('wl', ...),
                          'a': a.transpose('wl', ...),
                          'bb': bb.transpose('wl', ...)})
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

    def kd490_KD2(self, coef=KD2E['coef'], wls=KD2E['wls'], Kw490=KD2E['Kw490']):
        """Kd(490) (m-1) of the NASA KD2 algorithm.

        ``Kd490 = Kw490 + 10 ** sum(coef[i] X ** i)`` with
        ``X = log10(Rrs_blue / Rrs_green)``; NaN where a band is not positive.
        Designed for open ocean (Case 1) waters.

        Parameters
        ----------
        coef : list of float, optional
            Polynomial coefficients, default MERIS KD2E.
        wls : list of float, optional
            Blue and green wavelengths (nm), default ``[490, 560]``.
        Kw490 : float, optional
            Pure water Kd(490) (default 0.0166 m-1).
        """
        Rb, Rg = [self.Rrs.sel(wl=w, method=SEL_XARRAY_METHOD, drop=True) for w in wls]
        X = np.log10((Rb / Rg).where((Rb > 0) & (Rg > 0)))
        return Kw490 + 10 ** sum(c * X ** i for i, c in enumerate(coef))

    @staticmethod
    def kd_par_Mo07(Kd490):
        """Kd(PAR) (m-1) over the first optical depth from Kd(490), Morel et al. (2007).

        ``Kd_par = 0.0864 + 0.884 Kd490 - 0.00137 / Kd490``, NaN where
        ``Kd490 <= 0``. Designed for open ocean (Case 1) waters.
        """
        Kd490 = Kd490.where(Kd490 > 0)
        return 0.0864 + 0.884 * Kd490 - 0.00137 / Kd490

    @staticmethod
    def _depth_lee05(K1, K2, tau, n_iter=30, zmin=1e-3, zmax=2e3):
        """Solve ``K1 z + K2 z / sqrt(1 + z) = tau`` for z by bisection in log z (numpy).

        Only the pixels with finite ``K1 > 0`` and ``K2`` are solved, the
        others are NaN, as are the roots outside ]zmin, zmax[. With the
        default 30 iterations, the relative precision on z is about 1e-8.
        """
        K1, K2 = np.broadcast_arrays(np.asarray(K1, dtype=np.float64),
                                     np.asarray(K2, dtype=np.float64))
        out = np.full(K1.shape, np.nan)
        with np.errstate(invalid='ignore'):
            valid = np.isfinite(K1) & np.isfinite(K2) & (K1 > 0)
        k1, k2 = K1[valid], K2[valid]
        lo = np.full(k1.shape, np.log(zmin))
        hi = np.full(k1.shape, np.log(zmax))
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            z = np.exp(mid)
            below = k1 * z + k2 * z / np.sqrt(1 + z) < tau
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        z = np.exp(0.5 * (lo + hi))
        # no root within ]zmin, zmax[
        out[valid] = np.where((z > 1.1 * zmin) & (z < 0.95 * zmax), z, np.nan)
        return out

    @staticmethod
    def zeu_Lee05(a490, bb490, sza, percent=1):
        """Euphotic depth and Kd(PAR) over it, Lee et al. (2005).

        Lee, Z.; Du, K.; Arnone, R.; Liew, S.; Penta, B. Penetration of solar
        radiation in the upper ocean: a numerical model for oceanic and
        coastal waters. J. Geophys. Res. 2005, 110, C09019.

        ``Kpar(z) = K1 + K2 / sqrt(1 + z)``, the average from the surface to
        the depth z, with
        ``K1 = (-0.057 + 0.482 sqrt(a490) + 4.221 bb490) (1 + 0.090 sin(sza))``
        and ``K2 = (0.183 + 0.702 a490 - 2.567 bb490) (1.465 - 0.667 cos(sza))``
        (coefficients and in-air solar zenith angle as in NASA OCSSW). The
        euphotic depth solves ``Kpar(Zeu) Zeu = ln(100 / percent)`` (bisection,
        lazy for dask input), NaN where ``K1 <= 0`` or the root is outside
        1 mm - 2 km.

        Parameters
        ----------
        a490, bb490 : xarray.DataArray
            Total absorption and backscattering at 490 nm (m-1).
        sza : xarray.DataArray or float
            Solar zenith angle (degrees).
        percent : float, optional
            Percentage of the surface PAR defining the euphotic depth
            (default 1).

        Returns
        -------
        xarray.Dataset
            ``Zeu`` (m) and ``Kd_par`` (m-1), ``ln(100 / percent) / Zeu``.
        """
        theta = np.deg2rad(sza)
        K1 = (-0.057 + 0.482 * np.sqrt(a490) + 4.221 * bb490) * (1 + 0.090 * np.sin(theta))
        K2 = (0.183 + 0.702 * a490 - 2.567 * bb490) * (1.465 - 0.667 * np.cos(theta))
        tau = np.log(100 / percent)
        Zeu = xr.apply_ufunc(Transparency._depth_lee05, K1, K2, tau,
                             dask='parallelized', output_dtypes=[np.float64])
        return xr.Dataset({'Zeu': Zeu, 'Kd_par': tau / Zeu})
