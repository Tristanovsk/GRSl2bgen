import json
import os
import numpy as np
import xarray as xr
from importlib_resources import files
from . import __package__
from .config import SEL_XARRAY_METHOD


class Spm():
    def __init__(self,
                 raster,
                 param='Rrs',
                 xowt_prod=None,
                 Nclasses=2):
        self.raster = raster
        self.Rrs = raster[param]
        self.xowt_prod = xowt_prod
        self.Nclasses = Nclasses
        self.output = None

    def process(self):
        valid_limit = [0, 2000]
        self.spm_obs2co = self.obs2co(valid_limit=valid_limit)
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
        self.spm_nechad.name = 'SPM_nechad'
        self.spm_nechad.attrs = {
            'description': 'Concentration of suspended particulate matter from band 665 nm',
            'applicability': 'low to moderately turbid',
            'coef': '',
            'reference': 'Nechad et al., 2010',
            'units': 'mg/l',
            'range': valid_limit
        }

        merge_list = [self.spm_obs2co, self.turbi_dogliotti, self.spm_nechad]
        if self.xowt_prod is not None:
            spm_blend = self.spm_blending(
                self.xowt_prod,
                Nclasses=self.Nclasses,
                owt_dist_var='owt_dist_Cordeiro2022SPM',
                owt_index_var='owt_index_Cordeiro2022SPM',
            )
            merge_list.append(self.set_range(spm_blend))

        # join: current xarray default, set explicitly as it will change
        self.output = xr.merge(merge_list, compat='override', join='outer').drop_vars('wl')
        #self.output = self.output.compute(scheduler='processes')

    # ------------------------------------------------------------------
    # Existing algorithms (unchanged)
    # ------------------------------------------------------------------

    def set_range(self, param, minval=0, maxval=2000):
        return param.where((param > minval) & (param < maxval))

    def obs2co(self, switch=[0.07, 0.14], coef0=[610.94, 0.2324], coef1=[691.13, 2.5411], valid_limit=[None, None]):
        """ Switching Semi-analytical algorithm to retrieve suspended particulate
        matter (in mg/l) from remote sensing reflectances (Rrs, in sr-1).
        This algorithm was calibrated on GET radiometric database (Morin 2019)
        """
        red = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD)
        nir = self.Rrs.sel(wl=865, method=SEL_XARRAY_METHOD)
        # mask negative bands, not the ratio: two negative bands give a
        # positive but meaningless ratio
        ratio = nir.where(nir >= 0) / red.where(red > 0)
        spm_high = coef1[0] * ratio ** coef1[1]
        spm_low = self.nechad_relationship(red, coef0)
        w = (red - switch[0]) / (switch[1] - switch[0])
        spm_mixing = (1 - w) * spm_low + w * spm_high

        spm = red.where(red > switch[0], spm_low)
        spm = spm.where(red < switch[1], spm_high)
        spm = spm.where((red <= switch[0]) | (red >= switch[1]), spm_mixing)
        return spm.where((spm >= valid_limit[0]), 0).where(spm <= valid_limit[1])

    def turbi_D15(self, switch=[0.05, 0.07],
                  coef_l=[228.1, 0.1641],
                  coef_h=[3078.9, 0.2112],
                  valid_limit=[0, 2000]):
        ''' Switching Semi-analytical algorithm to retrieve Turbidity (in FNU)
        from remote sensing reflectances (Rrs, in sr-1).
        This algorithm was published in Dogliotti et al., 2015
        The switch thresholds apply to the water-leaving reflectance
        rho_w = pi * Rrs, as in the original paper.
        '''
        red = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD)
        rho_red = np.pi * red
        t_low = self.nechad_relationship(red, coef_l)
        t_high = self.nechad_relationship(red, coef_h)
        w = (rho_red - switch[0]) / (switch[1] - switch[0])
        t_mixing = (1 - w) * t_low + w * t_high

        t = red.where(rho_red > switch[0], t_low)
        t = t.where(rho_red < switch[1], t_high)
        t = t.where((rho_red <= switch[0]) | (rho_red >= switch[1]), t_mixing)
        return t.where(t >= valid_limit[0], 0).where(t <= valid_limit[1])

    def spm_N10(self, coefs=[342.1, 0.19563], valid_limit=[0, 2000]):
        ''' Semi-analytical algorithm to retrieve suspended particulate matter (in mg/l)
        from remote sensing reflectances (Rrs, in sr-1).
        This algorithm was published in Nechad et al., 2010
        '''
        red = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD)
        spm = self.nechad_relationship(red, coefs)
        return spm.where((spm >= valid_limit[0]), 0).where(spm <= valid_limit[1])

    @staticmethod
    def nechad_relationship(Rrs, coefs):
        rho_wl = np.pi * Rrs
        return coefs[0] * rho_wl / (1 - (rho_wl / coefs[1]))

    # ------------------------------------------------------------------
    # OWT-blended SPM (Tavares et al. 2025 / Cordeiro 2022)
    # ------------------------------------------------------------------

    # @staticmethod
    # def compute_spm_owt_distances(Rrs):
    #     """Per-pixel Euclidean distance to each of the 4 Cordeiro2022 SPM OWT mean spectra.
    #     Returns dist (4, H, W) float32. NaN where any of the 9 bands is NaN.
    #     """
    #     owt_means = _load_spm_owt_means()   # (4, 9)
    #     # sel on a list preserves the wl dim -> (9, H, W); one dask compute for all 9 bands
    #     rrs_stack = Rrs.sel(wl=_SPM_OWT_WLS).values.astype(np.float32)  # (9, H, W)
    #     nan_mask  = ~np.all(np.isfinite(rrs_stack), axis=0)              # (H, W)
    #     H, W = rrs_stack.shape[1], rrs_stack.shape[2]
    #     dist = np.full((4, H, W), np.nan, dtype=np.float32)
    #     for k in range(4):
    #         diff = rrs_stack - owt_means[k, :, None, None]
    #         dist[k] = np.where(nan_mask, np.nan, np.sqrt(np.sum(diff ** 2, axis=0)))
    #     return dist  # (4, H, W)

    # @staticmethod
    # def compute_owt_weights(dist_np, n=2):
    #     """Top-n OWT classes and blending weights from an (N, H, W) Euclidean distance array.
    #     Smaller distance = better match. Same formula as Chl.compute_owt_weights.
    #     Returns owt_classes (n, H, W) float32 (1-based) and owt_weights (n, H, W) float32.
    #     """
    #     N, H, W     = dist_np.shape
    #     sorted_idx  = np.argsort(dist_np, axis=0)
    #     sorted_dist = np.take_along_axis(dist_np, sorted_idx, axis=0)
    #     top_dist    = sorted_dist[:n + 1]    # (n+1, H, W)
    #     top_idx     = sorted_idx[:n]         # (n, H, W), 0-based
    #     d_best      = top_dist[0]
    #     d_ref       = top_dist[n]            # (n+1)-th as reference
    #     denom       = d_best - d_ref
    #     owt_classes = (top_idx + 1).astype(np.float32)
    #     owt_weights = np.full((n, H, W), np.nan, dtype=np.float32)
    #     valid       = np.isfinite(denom) & (denom != 0)
    #     for i in range(n):
    #         owt_weights[i] = np.where(
    #             valid, (top_dist[i] - d_ref) / denom, np.nan
    #         ).astype(np.float32)
    #     return owt_classes, owt_weights

    @staticmethod
    def _spm_jiang_green(R443, R490, R560, R665):
        """Jiang et al. 2021, QAA at 560 nm (OWT 1, clear)."""
        rrs443 = R443 / (0.52 + 1.7 * R443)
        rrs490 = R490 / (0.52 + 1.7 * R490)
        rrs560 = R560 / (0.52 + 1.7 * R560)
        rrs665 = R665 / (0.52 + 1.7 * R665)
        # mask a negative sqrt argument (rrs560 < -0.0158): NaN anyway, masking
        # first avoids the 'invalid value in sqrt' warning
        delta = 0.089 ** 2 + 4 * 0.125 * rrs560
        u560 = (-0.0895 + np.sqrt(delta.where(delta >= 0))) / (2 * 0.125)
        x = (rrs443 + rrs490) / (rrs560 + 5 * rrs665 ** 2 / rrs490)
        x = np.log10(x.where(x > 0))
        a560 = 0.06299986 + 10 ** (-1.146 - 1.366 * x - 0.469 * x ** 2)
        bbp = (u560 * a560) / (1 - u560) - 0.00078491
        return (94.48785 * bbp).astype(np.float32)

    @staticmethod
    def _spm_jiang_red(R443, R490, R560, R665):
        """Jiang et al. 2021, QAA at 665 nm (OWT 2, moderate)."""
        rrs665 = R665 / (0.52 + 1.7 * R665)
        # mask a negative sqrt argument (rrs665 < -0.0158): NaN anyway, masking
        # first avoids the 'invalid value in sqrt' warning
        delta = 0.089 ** 2 + 4 * 0.125 * rrs665
        u665 = (-0.0895 + np.sqrt(delta.where(delta >= 0))) / (2 * 0.125)
        # mask negative bands, not the ratio: two negative bands give a
        # positive but meaningless ratio
        blue = R443 + R490
        ratio = R665.where(R665 >= 0) / blue.where(blue > 0)
        a665 = 0.41395333 + 0.39 * ratio ** 1.14
        bbp = (u665 * a665) / (1 - u665) - 0.00037474
        return (113.87498 * bbp).astype(np.float32)

    @staticmethod
    def _spm_zhang2014(R705, acoef=None):
        """Zhang et al. 2014, power law at 705 nm (OWT 3, turbid)."""
        if acoef is None:
            acoef = [362507.0, 2.3222]
        a, b = acoef
        return (a * R705.where(R705 >= 0) ** b).astype(np.float32)

    @staticmethod
    def _spm_binding2010(R740):
        """Binding et al. 2010, semi-analytical at 740 nm (OWT 4, very turbid)."""
        rho   = R740 * np.pi
        scale = 51.162 / (0.554 * 0.019)           # ~4859.9
        denom = (0.54 * 128.123 / np.pi) + 0.48 * rho
        return (scale * (2.8 * rho / denom - 0.00027)).astype(np.float32)

    # def _owt_spm_from_class(self, class_px, limits=True):
    #     """Dispatch 1-based OWT class map (H, W) to the appropriate SPM algorithm."""
    #     R443 = self.Rrs.sel(wl=443, method=SEL_XARRAY_METHOD).values.squeeze()
    #     R490 = self.Rrs.sel(wl=490, method=SEL_XARRAY_METHOD).values.squeeze()
    #     R560 = self.Rrs.sel(wl=560, method=SEL_XARRAY_METHOD).values.squeeze()
    #     R665 = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD).values.squeeze()
    #     R705 = self.Rrs.sel(wl=705, method=SEL_XARRAY_METHOD).values.squeeze()
    #     R740 = self.Rrs.sel(wl=740, method=SEL_XARRAY_METHOD).values.squeeze()
    #     H, W = R665.shape
    #     spm  = np.full((H, W), np.nan, dtype=np.float32)

    #     mask = class_px == 1
    #     if mask.any():
    #         val = self._spm_jiang_green(R443, R490, R560, R665)
    #         if limits:
    #             val = np.where((val >= 0) & (val <= 50), val, np.nan)
    #         spm = np.where(mask, val, spm)

    #     mask = class_px == 2
    #     if mask.any():
    #         val = self._spm_jiang_red(R443, R490, R560, R665)
    #         if limits:
    #             val = np.where((val >= 10) & (val <= 500), val, np.nan)
    #         spm = np.where(mask, val, spm)

    #     mask = class_px == 3
    #     if mask.any():
    #         val = self._spm_zhang2014(R705)
    #         if limits:
    #             val = np.where((val >= 20) & (val <= 1000), val, np.nan)
    #         spm = np.where(mask, val, spm)

    #     mask = class_px == 4
    #     if mask.any():
    #         val = self._spm_binding2010(R740)
    #         if limits:
    #             val = np.where((val >= 50) & (val <= 2000), val, np.nan)
    #         spm = np.where(mask, val, spm)

    #     return spm.astype(np.float32)

    def spm_blending(self,
                     xowt_prod,
                     Nclasses=2,
                     owt_dist_var='owt_dist',
                     owt_index_var='owt_index',
                     eps=1e-6,
                     limits=True):
        """Blend OWT-classified SPM estimates using the ranked OWT distances.

        This follows :meth:`GRSl2bgen.chlorophyll_a.Chl.owt_blending`: the
        best ``Nclasses`` ranked OWT matches are selected, each estimate is
        weighted by ``1 / (distance + eps)``, and the weighted sum is divided
        by the sum of the weights of the valid (not NaN, in range) estimates,
        so an invalid estimate is left out rather than counted as 0. Pixels
        with no valid estimate are NaN.  The Cordeiro2022 OWT-to-algorithm recipe is
        OWT 1: Jiang et al. (2021, green), OWT 2: Jiang et al. (2021, red),
        OWT 3: Zhang et al. (2014), and OWT 4: Binding et al. (2010).

        Parameters
        ----------
        xowt_prod : xarray.Dataset
            Cordeiro2022 OWT output containing ranked distance and 1-based class
            index variables with dimensions ``(Nclasses, y, x)``.
        Nclasses : int, optional
            Number of ranked OWT matches to blend (default 2).  It is capped
            at the number present in ``xowt_prod``.
        owt_dist_var, owt_index_var : str, optional
            Names of the Cordeiro distance and class-index variables.
        eps : float, optional
            Small positive value added to distances before inversion.
        limits : bool, optional
            Apply the existing class-specific SPM validity ranges.

        Returns
        -------
        xarray.DataArray
            ``SPM_OWTblend_Ta2025`` in mg L-1.
        """
        if 'Nclasses' not in xowt_prod.dims:
            raise ValueError(
                'Cordeiro2022 OWT blending requires ranked OWT >= 2'
                'Set the Nclasses_to_be_saved >= 2'
            )

        Nclasses = min(xowt_prod.sizes['Nclasses'], Nclasses)
        xowt_prod = xowt_prod.isel(Nclasses=range(Nclasses))

        R443 = self.Rrs.sel(wl=443, method=SEL_XARRAY_METHOD)
        R490 = self.Rrs.sel(wl=490, method=SEL_XARRAY_METHOD)
        R560 = self.Rrs.sel(wl=560, method=SEL_XARRAY_METHOD)
        R665 = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD)
        R705 = self.Rrs.sel(wl=705, method=SEL_XARRAY_METHOD)
        R740 = self.Rrs.sel(wl=740, method=SEL_XARRAY_METHOD)

        # OWT class, algorithm and validity range (used if `limits`)
        recipe = [
            # OWT 1: Jiang et al. (2021), green-band QAA.
            ([1], lambda: self._spm_jiang_green(R443, R490, R560, R665), (0, 50)),
            # OWT 2: Jiang et al. (2021), red-band QAA.
            ([2], lambda: self._spm_jiang_red(R443, R490, R560, R665), (10, 500)),
            # OWT 3: Zhang et al. (2014), 705-nm power law.
            ([3], lambda: self._spm_zhang2014(R705), (20, 1000)),
            # OWT 4: Binding et al. (2010), 740-nm semi-analytical model.
            ([4], lambda: self._spm_binding2010(R740), (50, 2000)),
        ]

        # inverse-distance weighting as in Chl.owt_blending: the weights of a
        # class only enter the norm where its estimate is valid, so an
        # invalid or out-of-range estimate is left out of the blend instead
        # of counting as 0
        spm = 0
        norm = 0
        for owt_index_to_keep, algo, (minval, maxval) in recipe:
            mask = self.create_mask_from_owt(xowt_prod[owt_index_var], owt_index_to_keep)
            estimate = algo()
            if limits:
                estimate = estimate.where((estimate >= minval) & (estimate <= maxval))
            weights = (1 / (xowt_prod[owt_dist_var].where(mask) + eps)).where(estimate.notnull())
            spm += (estimate * weights).fillna(0)
            norm += weights.fillna(0)

        norm = norm.sum('Nclasses')
        spm = spm.sum('Nclasses') / norm.where(norm > 0)
        spm = spm.astype(np.float32)
        spm.name = 'SPM_OWTblend_Ta2025'
        spm.attrs = {
            'description': (
                'OWT-blended SPM from Cordeiro (2022) SPM OWT and '
                'class-specific SPM algorithms, weighted by inverse Euclidean '
                'distance'
            ),
            'references': (
                'Cordeiro, T. (2022); Jiang et al. (2021); '
                'Zhang et al. (2014); Binding et al. (2010)'
            ),
            'units': 'mg L-1',
            'n_classes_blended': Nclasses,
        }
        return spm

    @staticmethod
    def create_mask_from_owt(xowt_raster, owt_index_to_keep):
        """Provide a boolean mask from a 1-based OWT index."""
        first = True
        for owt_num in owt_index_to_keep:
            if first:
                mask = xowt_raster == owt_num
                first = False
            else:
                mask |= xowt_raster == owt_num
        return mask

    # def process_blended(self, n=2, limits=True):
    #     """OWT-blended SPM (Tavares et al. 2025), Cordeiro2022 4-class Euclidean OWT.
    #     Sets self.spm_blend (SPM_OWTblend) and self.owt_index_spm (owt_index_Cordeiro2022_SPM).
    #     """
    #     owt_dist = self.compute_spm_owt_distances(self.Rrs)     # (4, H, W)
    #     # capture before any zeroing: NaN dist means non-water pixel
    #     water_mask = np.any(np.isfinite(owt_dist), axis=0)
    #     owt_classes, owt_weights = self.compute_owt_weights(owt_dist, n=n)

    #     aux_spm = np.stack(
    #         [self._owt_spm_from_class(owt_classes[i], limits=limits) for i in range(n)]
    #     )  # (n, H, W)

    #     ref = aux_spm[0]
    #     for i in range(1, n):
    #         outlier = np.abs(aux_spm[i] - ref) > 4 * np.abs(ref)
    #         owt_weights[i] = np.where(outlier, 0.0, owt_weights[i])
    #         aux_spm[i]     = np.where(outlier, 0.0, aux_spm[i])

    #     for i in range(n):
    #         nan_mask = ~np.isfinite(aux_spm[i])
    #         owt_weights[i] = np.where(nan_mask, 0.0, owt_weights[i])
    #         aux_spm[i]     = np.where(nan_mask, 0.0, aux_spm[i])

    #     weight_sum = owt_weights.sum(axis=0)
    #     spm_blend  = np.where(
    #         weight_sum > 0,
    #         (owt_weights * aux_spm).sum(axis=0) / weight_sum,
    #         np.nan,
    #     ).astype(np.float32)

    #     ref_da = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD).drop_vars('wl', errors='ignore')
    #     scalar_coords = [c for c in ref_da.coords if c not in ref_da.dims and ref_da[c].ndim == 0]
    #     ref_da = ref_da.drop_vars(scalar_coords, errors='ignore')

    #     self.spm_blend = xr.DataArray(
    #         spm_blend,
    #         dims=ref_da.dims,
    #         coords=ref_da.coords,
    #         name='SPM_OWTblend',
    #         attrs={
    #             'description': 'OWT-blended SPM (Tavares et al. 2025), Cordeiro2022 4-class Euclidean OWT',
    #             'units': 'mg/L',
    #             'n_classes_blended': n,
    #             'reference': (
    #                 'Cordeiro, T. (2022); Jiang et al. (2021); '
    #                 'Zhang et al. (2014); Binding et al. (2010)'
    #             ),
    #         },
    #     )

    #     # dominant class index for diagnostics, analogous to owt_index_Spyrakos2018
    #     dominant   = owt_classes[0].astype(np.float32)
    #     self.owt_index_spm = xr.DataArray(
    #         np.where(water_mask, dominant, np.nan),
    #         dims=ref_da.dims,
    #         coords=ref_da.coords,
    #         name='owt_index_Cordeiro2022_SPM',
    #         attrs={'description': 'Dominant SPM OWT class (Cordeiro2022, 4-class Euclidean distance)'},
    #     )
