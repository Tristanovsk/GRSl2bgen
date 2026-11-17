import json
import os
import numpy as np
import xarray as xr
from importlib_resources import files
from . import __package__
from .config import SEL_XARRAY_METHOD

# OWT-blended SPM: SPM algorithm of each invRrs water type (1-based classes of
# data/owt_invrrs_water_types.nc, WT01-WT14) and its validity range (mg L-1).
# Recipe validated on GLORIA (3038 samples with TSS, MSI bands, spectral angle,
# two classes): MdSA 70.5 %, bias -12 %, against 72.5 % and -18 % with the
# Cordeiro (2022) OWT it replaces (GRSl2bgen notebook/tarasenko/08).
SPM_OWT_RECIPE = {
    'jiang_green': ([1, 2, 3, 4, 5, 6, 8, 9, 10], (0, 50)),   # biomass and CDOM types
    'jiang_red': ([7, 11], (10, 500)),                        # hypereutrophic, moderately turbid
    'zhang2014': ([12, 13, 14], (20, 1000)),                  # turbid to extremely turbid
}


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
                owt_dist_var='owt_dist_invRrsWT',
                owt_index_var='owt_index_invRrsWT',
            )
            spm_blend = self.set_range(spm_blend)
            spm_blend.attrs['range'] = [0, 2000]
            merge_list.append(spm_blend)

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

        spm = spm.where((red > 0) & (nir >= 0))
        if valid_limit[0] is not None:
            spm = spm.where(spm >= valid_limit[0])
        if valid_limit[1] is not None:
            spm = spm.where(spm <= valid_limit[1])

        return spm

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
    # Algorithms of the OWT-blended SPM
    # ------------------------------------------------------------------

    # @staticmethod
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

    def spm_blending(self,
                     xowt_prod,
                     Nclasses=2,
                     owt_dist_var='owt_dist',
                     owt_index_var='owt_index',
                     eps=1e-6,
                     limits=True,
                     recipe=None,
                     fallback=True):
        """Blend the SPM algorithms of the best-matching OWT (invRrs water types).

        The best ``Nclasses`` ranked OWT matches are selected (as
        :meth:`GRSl2bgen.chlorophyll_a.Chl.owt_blending`), each estimate is
        weighted by ``1 / (distance + eps)``, and the weighted sum is divided
        by the sum of the weights of the valid (not NaN, in range) estimates,
        so an invalid estimate is left out rather than counted as 0. Where no
        blended estimate is valid, the algorithm of the best class is used
        without its validity range (``fallback``), if its estimate is within
        0-2000 mg L-1. The algorithm of each class is given by `SPM_OWT_RECIPE`
        (invRrs water types, see ``data/owt_invrrs_water_types.nc``): Jiang et
        al. (2021) green band for the biomass and CDOM types, Jiang et al.
        (2021) red band for the hypereutrophic and moderately turbid types, and
        Zhang et al. (2014) 705 nm for the turbid to extremely turbid types.

        Parameters
        ----------
        xowt_prod : xarray.Dataset
            OWT output containing ranked distance and 1-based class index
            variables with dimensions ``(Nclasses, y, x)``.
        Nclasses : int, optional
            Number of ranked OWT matches to blend (default 2).  It is capped
            at the number present in ``xowt_prod``.
        owt_dist_var, owt_index_var : str, optional
            Names of the distance and class-index variables.
        eps : float, optional
            Small positive value added to distances before inversion.
        limits : bool, optional
            Apply the class-specific SPM validity ranges.
        recipe : dict, optional
            {algorithm: (1-based classes, (min, max))}, default `SPM_OWT_RECIPE`;
            algorithms: ``jiang_green``, ``jiang_red``, ``zhang2014``,
            ``binding2010``.
        fallback : bool, optional
            Use the algorithm of the best class without its validity range
            where no blended estimate is valid (default True).

        Returns
        -------
        xarray.DataArray
            ``SPM_OWTblend_Ta2025`` in mg L-1.
        """
        if 'Nclasses' not in xowt_prod.dims:
            raise ValueError(
                'OWT blending of SPM requires ranked OWT >= 2: '
                'set Nclasses_to_be_saved >= 2'
            )

        Nclasses = min(xowt_prod.sizes['Nclasses'], Nclasses)
        xowt_prod = xowt_prod.isel(Nclasses=range(Nclasses))

        R443 = self.Rrs.sel(wl=443, method=SEL_XARRAY_METHOD)
        R490 = self.Rrs.sel(wl=490, method=SEL_XARRAY_METHOD)
        R560 = self.Rrs.sel(wl=560, method=SEL_XARRAY_METHOD)
        R665 = self.Rrs.sel(wl=665, method=SEL_XARRAY_METHOD)
        R705 = self.Rrs.sel(wl=705, method=SEL_XARRAY_METHOD)
        R740 = self.Rrs.sel(wl=740, method=SEL_XARRAY_METHOD)

        algorithms = {
            'jiang_green': lambda: self._spm_jiang_green(R443, R490, R560, R665),
            'jiang_red': lambda: self._spm_jiang_red(R443, R490, R560, R665),
            'zhang2014': lambda: self._spm_zhang2014(R705),
            'binding2010': lambda: self._spm_binding2010(R740),
        }
        recipe = SPM_OWT_RECIPE if recipe is None else recipe
        estimates = {name: algorithms[name]() for name in recipe}

        # inverse-distance weighting as in Chl.owt_blending: the weights of a
        # class only enter the norm where its estimate is valid, so an
        # invalid or out-of-range estimate is left out of the blend instead
        # of counting as 0
        spm = 0
        norm = 0
        for name, (owt_index_to_keep, (minval, maxval)) in recipe.items():
            mask = self.create_mask_from_owt(xowt_prod[owt_index_var], owt_index_to_keep)
            estimate = estimates[name]
            if limits:
                estimate = estimate.where((estimate >= minval) & (estimate <= maxval))
            weights = (1 / (xowt_prod[owt_dist_var].where(mask) + eps)).where(estimate.notnull())
            spm += (estimate * weights).fillna(0)
            norm += weights.fillna(0)

        norm = norm.sum('Nclasses')
        spm = spm.sum('Nclasses') / norm.where(norm > 0)
        if fallback:
            # no valid blended estimate: algorithm of the best class without its validity range
            best = xowt_prod[owt_index_var].isel(Nclasses=0)
            for name, (owt_index_to_keep, _) in recipe.items():
                estimate = estimates[name]
                use = (spm.isnull() & self.create_mask_from_owt(best, owt_index_to_keep)
                       & (estimate >= 0) & (estimate <= 2000))
                spm = spm.where(~use, estimate)
        spm = spm.astype(np.float32)
        spm.name = 'SPM_OWTblend_Ta2025'
        spm.attrs = {
            'description': (
                'OWT-blended SPM: invRrs water types (spectral angle) and '
                'class-specific SPM algorithms, weighted by the inverse distance'
            ),
            'references': (
                'Harmel et al. (invRrs water types); Jiang et al. (2021); '
                'Zhang et al. (2014)'
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
