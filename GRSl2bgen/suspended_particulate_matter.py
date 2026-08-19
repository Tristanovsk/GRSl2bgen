import json
import os
import numpy as np
import xarray as xr
from importlib_resources import files
from . import __package__

_SPM_OWT_WLS   = [443, 490, 560, 665, 705, 740, 783, 842, 865]
_SPM_OWT_MEANS = None  # module-level cache; populated on first call


def _load_spm_owt_means():
    """Load Cordeiro2022 4-class SPM OWT mean spectra as (4, 9) float32 array."""
    global _SPM_OWT_MEANS
    if _SPM_OWT_MEANS is not None:
        return _SPM_OWT_MEANS
    f = files(__package__ + '.data').joinpath('Means_OWT_Cordeiro_S2A_SPM.json')
    with f.open('r') as fp:
        data = json.load(fp)
    band_keys = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band8A']
    _SPM_OWT_MEANS = np.array(
        [[data[f'OWT{k}'][b] for b in band_keys] for k in range(1, 5)],
        dtype=np.float32,
    )  # (4, 9)
    return _SPM_OWT_MEANS


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
        self.process_blended(n=2, limits=True)
        self.output = xr.merge([
            self.spm_obs2co, self.turbi_dogliotti, self.spm_nechad,
            self.spm_blend, self.owt_index_spm,
        ]).drop_vars('wl')
        self.output = self.output.compute(scheduler='processes')

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
        red, nir = self.Rrs.sel(wl=665), self.Rrs.sel(wl=865)
        spm_high = coef1[0] * (nir / red) ** coef1[1]
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
        '''
        red = self.Rrs.sel(wl=665)
        t_low = self.nechad_relationship(red, coef_l)
        t_high = self.nechad_relationship(red, coef_h)
        w = (red - switch[0]) / (switch[1] - switch[0])
        t_mixing = (1 - w) * t_low + w * t_high

        t = red.where(red > switch[0], t_low)
        t = t.where(red < switch[1], t_high)
        t = t.where((red <= switch[0]) | (red >= switch[1]), t_mixing)
        return t.where(t >= valid_limit[0], 0).where(t <= valid_limit[1])

    def spm_N10(self, coefs=[342.1, 0.19563], valid_limit=[0, 2000]):
        ''' Semi-analytical algorithm to retrieve suspended particulate matter (in mg/l)
        from remote sensing reflectances (Rrs, in sr-1).
        This algorithm was published in Nechad et al., 2010
        '''
        red = self.Rrs.sel(wl=665)
        spm = self.nechad_relationship(red, coefs)
        return spm.where((spm >= valid_limit[0]), 0).where(spm <= valid_limit[1])

    @staticmethod
    def nechad_relationship(Rrs, coefs):
        rho_wl = np.pi * Rrs
        return coefs[0] * rho_wl / (1 - (rho_wl / coefs[1]))

    # ------------------------------------------------------------------
    # OWT-blended SPM (Tavares et al. 2025 / Cordeiro 2022)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_spm_owt_distances(Rrs):
        """Per-pixel Euclidean distance to each of the 4 Cordeiro2022 SPM OWT mean spectra.
        Returns dist (4, H, W) float32. NaN where any of the 9 bands is NaN.
        """
        owt_means = _load_spm_owt_means()   # (4, 9)
        # sel on a list preserves the wl dim -> (9, H, W); one dask compute for all 9 bands
        rrs_stack = Rrs.sel(wl=_SPM_OWT_WLS).values.astype(np.float32)  # (9, H, W)
        nan_mask  = ~np.all(np.isfinite(rrs_stack), axis=0)              # (H, W)
        H, W = rrs_stack.shape[1], rrs_stack.shape[2]
        dist = np.full((4, H, W), np.nan, dtype=np.float32)
        for k in range(4):
            diff = rrs_stack - owt_means[k, :, None, None]
            dist[k] = np.where(nan_mask, np.nan, np.sqrt(np.sum(diff ** 2, axis=0)))
        return dist  # (4, H, W)

    @staticmethod
    def compute_owt_weights(dist_np, n=2):
        """Top-n OWT classes and blending weights from an (N, H, W) Euclidean distance array.
        Smaller distance = better match. Same formula as Chl.compute_owt_weights.
        Returns owt_classes (n, H, W) float32 (1-based) and owt_weights (n, H, W) float32.
        """
        N, H, W     = dist_np.shape
        sorted_idx  = np.argsort(dist_np, axis=0)
        sorted_dist = np.take_along_axis(dist_np, sorted_idx, axis=0)
        top_dist    = sorted_dist[:n + 1]    # (n+1, H, W)
        top_idx     = sorted_idx[:n]         # (n, H, W), 0-based
        d_best      = top_dist[0]
        d_ref       = top_dist[n]            # (n+1)-th as reference
        denom       = d_best - d_ref
        owt_classes = (top_idx + 1).astype(np.float32)
        owt_weights = np.full((n, H, W), np.nan, dtype=np.float32)
        valid       = np.isfinite(denom) & (denom != 0)
        for i in range(n):
            owt_weights[i] = np.where(
                valid, (top_dist[i] - d_ref) / denom, np.nan
            ).astype(np.float32)
        return owt_classes, owt_weights

    @staticmethod
    def _spm_jiang_green(R443, R490, R560, R665):
        """Jiang et al. 2021, QAA at 560 nm (OWT 1, clear)."""
        aw  = np.array([0.00515124, 0.01919594, 0.06299986, 0.41395333], dtype=np.float32)
        bbw = np.array([0.00215037, 0.00138116, 0.00078491, 0.00037474], dtype=np.float32)
        stack = np.stack([R443, R490, R560, R665], axis=0)
        rrs   = stack / (0.52 + 1.7 * stack)
        u     = (-0.0895 + np.sqrt(0.089 ** 2 + 4 * 0.125 * rrs)) / (2 * 0.125)
        x     = np.log10((rrs[0] + rrs[1]) / (rrs[2] + 5 * rrs[3] ** 2 / rrs[1]))
        a560  = aw[2] + 10 ** (-1.146 - 1.366 * x - 0.469 * x ** 2)
        bbp   = (u[2] * a560) / (1 - u[2]) - bbw[2]
        return (94.48785 * bbp).astype(np.float32)

    @staticmethod
    def _spm_jiang_red(R443, R490, R560, R665):
        """Jiang et al. 2021, QAA at 665 nm (OWT 2, moderate)."""
        aw  = np.array([0.00515124, 0.01919594, 0.06299986, 0.41395333], dtype=np.float32)
        bbw = np.array([0.00215037, 0.00138116, 0.00078491, 0.00037474], dtype=np.float32)
        stack = np.stack([R443, R490, R560, R665], axis=0)
        rrs   = stack / (0.52 + 1.7 * stack)
        u     = (-0.0895 + np.sqrt(0.089 ** 2 + 4 * 0.125 * rrs)) / (2 * 0.125)
        a665  = aw[3] + 0.39 * (R665 / (R443 + R490)) ** 1.14
        bbp   = (u[3] * a665) / (1 - u[3]) - bbw[3]
        return (113.87498 * bbp).astype(np.float32)

    @staticmethod
    def _spm_zhang2014(R705, acoef=None):
        """Zhang et al. 2014, power law at 705 nm (OWT 3, turbid)."""
        if acoef is None:
            acoef = [362507.0, 2.3222]
        a, b = acoef
        return (a * R705 ** b).astype(np.float32)

    @staticmethod
    def _spm_binding2010(R740):
        """Binding et al. 2010, semi-analytical at 740 nm (OWT 4, very turbid)."""
        rho   = R740 * np.pi
        scale = 51.162 / (0.554 * 0.019)           # ~4859.9
        denom = (0.54 * 128.123 / np.pi) + 0.48 * rho
        return (scale * (2.8 * rho / denom - 0.00027)).astype(np.float32)

    def _owt_spm_from_class(self, class_px, limits=True):
        """Dispatch 1-based OWT class map (H, W) to the appropriate SPM algorithm."""
        R443 = self.Rrs.sel(wl=443).values.squeeze()
        R490 = self.Rrs.sel(wl=490).values.squeeze()
        R560 = self.Rrs.sel(wl=560).values.squeeze()
        R665 = self.Rrs.sel(wl=665).values.squeeze()
        R705 = self.Rrs.sel(wl=705).values.squeeze()
        R740 = self.Rrs.sel(wl=740).values.squeeze()
        H, W = R665.shape
        spm  = np.full((H, W), np.nan, dtype=np.float32)

        mask = class_px == 1
        if mask.any():
            val = self._spm_jiang_green(R443, R490, R560, R665)
            if limits:
                val = np.where((val >= 0) & (val <= 50), val, np.nan)
            spm = np.where(mask, val, spm)

        mask = class_px == 2
        if mask.any():
            val = self._spm_jiang_red(R443, R490, R560, R665)
            if limits:
                val = np.where((val >= 10) & (val <= 500), val, np.nan)
            spm = np.where(mask, val, spm)

        mask = class_px == 3
        if mask.any():
            val = self._spm_zhang2014(R705)
            if limits:
                val = np.where((val >= 20) & (val <= 1000), val, np.nan)
            spm = np.where(mask, val, spm)

        mask = class_px == 4
        if mask.any():
            val = self._spm_binding2010(R740)
            if limits:
                val = np.where((val >= 50) & (val <= 2000), val, np.nan)
            spm = np.where(mask, val, spm)

        return spm.astype(np.float32)

    def process_blended(self, n=2, limits=True):
        """OWT-blended SPM (Tavares et al. 2025), Cordeiro2022 4-class Euclidean OWT.
        Sets self.spm_blend (SPM_OWTblend) and self.owt_index_spm (owt_index_Cordeiro2022_SPM).
        """
        owt_dist = self.compute_spm_owt_distances(self.Rrs)     # (4, H, W)
        # capture before any zeroing: NaN dist means non-water pixel
        water_mask = np.any(np.isfinite(owt_dist), axis=0)
        owt_classes, owt_weights = self.compute_owt_weights(owt_dist, n=n)

        aux_spm = np.stack(
            [self._owt_spm_from_class(owt_classes[i], limits=limits) for i in range(n)]
        )  # (n, H, W)

        ref = aux_spm[0]
        for i in range(1, n):
            outlier = np.abs(aux_spm[i] - ref) > 4 * np.abs(ref)
            owt_weights[i] = np.where(outlier, 0.0, owt_weights[i])
            aux_spm[i]     = np.where(outlier, 0.0, aux_spm[i])

        for i in range(n):
            nan_mask = ~np.isfinite(aux_spm[i])
            owt_weights[i] = np.where(nan_mask, 0.0, owt_weights[i])
            aux_spm[i]     = np.where(nan_mask, 0.0, aux_spm[i])

        weight_sum = owt_weights.sum(axis=0)
        spm_blend  = np.where(
            weight_sum > 0,
            (owt_weights * aux_spm).sum(axis=0) / weight_sum,
            np.nan,
        ).astype(np.float32)

        ref_da = self.Rrs.sel(wl=665).drop_vars('wl', errors='ignore')
        scalar_coords = [c for c in ref_da.coords if c not in ref_da.dims and ref_da[c].ndim == 0]
        ref_da = ref_da.drop_vars(scalar_coords, errors='ignore')

        self.spm_blend = xr.DataArray(
            spm_blend,
            dims=ref_da.dims,
            coords=ref_da.coords,
            name='SPM_OWTblend',
            attrs={
                'description': 'OWT-blended SPM (Tavares et al. 2025), Cordeiro2022 4-class Euclidean OWT',
                'units': 'mg/L',
                'n_classes_blended': n,
                'reference': (
                    'Cordeiro, T. (2022); Jiang et al. (2021); '
                    'Zhang et al. (2014); Binding et al. (2010)'
                ),
            },
        )

        # dominant class index for diagnostics, analogous to owt_index_Spyrakos2018
        dominant   = owt_classes[0].astype(np.float32)
        self.owt_index_spm = xr.DataArray(
            np.where(water_mask, dominant, np.nan),
            dims=ref_da.dims,
            coords=ref_da.coords,
            name='owt_index_Cordeiro2022_SPM',
            attrs={'description': 'Dominant SPM OWT class (Cordeiro2022, 4-class Euclidean distance)'},
        )
