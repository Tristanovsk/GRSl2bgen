import os
import logging 
import numpy as np
import xarray as xr


class Chl():
    def __init__(self,
                 raster,
                 param='Rrs',
                 owt_sam=None):

        self.raster = raster
        self.Rrs = raster[param]
        self.OC2ratio = self.OC2_ratio()
        self.OC3ratio = self.OC3_ratio()
        self.owt_sam = owt_sam
        self.output = None

    def process(self):
        # NASA OC2 for OCTS; bands 490, 565 nm
        acoef = [0.2236, -1.8296, 1.9094, -2.9481, -0.1718]
        self.chl_nasa_oc2 = self.OC2(acoef)
        self.chl_nasa_oc2 = self.set_range(self.chl_nasa_oc2)
        self.chl_nasa_oc2.name = 'Chla_OC2nasa'
        self.chl_nasa_oc2.attrs = {
            'description': 'Chl-a concentration from NASA OC2 with OCTS parameterization, bands 490, 565 nm',
            'applicability': 'oligotrophic waters',
            'coef':str(acoef),
            'reference': 'NASA OCx site',
            'units': 'mg m-3',
            'range':[0,2000]}

        # Gitelson-like Red-edge
        acoef = [232.329, 23.174]
        self.chl_M09B = self.M09B(acoef=acoef)
        self.chl_M09B = self.set_range(self.chl_M09B)
        self.chl_M09B.name = 'Chla_M09B'
        self.chl_M09B.attrs = {
            'description': 'Chl-a concentration from 705nm peak, bands 665, 705, 740 nm',
            'applicability': 'turbid and eutrophic waters',
            'coef':str(acoef),
            'reference': 'Moses, W.J.; Gitelson, A.A.; Berdnikov, S.; Povazhnyy, V. ' + \
                         'Estimation of chlorophyll-a concentration in case II waters using' + \
                         'MODIS and MERIS data:Successes and challenges. Environ. Res. Lett. 2009, 4, 1–8.',
            'units': 'mg m-3',
            'range':[0,2000]
            }

        # NIRB
        acoef = [11.2, 1.7]
        self.chl_NIRB = self.NIRB(acoef=acoef)
        self.chl_NIRB = self.set_range(self.chl_NIRB)
        self.chl_NIRB.name = 'Chla_NIRB'
        self.chl_NIRB.attrs = {
            'description': 'Chl-a concentration from 705/443 ratio, bands 443, 705 nm',
            'applicability': 'case-2 waters',
            'coef': str(acoef),
            'reference': 'Martin, S.; Bryère, P.; Gernez, P.; Renosh, P.R.; Doxaran, D. ' + \
                         'Towards Reliable High-Resolution Satellite Products for the Monitoring ' + \
                         'of Chlorophyll-a and Suspended Particulate Matter in Optically Shallow ' + \
                         'Coastal Lagoons. Remote Sens. 2025, 17, 3430. https://doi.org/10.3390/rs17203430',
            'units': 'mg m-3',
            'range': [0, 2000]
            }

        merge_list = [self.chl_nasa_oc2, self.chl_M09B, self.chl_NIRB]
        if self.owt_sam is not None:
            self.process_blended(self.owt_sam)
            merge_list.append(self.chl_blend)
        self.output = xr.merge(merge_list).drop_vars('wl').compute()

    def set_range(self, param, minval=0, maxval=1200):
        return param.where((param > minval) & (param < maxval))

    # ------------------------------------------------------------------
    # Tavares(2025) algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def _chl_gons(R665, R705, R783, acoef=None):
        if acoef is None:
            acoef = [0.425, 0.704, 1.063, 0.016]  # [aw665, aw708, a, b]
        aw665, aw708, a, b = acoef
        bb = 1.61 * R783 / (0.082 - 0.6 * R783)
        return ((R705 / R665) * (aw708 + bb) - aw665 - bb ** a) / b

    @staticmethod
    def _chl_ndci(R665, R705, acoef=None):
        if acoef is None:
            acoef = [14.039, 86.115, 194.325]  # [a, b, c]
        a, b, c = acoef
        index = (R705 - R665) / (R705 + R665)
        return a + b * index + c * index ** 2

    @staticmethod
    def _chl_gilerson2(R665, R705, acoef=None):
        if acoef is None:
            acoef = [0.022, 1.124]  # [a, b]
        a, b = acoef
        return (0.7864 * (R705 / R665) / a - 0.4245 / a) ** b

    # ------------------------------------------------------------------
    # OWT blending following Tavares et al. (2025)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_owt_weights(owt_sam_np, n=3):
        """Compute top-n OWT class IDs and SAM-derived blending weights.

        Parameters
        ----------
        owt_sam_np : ndarray (Nowt, H, W)
            SAM angles in radians : smaller = better spectral match.
        n : int
            Number of top classes to blend.

        Returns
        -------
        owt_classes : ndarray (n, H, W) float32 : 1-based class IDs.
        owt_weights : ndarray (n, H, W) float32 : blending weights in (0, 1].
        """
        Nowt, H, W = owt_sam_np.shape
        sorted_idx = np.argsort(owt_sam_np, axis=0)             # ascending
        sorted_angles = np.take_along_axis(owt_sam_np, sorted_idx, axis=0)

        top_angles = sorted_angles[:n + 1]   # (n+1, H, W)
        top_idx    = sorted_idx[:n]          # (n, H, W), 0-based

        theta_best = top_angles[0]           # (H, W) best 
        theta_ref  = top_angles[n]           # (H, W)  (n+1)-th worst
        denom = theta_best - theta_ref       # negative; zero : NaN weight

        owt_classes = (top_idx + 1).astype(np.float32)
        owt_weights = np.full((n, H, W), np.nan, dtype=np.float32)
        valid = np.isfinite(denom) & (denom != 0)
        for i in range(n):
            owt_weights[i] = np.where(
                valid,
                (top_angles[i] - theta_ref) / denom,
                np.nan
            ).astype(np.float32)
        return owt_classes, owt_weights

    def _owt_chl_from_class(self, class_px, limits=True):
        """Dispatch per-pixel OWT class IDs to the appropriate Chl-a algorithm.

        Parameters
        ----------
        class_px : ndarray (H, W) float32 : 1-based OWT class IDs (NaN = invalid).
        limits   : bool : apply per-algorithm valid range clipping.

        Returns
        -------
        chla : ndarray (H, W) float32
        """
        R665 = self.Rrs.sel(wl=665).values.squeeze()
        R705 = self.Rrs.sel(wl=705).values.squeeze()
        R783 = self.Rrs.sel(wl=783).values.squeeze()
        H, W = R665.shape
        chla = np.full((H, W), np.nan, dtype=np.float32)

        # Pre-compute OC2 ratio
        oc2_ratio = np.log10(
            self.Rrs.sel(wl=490).values.squeeze() /
            self.Rrs.sel(wl=560).values.squeeze()
        )

        def _ocx(ratio, acoef):
            logchl = sum(acoef[i] * ratio ** i for i in range(len(acoef)))
            return (10.0 ** logchl).astype(np.float32)

        # OWT 1, 6, 10 : Gons
        mask = np.isin(class_px, [1, 6, 10])
        if mask.any():
            val = self._chl_gons(R665, R705, R783).astype(np.float32)
            if limits:
                val = np.where((val > 1) & (val < 250), val, np.nan)
            chla = np.where(mask, val, chla)

        # OWT 2, 4, 5, 11, 12 : NDCI
        mask_ndci = np.isin(class_px, [2, 4, 5, 11, 12])
        if mask_ndci.any():
            val = self._chl_ndci(R665, R705).astype(np.float32)
            if limits:
                val = np.where((val > 5) & (val < 250), val, np.nan)
            chla = np.where(mask_ndci, val, chla)

        # OWT 2, 12 when NDCI > 20 : Gilerson2 override
        ndci_val = self._chl_ndci(R665, R705)
        mask_g2_override = np.isin(class_px, [2, 12]) & (ndci_val > 20)
        if mask_g2_override.any():
            val = self._chl_gilerson2(R665, R705).astype(np.float32)
            if limits:
                val = np.where((val > 5) & (val < 500), val, np.nan)
            chla = np.where(mask_g2_override, val, chla)

        # OWT 7, 8 : Gilerson2
        mask = np.isin(class_px, [7, 8])
        if mask.any():
            val = self._chl_gilerson2(R665, R705).astype(np.float32)
            if limits:
                val = np.where((val > 5) & (val < 500), val, np.nan)
            chla = np.where(mask, val, chla)

        # OWT 3 : OC2 (blue water)
        mask = class_px == 3
        if mask.any():
            val = _ocx(oc2_ratio, [0.1098, -0.755, -14.12, -117.0, -17.76])
            if limits:
                val = np.where((val > 0.01) & (val < 50), val, np.nan)
            chla = np.where(mask, val, chla)

        # OWT 9 : OC2 (turbid blue)
        mask = class_px == 9
        if mask.any():
            val = _ocx(oc2_ratio, [0.0536, 7.308, 116.2, 412.4, 463.5])
            if limits:
                val = np.where((val > 0.01) & (val < 50), val, np.nan)
            chla = np.where(mask, val, chla)

        # OWT 13 : OC2 (very clear)
        mask = class_px == 13
        if mask.any():
            val = _ocx(oc2_ratio, [-5020.0, 2.9e4, -6.1e4, 5.749e4, -2.026e4])
            if limits:
                val = np.where((val > 0.01) & (val < 50), val, np.nan)
            chla = np.where(mask, val, chla)

        return chla.astype(np.float32)

    def process_blended(self, owt_sam_np, n=3, limits=True):
        """Compute OWT-blended Chl-a and store result as `self.chl_blend`.

        Parameters
        ----------
        owt_sam_np : ndarray (Nowt, H, W) : SAM angles from Spyrakos2018 kernel.
        n          : int  : number of top OWT classes to blend.
        limits     : bool : apply per-algorithm valid range clipping.
        """
        # OWT class and short description (Spyrakos2018, 13 classes)
        _ALGO_MAP = {
            1: 'Gons',       2: 'NDCI',       3: 'OC2',
            4: 'NDCI',       5: 'NDCI',       6: 'Gons',
            7: 'Gilerson2',  8: 'Gilerson2',  9: 'OC2',
            10: 'Gons',      11: 'NDCI',      12: 'NDCI',
            13: 'OC2',
        }
        _OWT_DESC = {
            1:  'Hypereutrophic waters',
            2:  'Common case waters',
            3:  'Clear waters',
            4:  'Turbid waters with organic content',
            5:  'Sediment-laden waters',
            6:  'Balanced optical effects at shorter wavelengths',
            7:  'Highly productive cyanobacteria-dominated waters',
            8:  'Productive with cyanobacteria waters',
            9:  'OWT2 with higher Rrs at shorter wavelengths',
            10: 'CDOM-rich waters',
            11: 'CDOM-rich with cyanobacteria waters',
            12: 'Turbid waters with cyanobacteria',
            13: 'Very clear blue waters',
        }

        owt_classes, owt_weights = self.compute_owt_weights(owt_sam_np, n=n)
        # owt_classes, owt_weights: (n, H, W)

        # --- Blending summary ---
        dominant   = owt_classes[0]
        valid_px   = np.isfinite(owt_weights[0])  # NaN for non-water (all-NaN SAM angles)
        uniq, counts = np.unique(dominant[valid_px].astype(int), return_counts=True)
        cls_counts = dict(zip(uniq.tolist(), counts.tolist()))
        top_n_cls  = sorted(cls_counts, key=cls_counts.get, reverse=True)[:n]
        logging.info(f'OWT-blended Chl-a: blending top n={n} classes '
                     f'(Spyrakos2018, {owt_sam_np.shape[0]}-class SAM)')
        logging.info(f'  {"OWT":>4}  {"Algorithm":12}  Description')
        for cls in top_n_cls:
            logging.info(f'  {cls:>4}  {_ALGO_MAP.get(cls, "unknown"):12}  '
                         f'{_OWT_DESC.get(cls, "")}')
        # --- end summary ---

        aux_chla = np.stack(
            [self._owt_chl_from_class(owt_classes[i], limits=limits)
             for i in range(n)]
        )  # (n, H, W)

        # Outlier rejection for layers 1..n-1
        ref = aux_chla[0]
        for i in range(1, n):
            outlier = np.abs(aux_chla[i] - ref) > 4 * np.abs(ref)
            owt_weights[i] = np.where(outlier, 0.0, owt_weights[i])
            aux_chla[i] = np.where(outlier, 0.0, aux_chla[i])

        # Zero out NaN Chl pixels and their weights before summation
        for i in range(n):
            nan_mask = ~np.isfinite(aux_chla[i])
            owt_weights[i] = np.where(nan_mask, 0.0, owt_weights[i])
            aux_chla[i] = np.where(nan_mask, 0.0, aux_chla[i])

        weight_sum = owt_weights.sum(axis=0)
        chla_blend = np.where(
            weight_sum > 0,
            (owt_weights * aux_chla).sum(axis=0) / weight_sum,
            np.nan
        ).astype(np.float32)

        ref_da = self.Rrs.sel(wl=665).drop_vars('wl', errors='ignore')
        # Drop any remaining scalar per-band coords (e.g. central_wavelength) so
        # they don't conflict when chl_blend is merged with chl_nasa_oc2 / chl_M09B.
        scalar_coords = [c for c in ref_da.coords if c not in ref_da.dims and ref_da[c].ndim == 0]
        ref_da = ref_da.drop_vars(scalar_coords, errors='ignore')
        self.chl_blend = xr.DataArray(
            chla_blend,
            dims=ref_da.dims,
            coords=ref_da.coords,
            name='Chla_OWTblend',
            attrs={
                'description': 'OWT-blended Chl-a (Tavares et al. 2025), Spyrakos2018 13-class SAM',
                'units': 'mg m-3',
                'n_classes_blended': n,
            }
        )

    def OCX_chl(self, ratio, acoef):
        logchl = 0
        for i in range(len(acoef)):
            logchl += acoef[i] * ratio ** i
        chl = 10 ** (logchl)
        return chl

    def OC2_ratio(self):
        return np.log10(self.Rrs.sel(wl=490) / self.Rrs.sel(wl=560))

    def OC3_ratio(self):
        blue = self.Rrs.sel(wl=[443, 490]).max(dim='wl')
        return np.log10(blue / self.Rrs.sel(wl=560))

    def OC2(self, acoef):
        return self.OCX_chl(self.OC2ratio, acoef)

    def OC3(self, acoef):
        blue = np.max(self.Rrs.sel(wl=[443, 490]))
        ratio = np.log10(blue / self.Rrs.sel(wl=560))
        return self.OCX_chl(ratio, acoef)

    def RED2(self):
        return self.Rrs.sel(wl=705) / self.Rrs.sel(wl=665)

    def RED3(self):
        return (1 / self.Rrs.sel(wl=665) - 1 / self.Rrs.sel(wl=705)) * self.Rrs.sel(wl=740)

    # following Ogashawara et al. 2021
    def M09B(self, acoef=np.array([232.329, 23.174])):
        index = self.RED3()
        return (acoef[0] * index + acoef[1])

    def NIRB(self, acoef=np.array([11.2, 1.7])):
        index = self.Rrs.sel(wl=705) / self.Rrs.sel(wl=443)
        return acoef[0] * index ** acoef[1]

    def G10B(self, acoef=np.array([113.36, -16.45, 1.124])):
        index = self.RED3(self.Rrs)
        return ((acoef[0] * index + acoef[1]) ** acoef[2])

    def G11B(self, Rrs, acoef=np.array([315.5, 215.95, -25.66])):
        index = self.RED3(Rrs)
        return (acoef[0] * index ** 2 + acoef[1] * index + acoef[2])

    def A14B(self, Rrs, acoef=np.array([581.1, 25.5])):
        return self.M09B(Rrs, acoef=acoef)

    def B16B(self, Rrs, acoef=np.array([98.773, 34.763])):
        return self.M09B(Rrs, acoef=acoef)
