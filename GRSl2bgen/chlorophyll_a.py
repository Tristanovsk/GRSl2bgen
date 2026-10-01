"""Chlorophyll-a retrieval algorithms and Optical Water Type (OWT) blending."""

import os
import logging
import numpy as np
import xarray as xr


class Chl():
    """Chlorophyll-a (Chl-a) retrieval from remote-sensing reflectance.

    Groups several empirical and semi-analytical Chl-a algorithms and an
    OWT-based blending scheme (OWT from Spyrakos et al. 2018, recipe from
    Tavares et al. 2025).

    Parameters
    ----------
    raster : xarray.Dataset
        L2A raster product containing the variable `param`, with dimensions
        ``[y, x, wl]`` (``wl`` = wavelength in nm).
    param : str, optional
        Name of the reflectance variable used as input (default ``'Rrs'``,
        in sr-1).
    xowt_prod : xarray.Dataset, optional
        OWT product with variables ``owt_dist`` (SAM distances) and
        ``owt_index`` (1-based OWT class IDs), both with dimensions
        ``[Nowt, y, x]``, sorted so that the best match comes first along
        ``Nowt``. Used for the OWT-blended product.

    Attributes
    ----------
    raster : xarray.Dataset
        Input raster.
    Rrs : xarray.DataArray
        Reflectance variable ``raster[param]``.
    xowt_prod : xarray.Dataset or None
        OWT product.
    output : xarray.Dataset or None
        Merged Chl-a products, set by `process`.
    """



    def __init__(self,
                 raster,
                 param='Rrs',
                 xowt_prod=None):

        self.raster = raster
        self.Rrs = raster[param]
        # self.OC2ratio = self.OC2_ratio(self.Rrs)
        # self.OC3ratio = self.OC3_ratio()
        self.xowt_prod = xowt_prod
        self.output = None

    def process(self):
        """Run the stand-alone algorithms and store the merged result.

        Computes ``Chla_OC2nasa`` (NASA OC2, OCTS parameterization),
        ``Chla_M09B`` (Moses et al. 2009 red-edge) and ``Chla_NIRB``
        (705/443 ratio, Martin et al. 2025), masks values outside
        0-1200 mg m-3 with `set_range`, attaches metadata, and (if an OWT
        product is available) appends the OWT-blended product. The merged
        dataset (``wl`` dropped, computed eagerly) is stored in
        ``self.output``.

        Notes
        -----
        In the current code this method needs updating before it runs:
        ``M09B`` and ``NIRB`` require an ``Rrs`` argument, and the OWT branch
        refers to ``self.owt_sam`` / ``process_blended``, which no longer
        exist (use ``self.xowt_prod`` and `owt_blending`).
        """

        # NASA OC2 for OCTS; bands 490, 565 nm
        acoef = [0.2236, -1.8296, 1.9094, -2.9481, -0.1718]
        self.chl_nasa_oc2 = self.OC2(self.Rrs, acoef)
        self.chl_nasa_oc2 = self.set_range(self.chl_nasa_oc2)
        self.chl_nasa_oc2.name = 'Chla_OC2nasa'
        self.chl_nasa_oc2.attrs = {
            'description': 'Chl-a concentration from NASA OC2 with OCTS parameterization, bands 490, 565 nm',
            'applicability': 'oligotrophic waters',
            'coef': str(acoef),
            'reference': 'NASA OCx site',
            'units': 'mg m-3',
            'range': [0, 2000]}

        # Gitelson-like Red-edge
        acoef = [232.329, 23.174]
        self.chl_M09B = self.M09B(self.Rrs, acoef=acoef)
        self.chl_M09B = self.set_range(self.chl_M09B)
        self.chl_M09B.name = 'Chla_M09B'
        self.chl_M09B.attrs = {
            'description': 'Chl-a concentration from 705nm peak, bands 665, 705, 740 nm',
            'applicability': 'turbid and eutrophic waters',
            'coef': str(acoef),
            'reference': 'Moses, W.J.; Gitelson, A.A.; Berdnikov, S.; Povazhnyy, V. ' + \
                         'Estimation of chlorophyll-a concentration in case II waters using' + \
                         'MODIS and MERIS data:Successes and challenges. Environ. Res. Lett. 2009, 4, 1–8.',
            'units': 'mg m-3',
            'range': [0, 2000]
        }

        # NIRB
        acoef = [11.2, 1.7]
        self.chl_NIRB = self.NIRB(self.Rrs, acoef=acoef)
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
        """Mask values outside an open interval.

        Parameters
        ----------
        param : xarray.DataArray
            Chl-a field.
        minval, maxval : float, optional
            Exclusive lower and upper bounds (default 0 and 1200 mg m-3).

        Returns
        -------
        xarray.DataArray
            `param` with values outside ``(minval, maxval)`` set to NaN.
        """

        return param.where((param > minval) & (param < maxval))

    def owt_blending(self,
                     xowt_prod,
                     Nclasses = 3,
                     eps=1e-6):

        """OWT-blended Chl-a following the recipe of Tavares et al. (2025).

        For each pixel, the `Nclasses` best-matching OWT classes
        (Spyrakos et al. 2018) are used. Each class is assigned an algorithm
        and its Chl-a estimate is weighted by the inverse SAM distance
        ``1 / (owt_dist + eps)``; the weighted sum is divided by the sum of
        the weights.

        Recipe: OWT 1, 6, 10 -> Gons; OWT 2, 4, 5, 11, 12 -> NDCI;
        OWT 7, 8 -> Gilerson2; OWT 3, 9, 13 -> NASA OC2.

        Parameters
        ----------
        xowt_prod : xarray.Dataset
            OWT product with ``owt_dist`` (SAM distance) and ``owt_index``
            (1-based OWT class ID), dimensions ``[Nowt, y, x]``. Entries must
            be ordered from best to worst match along ``Nowt``, because the
            first `Nclasses` entries are selected with ``isel``.
        Nclasses : int, optional
            Number of top OWT classes to blend (default 3); limited to the
            size of ``Nowt``.
        eps : float, optional
            Small value avoiding division by zero in the weights.

        Returns
        -------
        xarray.DataArray
            Blended Chl-a (mg m-3), named ``Chla_OWTblend_Ta2025``, with
            description, references, units and ``n_classes_blended`` as
            attributes.
        """

        # ---------------------------------------------------
        # set number of top OWT classes to be used for algorithm weighting
        # ---------------------------------------------------
        Nclasses = np.min([len(xowt_prod.Nowt), Nclasses])
        xowt_prod = xowt_prod.isel(Nowt=range(Nclasses))

        # ---------------------------------------------------
        # Compute norm from owt SAM distance for each pixel
        # ---------------------------------------------------
        norm = 1 / (xowt_prod.owt_dist + eps)
        # the summation assumes nan identical to 0
        norm = norm.sum('Nowt')
        # replace empty value from 0 to nan
        norm = norm.where(norm > 0)

        # ---------------------------------------------------
        # Recipe and algorithm application
        # ---------------------------------------------------

        owt_index_to_keep = [1, 6, 10]
        mask = self.create_mask_from_owt(xowt_prod.owt_index, owt_index_to_keep)
        weights = 1 / (xowt_prod.owt_dist.where(mask) + eps)
        chla = (self.chl_gons(self.Rrs.where(mask)) * weights).fillna(0)

        # OWT 2, 4, 5, 11, 12 : NDCI
        owt_index_to_keep = [2, 4, 5, 11, 12]
        mask = self.create_mask_from_owt(xowt_prod.owt_index, owt_index_to_keep)
        weights = 1 / (xowt_prod.owt_dist.where(mask) + eps)
        chla += (self.chl_ndci(self.Rrs.where(mask)) * weights).fillna(0)

        # OWT 7, 8 : Gilerson2
        owt_index_to_keep = [7, 8]
        mask = self.create_mask_from_owt(xowt_prod.owt_index, owt_index_to_keep)
        weights = 1 / (xowt_prod.owt_dist.where(mask) + eps)
        chla += (self.chl_gilerson2(self.Rrs.where(mask)) * weights).fillna(0)

        # clear waters
        owt_index_to_keep = [3, 9, 13]
        mask = self.create_mask_from_owt(xowt_prod.owt_index, owt_index_to_keep)
        weights = 1 / (xowt_prod.owt_dist.where(mask) + eps)
        acoef = [0.2236, -1.8296, 1.9094, -2.9481, -0.1718]
        chla += (self.OC2(self.Rrs.where(mask), acoef) * weights).fillna(0)

        # get final product after normalization
        chla = chla.sum('Nowt') / norm

        # add metadata
        chla.attrs = {
            'description': 'OWT-blended Chl-a based on OWT  (using SAM) from Spyrakos et al. (2018) and recipe by Tavares et al. (2025)',
            'references': 'Spyrakos, E.; O’Donnell, R.; Hunter, P.D.; Miller, C.; Scott, M.; Simis, S.G.; Neil, C.; Barbosa, C.C.; Binding, C.E.; Bradt, S.; et al.' \
                          + ' Optical types of inland and coastal waters. Limnol. Oceanogr. 2018, 63, 846–870.' \
                          + '\nTavares, M.H.; Guimaraes, D.; Roussillon, J.; Baute, V.; Cucherousset, J.; Bouletreau, S.; Martinez, J.-M.' \
                          + ' A Framework to Retrieve Water Quality Parameters in Small, ' \
                          + ' Optically Diverse Freshwater Ecosystems Using Sentinel-2 MSI Imagery. Remote Sens. 2025, 17, 2729. ',
            'units': 'mg m-3',
            'n_classes_blended': Nclasses,
        }
        chla.name = 'Chla_OWTblend_Ta2025'

        return chla



    # ------------------------------------------------------------------
    # OWT blending following Tavares et al. (2025)
    # ------------------------------------------------------------------


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

    def OCX_chl(self, ratio, acoef):
        """Evaluate an OCx polynomial: ``10 ** sum(acoef[i] * ratio**i)``.

        Parameters
        ----------
        ratio : xarray.DataArray or ndarray
            Log10 band ratio.
        acoef : sequence of float
            Polynomial coefficients, lowest order first.

        Returns
        -------
        xarray.DataArray or ndarray
            Chl-a (mg m-3).
        """
        logchl = 0
        for i in range(len(acoef)):
            logchl += acoef[i] * ratio ** i
        chl = 10 ** (logchl)
        return chl

    def OC2_ratio(self, Rrs):
        """Return ``log10(Rrs490 / Rrs560)``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension containing 490 and 560 nm.
        """
        return np.log10(Rrs.sel(wl=490) / Rrs.sel(wl=560))

    def OC3_ratio(self, Rrs):
        """Return ``log10(max(Rrs443, Rrs490) / Rrs560)``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension containing 443, 490, 560 nm.
        """
        blue = Rrs.sel(wl=[443, 490]).max(dim='wl')
        return np.log10(blue / Rrs.sel(wl=560))

    def OC2(self, Rrs, acoef):
        """OC2 Chl-a from the 490/560 ratio and polynomial coefficients.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension.
        acoef : sequence of float
            OCx polynomial coefficients, lowest order first.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """
        OC2ratio = self.OC2_ratio(Rrs)
        return self.OCX_chl(OC2ratio, acoef)

    def OC3(self, Rrs, acoef):
        """OC3 Chl-a from the max(443, 490)/560 ratio and coefficients.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension.
        acoef : sequence of float
            OCx polynomial coefficients, lowest order first.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).

        Notes
        -----
        ``np.max`` is taken over the whole array, giving a single blue value
        for the scene; `OC3_ratio` computes the maximum per pixel.
        """
        blue = np.max(Rrs.sel(wl=[443, 490]))
        ratio = np.log10(blue / Rrs.sel(wl=560))
        return self.OCX_chl(ratio, acoef)

    def RED2(self, Rrs):
        """Two-band red-edge ratio ``Rrs705 / Rrs665``."""
        return Rrs.sel(wl=705) / Rrs.sel(wl=665)

    def RED3(self, Rrs):
        """Three-band red-edge index ``(1/Rrs665 - 1/Rrs705) * Rrs740``."""
        return (1 / Rrs.sel(wl=665) - 1 / Rrs.sel(wl=705)) * Rrs.sel(wl=740)

    # following Ogashawara et al. 2021
    def M09B(self, Rrs, acoef=np.array([232.329, 23.174])):
        """Moses et al. (2009) three-band Chl-a: ``a0 * RED3 + a1``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension (665, 705, 740 nm).
        acoef : sequence of float, optional
            ``[a0, a1]``, default ``[232.329, 23.174]``.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """
        index = self.RED3(Rrs)
        return (acoef[0] * index + acoef[1])

    def NIRB(self, Rrs, acoef=np.array([11.2, 1.7])):
        """NIR-blue ratio Chl-a: ``a0 * (Rrs705 / Rrs443) ** a1``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension (443, 705 nm).
        acoef : sequence of float, optional
            ``[a0, a1]``, default ``[11.2, 1.7]``.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """
        index = Rrs.sel(wl=705) / Rrs.sel(wl=443)
        return acoef[0] * index ** acoef[1]

    def G10B(self, Rrs, acoef=np.array([113.36, -16.45, 1.124])):
        """Gitelson-type power law on RED3: ``(a0 * RED3 + a1) ** a2``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension (665, 705, 740 nm).
        acoef : sequence of float, optional
            ``[a0, a1, a2]``, default ``[113.36, -16.45, 1.124]``.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """
        index = self.RED3(Rrs)
        return ((acoef[0] * index + acoef[1]) ** acoef[2])

    def G11B(self, Rrs, acoef=np.array([315.5, 215.95, -25.66])):
        """Gitelson-type quadratic on RED3: ``a0 * RED3**2 + a1 * RED3 + a2``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension (665, 705, 740 nm).
        acoef : sequence of float, optional
            ``[a0, a1, a2]``, default ``[315.5, 215.95, -25.66]``.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """
        index = self.RED3(Rrs)
        return (acoef[0] * index ** 2 + acoef[1] * index + acoef[2])

    def A14B(self, Rrs, acoef=np.array([581.1, 25.5])):
        """`M09B` form with the A14B coefficients (default ``[581.1, 25.5]``).

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension (665, 705, 740 nm).
        acoef : sequence of float, optional
            ``[a0, a1]``.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """
        return self.M09B(Rrs, acoef=acoef)

    def B16B(self, Rrs, acoef=np.array([98.773, 34.763])):
        """`M09B` form with the B16B coefficients (default ``[98.773, 34.763]``).

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension (665, 705, 740 nm).
        acoef : sequence of float, optional
            ``[a0, a1]``.

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """
        return self.M09B(Rrs, acoef=acoef)

    def chl_gons(self,
                 Rrs,
                 acoef=[0.425, 0.704, 1.063, 0.016],
                 wls=[665, 705, 783],
                 method='nearest'):
        """Gons et al. (2005) semi-analytical Chl-a.

        Gons, H.J.; Rijkeboer, M.; Ruddick, K.G. Effect of a waveband shift
        on chlorophyll retrieval from MERIS imagery of inland and coastal
        waters. J. Plankton Res. 2005, 27, 125-127.

        Backscattering is estimated from 783 nm,
        ``bb = 1.61 * R783 / (0.082 - 0.6 * R783)``, and
        ``Chl = ((R705 / R665) * (aw708 + bb) - aw665 - bb ** a) / b``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance (sr-1) with a ``wl`` dimension.
        acoef : list of float, optional
            ``[aw665, aw708, a, b]``, default ``[0.425, 0.704, 1.063, 0.016]``.
        wls : list of float, optional
            Red, red-edge and NIR wavelengths (nm), default
            ``[665, 705, 783]``.
        method : str, optional
            Selection method passed to ``Rrs.sel`` (default ``'nearest'``).

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """

        R665 = Rrs.sel(wl=wls[0], method=method)
        R705 = Rrs.sel(wl=wls[1], method=method)
        R783 = Rrs.sel(wl=wls[2], method=method)

        aw665, aw708, a, b = acoef
        bb = 1.61 * R783 / (0.082 - 0.6 * R783)
        return ((R705 / R665) * (aw708 + bb) - aw665 - bb ** a) / b

    def chl_ndci(self,
                 Rrs,
                 acoef=[14.039, 86.115, 194.325],
                 wls=[665, 705],
                 method='nearest'):
        """Normalized Difference Chlorophyll Index (NDCI) Chl-a.

        Mishra, S.; Mishra, D.R. Normalized difference chlorophyll index: A
        novel model for remote estimation of chlorophyll-a concentration in
        turbid productive waters. Remote Sens. Environ. 2012, 117, 394-406.

        ``index = (R705 - R665) / (R705 + R665)``;
        ``Chl = a + b * index + c * index**2``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension.
        acoef : list of float, optional
            ``[a, b, c]``, default ``[14.039, 86.115, 194.325]``.
        wls : list of float, optional
            Red and red-edge wavelengths (nm), default ``[665, 705]``.
        method : str, optional
            Selection method passed to ``Rrs.sel`` (default ``'nearest'``).

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """

        R665 = Rrs.sel(wl=wls[0], method=method)
        R705 = Rrs.sel(wl=wls[1], method=method)

        a, b, c = acoef
        index = (R705 - R665) / (R705 + R665)

        return a + b * index + c * index ** 2

    def chl_gilerson2(self,
                      Rrs,
                      acoef=[0.022, 1.124],
                      wls=[665, 705],
                      method='nearest'
                      ):
        """Gilerson et al. (2010) two-band Chl-a.

        Gilerson, A.A.; Gitelson, A.A.; Zhou, J.; Gurlin, D.; Moses, W.;
        Ioannou, I.; Ahmed, S.A. Algorithms for remote estimation of
        chlorophyll-a in coastal and inland waters using red and near
        infrared bands. Opt. Express 2010, 18, 24109-24125.

        ``Chl = (0.7864 * (R705 / R665) / a - 0.4245 / a) ** b``.

        Parameters
        ----------
        Rrs : xarray.DataArray
            Reflectance with a ``wl`` dimension.
        acoef : list of float, optional
            ``[a, b]``, default ``[0.022, 1.124]``.
        wls : list of float, optional
            Red and red-edge wavelengths (nm), default ``[665, 705]``.
        method : str, optional
            Selection method passed to ``Rrs.sel`` (default ``'nearest'``).

        Returns
        -------
        xarray.DataArray
            Chl-a (mg m-3).
        """

        R665 = Rrs.sel(wl=wls[0], method=method)
        R705 = Rrs.sel(wl=wls[1], method=method)

        a, b = acoef
        return (0.7864 * (R705 / R665) / a - 0.4245 / a) ** b

    @staticmethod
    def create_mask_from_owt(xowt_raster,
                             owt_index_to_keep):
        """Provide a boolean mask from an OWT index.

        Parameters
        ----------
        xowt_raster : xarray.DataArray
            OWT index.
        owt_index_to_keep : iterable of int
            Indices of the OWT to keep within the mask.

        Returns
        -------
        xarray.DataArray
            Boolean mask, True where the index is in `owt_index_to_keep`.
        """

        first = True
        for owt_num in owt_index_to_keep:
            if first:
                mask = (xowt_raster == owt_num)
                first = False
            else:
                mask |= (xowt_raster == owt_num)
        return mask