"""Optical Water Type (OWT) classification with the Spectral Angle Mapper (lazy, dask-friendly)."""

import numpy as np
import pandas as pd
import xarray as xr
import logging

import matplotlib as mpl
import matplotlib.patches as mpatches

from importlib_resources import files

from . import __package__
from .sam import sam as _sam, sam_dataarray
from .euclidean import euclidean as _euclidean, euclidean_dataarray

OWT_Spyrakos2018_file = 'Spyrakos_et_al_2018_OWT_inland_mean_standardised.csv'
OWT_Bi2024_file = 'Bi_etal_2024_OWT_mean_spec_v01.csv'

OWT_Spyrakos2018_file = files(__package__ +
                              '.data').joinpath(OWT_Spyrakos2018_file)
OWT_Bi2024_file = files(__package__ +
                        '.data').joinpath(OWT_Bi2024_file)

OWT_tarasenko2025_file = files(__package__ +
                               '.data').joinpath('owt_tarasenko2025.nc')


class OWT():
    """Optical Water Type (OWT) retrieval from L2A images.

        Classifies every pixel with the Spectral Angle Mapper (SAM) against the
        mean spectra of an OWT database and keeps the `Nclasses_to_be_saved`
        best-matching classes. The computation is lazy for dask-backed input:
        nothing is evaluated until the result is written or computed.

        Parameters
        ----------
        raster : xarray.Dataset
            L2A raster with the variable ``Rrs`` (dimensions y, x, wl in any
            order; ``wl`` in nm).
        xowt : xarray.DataArray, optional
            Externally supplied OWT spectra with dimensions (owt, wl). If given,
            `owt_database` is ignored.
        owt_database : {'Spyrakos2018', 'Bi2024'}, optional
            Built-in OWT database.
        owt_database_name : str, optional
            Suffix for the output variable names, e.g. ``'Spyrakos2018'`` gives
            ``owt_index_Spyrakos2018`` and ``owt_dist_Spyrakos2018``.
        param : str, optional
            Spectra to use. ``'m_nRrs'`` (normalised reflectance, nm-1) for
            Spyrakos2018; ``'m_nRrs'`` or ``'m_Rrs'`` (sr-1) for Bi2024.
        spectral_distance : str, optional
            Spectral distance to use (Default: ``'sam'``):
                - 'sam' for Spectral Angle Mapper
                - 'euclidean' for Euclidean Distance
        Nclasses_to_be_saved : int, optional
            Number of best-matching classes kept per pixel (default 1). Higher
            values slow down the blended retrievals that use them.
        wl_range : slice, optional
            Spectral range used for the SAM (default ``slice(350, 800)``).
        chunk : int, optional
            Spatial chunk size (pixels) for dask-backed input.
        Nproc : int, optional
            Deprecated and unused (kept for backward compatibility): parallelism
            now comes from dask and numba.

        Attributes
        ----------
        xowt : xarray.Dataset
            Result of `multi_process`: ``owt_dist_<name>`` (radians) and
            ``owt_index_<name>`` (1-based class position).
        """

    def __init__(self,
                 raster,
                 xowt=None,
                 owt_database="Spyrakos2018",
                 owt_database_name='',
                 param='m_nRrs',
                 spectral_distance='sam',
                 Nclasses_to_be_saved=1,
                 wl_range=slice(350, 800),
                 chunk=1024,
                 Nproc=8):

        self.param = param
        self.chunk = chunk
        self.Nproc = Nproc
        self.raster = raster

        self.Rrs = raster.Rrs.sel(wl=wl_range)
        self.Nwl = self.Rrs.sizes['wl']
        self.height = self.Rrs.sizes['y']
        self.width = self.Rrs.sizes['x']

        if spectral_distance == "sam":
            self.spectral_distance = sam_dataarray
        elif spectral_distance == "euclidean":
            self.spectral_distance = euclidean_dataarray
        else:
            raise ValueError(f'unknown sepctral distance {spectral_distance!r}')

        self.owt_database = owt_database
        self.owt_database_name = owt_database_name
        if len(self.owt_database_name) > 0:
            self.owt_database_name = '_' + self.owt_database_name

        self.owt_index_name = "owt_index" + self.owt_database_name
        self.owt_dist_name = "owt_dist" + self.owt_database_name
        self.Nclasses_dim = "Nclasses"
        self.Nclasses_to_be_saved = Nclasses_to_be_saved
        self.owt_info = {}

        if xowt is not None:
            self.owt = xowt
            self.attrs_owt = xowt.owt.values
            self.cmap_owt = mpl.colormaps['Spectral_r']
        else:
            if self.owt_database == 'Spyrakos2018':
                owt = pd.read_csv(OWT_Spyrakos2018_file, index_col=0).stack().to_xarray().astype(np.float32)
                owt = owt.rename({'level_1': 'wl'})
                owt['wl'] = owt.wl.astype(np.float32)
                owt.name = 'm_nRrs'
                owt = owt.to_dataset()
                self.owt_info = {
                    1: dict(color='olivedrab', label='OWT1: Hypereutrophic waters'),
                    2: dict(color='black', label='OWT2: Common case waters'),
                    3: dict(color='cadetblue', label='OWT3: Clear waters'),
                    4: dict(color='tan', label='OWT4: Turbid waters with organic content'),
                    5: dict(color='chocolate', label='OWT5: Sediment-laden waters'),
                    6: dict(color='teal', label='OWT6: Balanced optical effects at shorter wavelengths'),
                    7: dict(color='blueviolet', label='OWT7: Highly productive cyanobacteria-dominated waters'),
                    8: dict(color='plum', label='OWT8: Productive with cyanobacteria waters'),
                    9: dict(color='red', label='OWT9: OWT2 with higher $R_{rs}$ at shorter wavelengths'),  # 'slategrey'
                    10: dict(color='orange', label='OWT10: CDOM-rich waters'),
                    11: dict(color='gold', label='OWT11: CDOM-rich with cyanobacteria waters'),
                    12: dict(color='firebrick', label='OWT12: Turbid waters with cyanobacteria'),
                    13: dict(color='mediumblue', label='OWT13: Very clear blue waters'),
                }

            elif self.owt_database == 'Bi2024':
                owt = pd.read_csv(OWT_Bi2024_file, index_col=[0, 1]).to_xarray()
                owt = owt.rename({'type': 'owt', 'wavelen': 'wl'})
                owt['owt'] = range(1, 11)

                self.owt_info = {
                    1: dict(color='blueviolet',
                            label='1. Extremely clear and oligotrophic indigo-blue waters with high reflectance in the short visible wavelengths.'),
                    2: dict(color='mediumblue',
                            label='2. Blue waters with similar biomass level as OWT 1 but with slightly higher detritus and CDOM content.'),
                    3: dict(color='cadetblue',
                            label='3a. Turquoise waters with slightly higher phytoplankton, detritus, and CDOM compared to the first two types.'),
                    4: dict(color='teal',
                            label='3b. A special case of OWT 3a with similar detritus and CDOM distribution but with strong scattering and little absorbing particles \nlike in the case of Coccolithophore blooms. This type usually appears brighter and exhibits a remarkable ~490 nm reflectance peak.'),
                    5: dict(color='plum',
                            label='4a. Greenish water found in coastal and inland environments, with higher biomass compared to the previous water types.\nReflectance in short wavelengths is usually depressed by the absorption of particles and CDOM.'),
                    6: dict(color='tan',
                            label='4b. A special case of OWT 4a, sharing similar detritus and CDOM distribution, exhibiting phytoplankton blooms \nwith higher scattering coefficients, e.g., Coccolithophore bloom. The color of this type shows a very bright green.'),
                    7: dict(color='olivedrab',
                            label='5a. Green eutrophic water, with significantly higher phytoplankton biomass, \nexhibiting a bimodal reflectance shape with typical peaks at ~560 and ~709 nm.'),
                    8: dict(color='gold',
                            label='5b. Green hyper-eutrophic water, with even higher biomass than that of OWT 5a (over several orders of magnitude), \ndisplaying a reflectance plateau in the Near Infrared Region, NIR (vegetation-like spectrum).'),
                    9: dict(color='chocolate',
                            label='6. Bright brown water with high detritus concentrations, \nwhich has a high reflectance determined by scattering.'),
                    # 'slategrey'
                    10: dict(color='red',
                             label='7. Dark brown to black water with very high CDOM concentration, \nwhich has low reflectance in the entire visible range and is dominated by absorption.'),
                    # 11: dict(color='orange', label='OWT11: CDOM-rich with cyanobacteria waters'),
                    # 12: dict(color='firebrick', label='OWT12: Turbid waters with cyanobacteria'),
                    # 13: dict(color='mediumblue', label='OWT13: Very clear blue waters'),
                }
            else:
                raise ValueError(f'unknown OWT database {self.owt_database!r}')

            self.owt = owt[self.param]
            colors = []
            attrs = ''
            for key, info in self.owt_info.items():
                colors.append(info['color'])
                attrs += str(key) + ":" + info['label'] + '\n'

            self.attrs_owt = attrs
            self.cmap_owt = mpl.colors.ListedColormap(colors)

        self.Nowt = len(self.owt.owt)
        # class spectra interpolated on the image wavelengths, shape (owt, wl)
        self.Rrs_owt = (self.owt.interp(wl=self.Rrs.wl)
                        .astype(np.float32)
                        .squeeze()
                        .transpose('owt', 'wl'))
        self.output = None

    @staticmethod
    def xSAM(R1, R2):
        """Spectral angle (radians) between two xarray spectra along ``wl``."""
        denum = (R1 * R2).sum('wl')
        denom = (R1 ** 2).sum('wl') ** 0.5 * (R2 ** 2).sum('wl') ** 0.5
        return np.arccos(denum / denom)

    @staticmethod
    def SAM(Rrs, Rrs_owt, Nclasses_to_be_saved):
        """Numba SAM with the legacy signature (kept for backward compatibility).

        Parameters
        ----------
        Rrs : ndarray, shape (Nwl, Ny, Nx)
        Rrs_owt : ndarray, shape (Nowt, Nwl)
        Nclasses_to_be_saved : int
            Number of best classes kept per pixel.

        Returns
        -------
        dist, index : ndarray, shape (Nclasses_to_be_saved, Ny, Nx), float32
            Angles in radians (ascending) and 1-based class positions.
        """
        return _sam(np.moveaxis(np.asarray(Rrs), 0, -1), Rrs_owt, Nclasses_to_be_saved)

    @staticmethod
    def SCS(R1, R2):
        """Spectral correlation similarity between two xarray spectra along ``wl``."""
        R1_avg = R1.mean('wl')
        R2_avg = R2.mean('wl')
        R1_std = R1.std('wl')
        R2_std = R2.std('wl')
        Nwl = len(R1.wl)
        return 1 / (Nwl) * ((R1 - R1_avg) * (R2 - R2_avg)).sum('wl') / (R1_std * R2_std)

    def multi_process(self):
        """Classify the image and build the OWT dataset (lazy for dask input).

        Uses `sam_dataarray` or `euclidean_dataarray`: for dask-backed ``Rrs`` the SAM (or Euclidean distance)
        is computed chunk by chunk by the dask scheduler; for in-memory data a parallel
        numba kernel is used. There is no shared memory and no worker pool.

        Returns
        -------
        xarray.Dataset
            ``owt_dist_<name>`` (radians) and ``owt_index_<name>`` (1-based
            class position) with dimensions ``[Nclasses, y, x]``; the
            ``Nclasses`` dimension is dropped when only one class is kept.
        """

        logging.info('OWT classification')

        Rrs = self.Rrs
        if Rrs.chunks is not None and self.chunk:
            Rrs = Rrs.chunk({'y': self.chunk, 'x': self.chunk, 'wl': -1})

        dist, index = self.spectral_distance(Rrs,
                                             self.Rrs_owt.values,
                                             Nclasses_to_be_saved=self.Nclasses_to_be_saved,
                                             wl_dim='wl',
                                             name_dim=self.Nclasses_dim)

        logging.info('construct xarray owt product')
        self.xowt = xr.Dataset({self.owt_dist_name: dist.rename(self.owt_dist_name),
                                self.owt_index_name: index.rename(self.owt_index_name)})
        self.xowt = self.xowt.assign_coords(
            {self.Nclasses_dim: np.arange(self.Nclasses_to_be_saved)})
        if self.Nclasses_to_be_saved == 1:
            self.xowt = self.xowt.isel({self.Nclasses_dim: 0}, drop=True)

        self.xowt[self.owt_dist_name].attrs['units'] = 'radians'
        self.xowt[self.owt_index_name].attrs['definition'] = self.attrs_owt

        return self.xowt

    def set_range(self, param, minval=0, maxval=30):
        """Mask values outside the open interval ``(minval, maxval)``."""
        return param.where((param > minval) & (param < maxval))

    def plot(self):
        """Plot the OWT spectra with their colours (built-in databases only)."""
        import matplotlib.pyplot as plt

        if not self.owt_info:
            raise NotImplementedError('plot needs owt_info, which is not available for externally supplied OWT')
        patch = []
        for key, info in self.owt_info.items():
            patch.append(mpatches.Patch(color=info['color'], label=info['label']))

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(9, 6))
        ax.minorticks_on()
        for iowt, group in self.owt.groupby('owt'):
            group.plot(color=self.owt_info[iowt]['color'], lw=3)
        ax.set_title('')
        if self.param == "m_nRrs":
            ax.set_ylabel(r'$Standardized\ R_{rs}\ (nm^{-1})$', fontsize=20)
        elif self.param == "m_Rrs":
            ax.set_ylabel(r'$R_{rs}\ (sr^{-1})$', fontsize=20)

        ax.set_xlabel(r'$Wavelength\ (nm)$', fontsize=20)
        plt.legend(handles=patch, fontsize=13, bbox_to_anchor=(1, .5, 0.5, 0.5))

        return fig, ax


class OWT_process():
    """Run the OWT classifications used by the L2B processing chain.

    Classifies the raster against the Spyrakos et al. (2018) OWT (top 3
    classes), the Bi et al. (2024) OWT and the Tarasenko et al. (2025) OWT
    (best class each).

    Parameters
    ----------
    raster : xarray.Dataset
        L2A raster with ``Rrs`` (see `OWT`).
    spectral_distance : str, optional
            Spectral distance to use (Default: ``'sam'``):
                - 'sam' for Spectral Angle Mapper
                - 'euclidean' for Euclidean Distance
    chunk : int, optional
        Spatial chunk size for dask-backed input (default 1024).
    Nproc : int, optional
        Deprecated and unused.

    Attributes
    ----------
    xowt_spyrakos2018, xowt_bi2024, xowt_Ta2025 : xarray.Dataset
        Per-database results.
    owt_sam_spyrakos2018 : xarray.Dataset
        Spyrakos2018 result in the form expected by ``Chl``: variables
        ``owt_dist`` and ``owt_index`` with dimensions ``[Nowt, y, x]``.
    output : xarray.Dataset
        Merge of the three results (variables with database suffix).
    """

    def __init__(self,
                 raster,
                 spectral_distance='sam',
                 chunk=1024,
                 Nproc=8):
        self.raster = raster
        self.spectral_distance = spectral_distance
        self.chunk = chunk
        self.Nproc = Nproc

    def execute(self):
        owt_database = 'Spyrakos2018'
        OWT_kernel = OWT(self.raster,
                         owt_database=owt_database,
                         param='m_nRrs',
                         owt_database_name=owt_database,
                         spectral_distance=self.spectral_distance,
                         chunk=self.chunk,
                         Nproc=self.Nproc,
                         Nclasses_to_be_saved=3
                         )
        self.xowt_spyrakos2018 = OWT_kernel.multi_process()

        owt_database = 'Bi2024'
        OWT_kernel = OWT(self.raster,
                         owt_database=owt_database,
                         param='m_Rrs',
                         owt_database_name=owt_database,
                         chunk=self.chunk,
                         Nproc=self.Nproc
                         )
        self.xowt_bi2024 = OWT_kernel.multi_process()

        # 2025-10-27 add OWT used in Tarasenko et al., 2025
        with xr.open_dataarray(OWT_tarasenko2025_file) as xowt:
            xowt = xowt.load()
        OWT_kernel = OWT(self.raster,
                         xowt=xowt,
                         owt_database_name='Ta2025',
                         param='m_nRrs',
                         chunk=self.chunk,
                         Nproc=self.Nproc
                         )
        self.xowt_Ta2025 = OWT_kernel.multi_process()

        self.output = xr.merge([self.xowt_spyrakos2018,
                                self.xowt_bi2024,
                                self.xowt_Ta2025])
