import os

import numpy as np
import pandas as pd
import xarray as xr

from numba import njit
import logging

from multiprocessing import Pool  # Process pool
from multiprocessing import sharedctypes
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from importlib_resources import files

from . import __package__

OWT_Spyrakos2018_file = 'Spyrakos_et_al_2018_OWT_inland_mean_standardised.csv'
OWT_Bi2024_file = 'Bi_etal_2024_OWT_mean_spec_v01.csv'

OWT_Spyrakos2018_file = files(__package__ +
                              '.data').joinpath(OWT_Spyrakos2018_file)
OWT_Bi2024_file = files(__package__ +
                        '.data').joinpath(OWT_Bi2024_file)


class OWT():
    def __init__(self,
                 raster,
                 owt_database="Spyrakos2018",
                 param='m_nRrs',
                 suffix='',
                 wl_range=slice(350, 800),
                 chunk=1024,
                 Nproc=8):
        '''
        Routine for Optical Water Types (OWT) retrieval from L2A images based on several OWT database and robust spectral metric.

        :param raster: satellite image raster
        :param owt_database: name of the OWT database to use within ["Spyrakos2018","Bi2024"]
        :param param: choose the parameter to use:
                      - for "Spyrakos2018", should be "m_nRrs" (normalized reflectance in nm-1)
                      - for "Bi2024", 2 choices: "m_nRrs" (normalized reflectance in nm-1) or "m_Rrs" (reflectance in sr-1)
        :param wl_range: spectral range to apply the Spectral angle mapper
        :param chunk: chunk size for multiprocessing
        :param Nproc: number of CPU for multiprocessing
        '''

        self.param = param
        self.chunk = chunk
        self.Nproc = Nproc
        self.raster = raster

        self.Rrs = raster.Rrs.sel(wl=wl_range)
        self.Nwl, self.height, self.width = self.Rrs.shape

        self.owt_database = owt_database
        self.suffix = suffix
        if len(self.suffix) > 0:
            self.suffix = '_' + self.suffix

        self.owt_index_name = "owt_index" + self.suffix
        self.owt_dist_name = "owt_dist" + self.suffix

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
                        label='Extremely clear and oligotrophic indigo-blue waters with high reflectance in the short visible wavelengths.'),
                2: dict(color='mediumblue',
                        label='Blue waters with similar biomass level as OWT 1 but with slightly higher detritus and CDOM content.'),
                3: dict(color='cadetblue',
                        label='Turquoise waters with slightly higher phytoplankton, detritus, and CDOM compared to the first two types.'),
                4: dict(color='teal',
                        label='A special case of OWT 3 with similar detritus and CDOM distribution but with strong scattering and little absorbing particles \nlike in the case of Coccolithophore blooms. This type usually appears brighter and exhibits a remarkable ~490 nm reflectance peak.'),
                5: dict(color='plum',
                        label='Greenish water found in coastal and inland environments, with higher biomass compared to the previous water types.\nReflectance in short wavelengths is usually depressed by the absorption of particles and CDOM.'),
                6: dict(color='tan',
                        label='A special case of OWT 5, sharing similar detritus and CDOM distribution, exhibiting phytoplankton blooms \nwith higher scattering coefficients, e.g., Coccolithophore bloom. The color of this type shows a very bright green.'),
                7: dict(color='olivedrab',
                        label='Green eutrophic water, with significantly higher phytoplankton biomass, \nexhibiting a bimodal reflectance shape with typical peaks at ~560 and ~709 nm.'),
                8: dict(color='gold',
                        label='Green hyper-eutrophic water, with even higher biomass than that of OWT 5a (over several orders of magnitude), \ndisplaying a reflectance plateau in the Near Infrared Region, NIR (vegetation-like spectrum).'),
                9: dict(color='chocolate',
                        label='Bright brown water with high detritus concentrations, \nwhich has a high reflectance determined by scattering.'),
                # 'slategrey'
                10: dict(color='red',
                         label='Dark brown to black water with very high CDOM concentration, \nwhich has low reflectance in the entire visible range and is dominated by absorption.'),
                # 11: dict(color='orange', label='OWT11: CDOM-rich with cyanobacteria waters'),
                # 12: dict(color='firebrick', label='OWT12: Turbid waters with cyanobacteria'),
                # 13: dict(color='mediumblue', label='OWT13: Very clear blue waters'),
            }

        self.owt = owt[self.param]
        self.Nowt = len(self.owt.owt)
        self.Rrs_owt = self.owt.interp(wl=self.Rrs.wl).astype(np.float32).squeeze()

        self.output = None

        colors = []
        attrs = ''
        for key, info in self.owt_info.items():
            colors.append(info['color'])
            attrs += str(key) + ":" + info['label'] + '\n'

        self.attrs_owt = attrs
        self.cmap_owt = mpl.colors.ListedColormap(colors)

    @staticmethod
    def xSAM(R1, R2):
        denum = (R1 * R2).sum('wl')
        denom = (R1 ** 2).sum('wl') ** 0.5 * (R2 ** 2).sum('wl') ** 0.5
        return np.arccos(denum / denom)

    @staticmethod
    @njit()
    def SAM(Rrs, Rrs_owt,
            Nwl, Ny, Nx, Nowt):
        '''
        def SAM(R1,R2):
        denum=(R1*R2).sum('wl')
        denom = (R1**2).sum('wl')**0.5 * (R2**2).sum('wl')**0.5
        return np.arccos(denum/denom)
        '''

        arr_sam = np.full((Nowt, Ny, Nx), np.nan, dtype=np.float32)
        arr_index = np.full((Ny, Nx), np.nan, dtype=np.float32)
        Rrs_owt_mod = np.full((Nowt), 0., dtype=np.float32)

        for iowt in range(Nowt):
            for iwl in range(Nwl):
                Rrs_owt_mod[iowt] += Rrs_owt[iowt, iwl] ** 2
            Rrs_owt_mod[iowt] = Rrs_owt_mod[iowt] ** 0.5

        for _iy in range(Ny):
            for _ix in range(Nx):
                if np.isnan(Rrs[0, _iy, _ix]):
                    continue
                for iowt in range(Nowt):
                    denum = 0.
                    Rrs_mod = 0.

                    for iwl in range(Nwl):
                        denum += Rrs[iwl, _iy, _ix] * Rrs_owt[iowt, iwl]
                        Rrs_mod += Rrs[iwl, _iy, _ix] ** 2
                    Rrs_mod = Rrs_mod ** 0.5
                    arr_sam[iowt, _iy, _ix] = np.arccos(denum / (Rrs_mod * Rrs_owt_mod[iowt]))
                arr_index[_iy, _ix] = np.argmin(arr_sam[:, _iy, _ix]) + 1

        return arr_sam, arr_index

    @staticmethod
    def SCS(R1, R2):
        R1_avg = R1.mean('wl')
        R2_avg = R2.mean('wl')
        R1_std = R1.std('wl')
        R2_std = R2.std('wl')
        Nwl = len(R1.wl)
        return 1 / (Nwl) * ((R1 - R1_avg) * (R2 - R2_avg)).sum('wl') / (R1_std * R2_std)

    def process_raster(self):

        chunk = self.chunk
        height, width, Nowt = self.height, self.width, self.Nowt

        owt_index = np.full((height, width), 0, dtype=np.float32)
        owt_dist = np.full((height, width), 0, dtype=np.float32)
        # np.seterr(divide='ignore', invalid='ignore')
        # import warnings
        # with warnings.catch_warnings():
        #    warnings.filterwarnings('ignore', 'invalid value encountered in divide', RuntimeWarning)
        for iy in range(0, height, chunk):
            yc = min(height, iy + chunk)

            for ix in range(0, width, chunk):
                xc = min(width, ix + chunk)

                _Rrs = self.Rrs[:, iy:yc, ix:xc]
                Nwl, Ny, Nx = _Rrs.shape
                owt_sam, owt_index[iy:yc, ix:xc] = self.SAM(_Rrs.values,
                                                            self.Rrs_owt.values,
                                                            Nwl, Ny, Nx, Nowt)

                # owt_scs = SCS(_Rrs,Rrs_owt.values)

                # tmp = owt_scs + (1-2*owt_sam/np.pi)/2

                tmp = -1 * owt_sam / np.pi
                tmp_max = np.max(tmp, axis=0)
                owt_dist[iy:yc, ix:xc] = tmp_max

        self.xowt = xr.Dataset(dict(owt_dist=(["y", "x"], owt_dist),
                                    owt_index=(["y", "x"], owt_index), ),
                               coords=dict(x=self.Rrs.x,
                                           y=self.Rrs.y),
                               )

    def multi_process(self):

        chunk = self.chunk
        height, width, Nowt = self.height, self.width, self.Nowt
        logging.info('OWT classification')
        global chunk_process
        owt_index = np.ctypeslib.as_ctypes(np.full((height, width), np.nan, dtype=np.float32))
        owt_dist = np.ctypeslib.as_ctypes(np.full((height, width), np.nan, dtype=np.float32))
        shared_owt_index = sharedctypes.RawArray(owt_index._type_, owt_index)
        shared_owt_dist = sharedctypes.RawArray(owt_dist._type_, owt_dist)

        def chunk_process(args):
            iy, ix = args
            yc = min(height, iy + chunk)
            xc = min(width, ix + chunk)
            tmp_owt_index = np.ctypeslib.as_array(shared_owt_index)
            tmp_owt_dist = np.ctypeslib.as_array(shared_owt_dist)

            _Rrs = self.Rrs[:, iy:yc, ix:xc]
            Nwl, Ny, Nx = _Rrs.shape
            owt_sam, tmp_owt_index[iy:yc, ix:xc] = self.SAM(_Rrs.values,
                                                            self.Rrs_owt.values,
                                                            Nwl, Ny, Nx, Nowt)

            # owt_scs = SCS(_Rrs,Rrs_owt.values)

            # tmp = owt_scs + (1-2*owt_sam/np.pi)/2

            tmp = -1 * owt_sam / np.pi
            tmp_owt_dist[iy:yc, ix:xc] = np.max(tmp, axis=0)

        window_idxs = [(i, j) for i, j in
                       itertools.product(range(0, height, chunk),
                                         range(0, width, chunk))]

        global pool
        pool = Pool(self.Nproc)
        res = pool.map(chunk_process, window_idxs)
        pool.terminate()
        pool = None
        logging.info('success')

        ######################################
        # construct l2a object
        ######################################
        logging.info('construct xarray owt product')
        self.xowt = xr.Dataset(data_vars={self.owt_dist_name: (["y", "x"], np.ctypeslib.as_array(shared_owt_dist)),
                                          self.owt_index_name: (["y", "x"], np.ctypeslib.as_array(shared_owt_index)), },
                               coords=dict(x=self.Rrs.x,
                                           y=self.Rrs.y),
                               )
        self.xowt[self.owt_index_name].attrs['definition'] = self.attrs_owt

        return self.xowt

    def set_range(self, param, minval=0, maxval=30):
        return param.where((param > minval) & (param < maxval))

    def plot(self):

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
    def __init__(self,
                 raster,
                 chunk=1024,
                 Nproc=8):
        '''

        :param raster:
        :param chunk:
        :param Nproc:
        '''

        self.raster = raster
        self.chunk = chunk
        self.Nproc = Nproc

    def execute(self):
        owt_database = 'Spyrakos2018'
        OWT_kernel = OWT(self.raster,
                         owt_database=owt_database,
                         param='m_nRrs',
                         suffix=owt_database,
                         chunk=self.chunk,
                         Nproc=self.Nproc
                         )
        self.xowt_spyrakos2018 = OWT_kernel.multi_process()

        owt_database = 'Bi2024'
        OWT_kernel = OWT(self.raster,
                         owt_database=owt_database,
                         param='m_Rrs',
                         suffix=owt_database,
                         chunk=self.chunk,
                         Nproc=self.Nproc
                         )
        self.xowt_bi2024 = OWT_kernel.multi_process()

        self.output = xr.merge([self.xowt_spyrakos2018,
                                self.xowt_bi2024])
