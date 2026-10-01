import os

import numpy as np
import pandas as pd
import xarray as xr

from numba import njit
import logging

from multiprocessing import Pool  # Process pool
from multiprocessing import sharedctypes
import itertools

import dask

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

OWT_tarasenko2025_file = files(__package__ +
                               '.data').joinpath('owt_tarasenko2025.nc')


class OWT():
    '''
            Routine for Optical Water Types (OWT) retrieval from L2A images based on several OWT database and robust spectral metric.

            :param raster: satellite image raster
            :param owt_database: name of the OWT database to use within ["Spyrakos2018","Bi2024","Ta2025"]
            :param param: choose the parameter to use:
                          - for "Spyrakos2018", should be "m_nRrs" (normalized reflectance in nm-1)
                          - for "Bi2024", 2 choices: "m_nRrs" (normalized reflectance in nm-1) or "m_Rrs" (reflectance in sr-1)
            :param owt_database_name: string used to name the output xarray variable (e.g., "owt_index_" + owt_database_name)
            :param provide_sam_distance: if True provide the array [Nowt_to_be_saved, y, x] of the Nowt_te_be_saved top SAM values
            :type provide_sam_distance: Boolean
            :param Nowt_to_be_saved: number of top classes of SAM to be saved in array, default = 3
                                (too high values might slow down further retrieval from "blended algorithms")
            :param wl_range: spectral range to apply the Spectral angle mapper
            :param chunk: chunk size for multiprocessing
            :param Nproc: number of CPU for multiprocessing
    '''

    def __init__(self,
                 raster,
                 xowt=None,
                 owt_database="Spyrakos2018",
                 owt_database_name='',
                 param='m_nRrs',
                 Nowt_to_be_saved=1,
                 wl_range=slice(350, 800),
                 chunk=1024,
                 Nproc=8):

        self.param = param
        self.chunk = chunk
        self.Nproc = Nproc
        self.raster = raster

        self.Rrs = raster.Rrs.sel(wl=wl_range)
        self.Nwl, self.height, self.width = self.Rrs.shape

        self.owt_database = owt_database
        self.owt_database_name = owt_database_name
        if len(self.owt_database_name) > 0:
            self.owt_database_name = '_' + self.owt_database_name

        self.owt_index_name = "owt_index" + self.owt_database_name
        self.owt_dist_name = "owt_dist" + self.owt_database_name
        self.Nowt_dim ="Nowt" + self.owt_database_name
        self.Nowt_to_be_saved = Nowt_to_be_saved

        if xowt is not None:
            self.owt = xowt
            self.attrs_owt = xowt.owt.values
            self.cmap_owt = plt.cm.Spectral_r
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

            self.owt = owt[self.param]
            colors = []
            attrs = ''
            for key, info in self.owt_info.items():
                colors.append(info['color'])
                attrs += str(key) + ":" + info['label'] + '\n'

            self.attrs_owt = attrs
            self.cmap_owt = mpl.colors.ListedColormap(colors)

        self.Nowt = len(self.owt.owt)
        self.Rrs_owt = self.owt.interp(wl=self.Rrs.wl).astype(np.float32).squeeze()
        self.output = None

    @staticmethod
    def xSAM(R1, R2):
        denum = (R1 * R2).sum('wl')
        denom = (R1 ** 2).sum('wl') ** 0.5 * (R2 ** 2).sum('wl') ** 0.5
        return np.arccos(denum / denom)

    @staticmethod
    @njit(parallel=True)
    def SAM(Rrs, Rrs_owt, Nwl, Ny, Nx, Nowt, Nowt_to_be_saved):
        '''
            def SAM(R1,R2):
            denum=(R1*R2).sum('wl')
            denom = (R1**2).sum('wl')**0.5 * (R2**2).sum('wl')**0.5
            return np.arccos(denum/denom)
        '''
        arr_sam = np.full((Nowt_to_be_saved, Ny, Nx), np.nan, dtype=np.float32)
        arr_index = np.full((Nowt_to_be_saved, Ny, Nx), np.nan, dtype=np.float32)

        Rrs_owt_mod = np.zeros(Nowt, dtype=np.float32)
        for iowt in range(Nowt):
            s = 0.
            for iwl in range(Nwl):
                s += Rrs_owt[iowt, iwl] ** 2
            Rrs_owt_mod[iowt] = s ** 0.5

        for _iy in range(Ny):
            tmp = np.empty(Nowt, dtype=np.float32)  # contiguous, private per thread
            for _ix in range(Nx):
                if np.isnan(Rrs[0, _iy, _ix]):
                    continue

                Rrs_mod = 0.
                for iwl in range(Nwl):
                    Rrs_mod += Rrs[iwl, _iy, _ix] ** 2
                Rrs_mod = Rrs_mod ** 0.5
                if Rrs_mod == 0.:
                    continue

                for iowt in range(Nowt):
                    denum = 0.
                    for iwl in range(Nwl):
                        denum += Rrs[iwl, _iy, _ix] * Rrs_owt[iowt, iwl]
                    cosang = denum / (Rrs_mod * Rrs_owt_mod[iowt])
                    cosang = min(1., max(-1., cosang))
                    tmp[iowt] = np.arccos(cosang)

                # sort by increasing SAM and get indices
                # get the smallest sam and respective owt number
                for k in range(Nowt_to_be_saved):
                    best = 0
                    bestval = np.inf
                    for iowt in range(Nowt):
                        if tmp[iowt] < bestval:
                            bestval = tmp[iowt]
                            best = iowt
                    arr_sam[k, _iy, _ix] = tmp[best]
                    arr_index[k, _iy, _ix] = best + 1
                    tmp[best] = np.inf  # exclude from the next pass

        return arr_sam, arr_index

    # @staticmethod
    # @njit()
    # def SAM(Rrs,
    #         Rrs_owt,
    #         Nwl,
    #         Ny,
    #         Nx,
    #         Nowt,
    #         Nowt_to_be_saved):
    #     '''
    #     def SAM(R1,R2):
    #     denum=(R1*R2).sum('wl')
    #     denom = (R1**2).sum('wl')**0.5 * (R2**2).sum('wl')**0.5
    #     return np.arccos(denum/denom)
    #     '''
    #
    #     arr_sam = np.full((Nowt, Ny, Nx), np.nan, dtype=np.float32)
    #     arr_index = np.full((Nowt_to_be_saved,Ny, Nx), np.nan, dtype=np.float32)
    #     Rrs_owt_mod = np.full((Nowt), 0., dtype=np.float32)
    #
    #     for iowt in range(Nowt):
    #         for iwl in range(Nwl):
    #             Rrs_owt_mod[iowt] += Rrs_owt[iowt, iwl] ** 2
    #         Rrs_owt_mod[iowt] = Rrs_owt_mod[iowt] ** 0.5
    #
    #     for _iy in range(Ny):
    #         for _ix in range(Nx):
    #             if np.isnan(Rrs[0, _iy, _ix]):
    #                 continue
    #             for iowt in range(Nowt):
    #                 denum = 0.
    #                 Rrs_mod = 0.
    #
    #                 for iwl in range(Nwl):
    #                     denum += Rrs[iwl, _iy, _ix] * Rrs_owt[iowt, iwl]
    #                     Rrs_mod += Rrs[iwl, _iy, _ix] ** 2
    #                 Rrs_mod = Rrs_mod ** 0.5
    #                 arr_sam[iowt, _iy, _ix] = np.arccos(denum / (Rrs_mod * Rrs_owt_mod[iowt]))
    #             arr_sam[:, _iy, _ix].sort(axis=0)
    #             arr_index[:,_iy, _ix] = (arr_sam[:Nowt_to_be_saved, _iy, _ix]) + 1
    #
    #     return arr_sam, arr_index

    @staticmethod
    def SCS(R1, R2):
        R1_avg = R1.mean('wl')
        R2_avg = R2.mean('wl')
        R1_std = R1.std('wl')
        R2_std = R2.std('wl')
        Nwl = len(R1.wl)
        return 1 / (Nwl) * ((R1 - R1_avg) * (R2 - R2_avg)).sum('wl') / (R1_std * R2_std)

    def multi_process(self):

        chunk = self.chunk
        height, width, Nowt = self.height, self.width, self.Nowt
        logging.info('OWT classification')
        global chunk_process
        owt_index = np.ctypeslib.as_ctypes(np.full((self.Nowt_to_be_saved,height, width), np.nan, dtype=np.float32))
        owt_dist = np.ctypeslib.as_ctypes(np.full((self.Nowt_to_be_saved,height, width), np.nan, dtype=np.float32))

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
            owt_sam, tmp_owt_index[:,iy:yc, ix:xc] = self.SAM(_Rrs.values,
                                                            self.Rrs_owt.values,
                                                            Nwl, Ny, Nx, Nowt,self.Nowt_to_be_saved)

            tmp_owt_dist[:, iy:yc, ix:xc]=owt_sam[:self.Nowt_to_be_saved]

            # TODO implement spectral correlation similarity (SCS) + MSAS (see Bonnier et al, 2024): maybe not necessary small benefit for high computational cost!
            # issue with reshape arrays
            # owt_scs = self.SCS(_Rrs,self.Rrs_owt)
            # tmp = owt_scs + (1-2*owt_sam/np.pi)/2

        window_idxs = [(i, j) for i, j in
                       itertools.product(range(0, height, chunk),
                                         range(0, width, chunk))]

        jobs = [dask.delayed(chunk_process)(arg) for arg in window_idxs]
        dask.compute(jobs)

        logging.info('success')

        ######################################
        # construct l2a object
        ######################################
        logging.info('construct xarray owt product')


        self.xowt = xr.Dataset(data_vars={self.owt_dist_name: ([self.Nowt_dim,"y", "x"], np.ctypeslib.as_array(shared_owt_dist)),
                                          self.owt_index_name: ([self.Nowt_dim,"y", "x"], np.ctypeslib.as_array(shared_owt_index)), },
                               coords={self.Nowt_dim:range(self.Nowt_to_be_saved),
                                       "x":self.Rrs.x,
                                       "y":self.Rrs.y}
                               ).squeeze()
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
                         owt_database_name=owt_database,
                         chunk=self.chunk,
                         Nproc=self.Nproc,
                         Nowt_to_be_saved=3
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
        xowt = xr.open_dataarray(OWT_tarasenko2025_file)
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
