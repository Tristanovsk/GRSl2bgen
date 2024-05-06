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


class OWT():
    def __init__(self,
                 raster,
                 wl_range=slice(350, 800),
                 chunk=1024,
                 Nproc=8):

        self.chunk = chunk
        self.Nproc = Nproc
        self.raster = raster
        self.Rrs = raster.Rrs.sel(wl=wl_range)
        self.Nwl, self.height, self.width = self.Rrs.shape

        owt_file = '/DATA/projet/vrac/owt/Spyrakos_et_al_2018_OWT_inland_mean_standardised.csv'
        owt = pd.read_csv(owt_file, index_col=0).stack().to_xarray().astype(np.float32)
        owt = owt.rename({'level_1': 'wl'})
        owt['wl'] = owt.wl.astype(np.float32)
        self.owt = owt
        self.Nowt = len(owt.owt)
        self.Rrs_owt = owt.interp(wl=self.Rrs.wl).astype(np.float32)

        colors = ['olivedrab', 'black', 'cadetblue', 'tan', 'chocolate', 'teal', 'blueviolet', 'plum', 'red', 'orange',
                  'gold', 'firebrick', 'mediumblue']
        self.cmap_owt = mpl.colors.ListedColormap(colors)

        self.output = None

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

    def process(self):

        chunk = self.chunk
        height, width, Nowt = self.height, self.width, self.Nowt

        owt_index = np.full((height, width), 0, dtype=np.float32)
        owt_dist = np.full((height, width), 0, dtype=np.float32)
        #np.seterr(divide='ignore', invalid='ignore')
        #import warnings
        #with warnings.catch_warnings():
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
        self.xowt = xr.Dataset(dict(owt_dist=(["y", "x"],  np.ctypeslib.as_array(shared_owt_dist)),
                                    owt_index=(["y", "x"],  np.ctypeslib.as_array(shared_owt_index)), ),
                               coords=dict(x=self.Rrs.x,
                                           y=self.Rrs.y),
                               )


    def set_range(self, param, minval=0, maxval=30):
        return param.where((param > minval) & (param < maxval))

    def plot(self):

        owt_info = {
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


        patch = []
        for key, info in owt_info.items():
            patch.append(mpatches.Patch(color=info['color'], label=info['label']))

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(9, 6))
        ax.minorticks_on()
        for iowt, group in self.owt.groupby('owt'):
            group.plot(color=owt_info[iowt]['color'], lw=3)
        ax.set_title('')
        ax.set_ylabel('$Standardized\ R_{rs}\ (nm^{-1})$', fontsize=20)
        ax.set_xlabel('$Wavelength\ (nm)$', fontsize=20)
        plt.legend(handles=patch, fontsize=13, bbox_to_anchor=(1, .5, 0.5, 0.5))

        return fig, ax
