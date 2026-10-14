"""Container for an L2A (atmospherically corrected) image product."""

import os, sys, re, glob

import numpy as np
import xarray as xr
import rioxarray
import datetime

from dateutil import parser
import logging

from . import __package__, __version__

opj = os.path.join


class Product():
    """L2A image product loaded from disk or from an in-memory dataset.

    Reads an L2A product (NetCDF, Zarr, possibly a multiscale Zarr pyramid,
    or a directory holding one of them) into an `xarray.Dataset`, and, for
    products using
    the legacy "beam" metadata profile, reshapes the per-band variables
    ``Rrs_<wl>`` and ``Rrs_g_<wl>`` into datacubes ``Rrs`` and ``Rrs_g`` with a
    ``wl`` dimension.

    Parameters
    ----------
    l2a_obj : str or xarray.Dataset
        Path to the L2A product or an already loaded dataset (see
        `__init__`).

    Attributes
    ----------
    processor : str
        Processor identifier, ``'<package>_<version>'``.
    raster : xarray.Dataset
        Main L2A dataset. Not set if the input format is not recognized.
    ancillary : xarray.Dataset or None
        Ancillary dataset: the ``_anc.nc`` file of a NetCDF directory input or
        the ``ancillary`` group of a Zarr pyramid, otherwise None.
    """


    def __init__(self, l2a_obj, chunks=None, level=0):
        """Load the L2A product.

        Parameters
        ----------
        chunks : dict, optional
            Dask chunks for NetCDF inputs. Default: the chunks stored in the
            file (each stored chunk is read once, without the xarray warning
            raised when requested chunks split stored ones), then merged into
            one chunk along ``wl`` (band maths and ``sel`` need all bands).
            Not applied to Zarr inputs, which keep their stored chunks.
        level : int, optional
            Pyramid level read from a multiscale Zarr store (default 0, full
            resolution; level ``k`` is ``2**k`` coarser). Ignored for the
            other inputs.

        l2a_obj : str or xarray.Dataset
            Input product, one of:

            - path to a ``.nc`` file: opened with ``xr.open_dataset``,
              chunked with a single chunk along ``wl``;
            - path to a Zarr store (extension containing ``zarr``): opened
              with ``xr.open_zarr``. If the store is a multiscale pyramid
              (root ``multiscales`` attribute, as written by GRS), the group
              of pyramid `level` is opened, and its ``ancillary`` group, if
              any, as ``ancillary``;
            - path to a directory ``<dir>/`` containing ``<dir>.zarr`` (read
              as above), or ``<dir>.nc`` (main product) and ``<dir>_anc.nc``
              (ancillary data), where the file names match the directory's
              base name;
            - an `xarray.Dataset`, used as is (no reshaping is applied and
              ``ancillary`` is None).

        Notes
        -----
        If the path format is not recognized, a message is logged and the
        constructor returns without setting ``raster``.

        If the loaded file has the metadata attribute
        ``metadata_profile == 'beam'``, the ``Rrs_<wl>`` and ``Rrs_g_<wl>``
        variables are stacked along a new ``wl`` dimension (chunk size 1)
        and merged with the remaining variables. This block is marked as
        deprecated in the code (TODO). It requires integer wavelengths
        because the variable names are built with ``'{:d}'``.
        """

        self.processor = __package__ + '_' + __version__
        # datasets opened from disk here, closed by `close`; a dataset passed
        # in by the caller is not ours to close
        self._opened = []
        # chunks requested at opening must not split the stored chunks, so the
        # default opens with the stored ones and only merges `wl` afterwards
        rechunk = None
        if chunks is None:
            chunks = {}
            rechunk = {'wl': -1}

        ##################################
        # Get image data
        ##################################
        if isinstance(l2a_obj, str):
            logging.info('Load L2A from files')
            l2a_obj = l2a_obj.rstrip('/')
            basename = os.path.basename(l2a_obj)
            # get extension
            extension = l2a_obj.split('.')[-1]
            if extension == 'nc':
                raster = xr.open_dataset(l2a_obj, decode_coords='all', chunks=chunks)
                self.raster = self._rechunk(raster, rechunk)
                self.ancillary = None
                self._opened = [raster]
            elif 'zarr' in extension:
                self._open_zarr(l2a_obj, level)

            elif os.path.isdir(opj(l2a_obj, basename + '.zarr')):
                self._open_zarr(opj(l2a_obj, basename + '.zarr'), level)

            elif os.path.isdir(l2a_obj):

                main_file = opj(l2a_obj, basename + '.nc')
                ancillary_file = opj(l2a_obj, basename + '_anc.nc')

                raster = xr.open_dataset(main_file, decode_coords='all', chunks=chunks)
                self.raster = self._rechunk(raster, rechunk)
                self.ancillary = xr.open_dataset(ancillary_file, decode_coords='all')
                self._opened = [raster, self.ancillary]


            else:
                logging.info('input file format not recognized, stop')
                return

        elif isinstance(l2a_obj, xr.Dataset):
            self.raster = l2a_obj
            self.ancillary = None
            return

        # TODO deprecate and remove this part
        profile = self.raster.attrs.get("metadata_profile")
        if profile != "beam":
            return

        # reshape into datacube:
        wls =  self.raster.wl

        if self.raster.dims.__contains__('wl'):
            self.raster=self.raster.drop_dims('wl')

        Rrs_vars = []
        Rrs_g_vars = []
        for wl in wls:
            Rrs_vars.append('Rrs_{:d}'.format(wl))
            Rrs_g_vars.append('Rrs_g_{:d}'.format(wl))

        Rrs = self.raster[Rrs_vars].to_array(dim='wl', name='Rrs').chunk({'wl': 1})
        Rrs = Rrs.assign_coords({'wl': wls})
        raster = self.raster.drop_vars(Rrs_vars)
        Rrs_g = self.raster[Rrs_g_vars].to_array(dim='wl', name='Rrs_g').chunk({'wl': 1})
        Rrs_g = Rrs_g.assign_coords({'wl': wls})
        raster = raster.drop_vars(Rrs_g_vars)
        self.raster = xr.merge([raster, Rrs, Rrs_g], compat='no_conflicts', join='outer')

    def _open_zarr(self, store, level=0):
        """Open a Zarr store, or one level of a multiscale Zarr pyramid.

        A pyramid is recognized by the ``multiscales`` attribute of its root
        group, whose ``layout`` lists the group of each level. The group of
        `level` is opened as ``raster`` and the ``ancillary`` group, if
        present, as ``ancillary``. Other stores are opened as a single
        dataset, with ``ancillary`` set to None.
        """
        root = xr.open_zarr(store)
        multiscales = root.attrs.get('multiscales')
        if not multiscales:
            root.close()
            self.raster = xr.open_zarr(store, decode_coords='all')
            self.ancillary = None
            self._opened = [self.raster]
            return

        layout = multiscales[0]['layout']
        if not 0 <= level < len(layout):
            root.close()
            raise ValueError(f'pyramid level {level} not in [0, {len(layout) - 1}]')
        group = layout[level]['group']
        logging.info(f'Load pyramid level {level} (group {group!r}) of {store}')
        self.raster = xr.open_zarr(store, group=group, decode_coords='all')
        self._opened = [self.raster]
        self.ancillary = None
        if os.path.isdir(opj(store, 'ancillary')):
            self.ancillary = xr.open_zarr(store, group='ancillary', decode_coords='all')
            self._opened.append(self.ancillary)
        root.close()

    @staticmethod
    def _rechunk(raster, rechunk):
        """Apply `rechunk` to the dimensions of `raster` that have it."""
        if rechunk is None:
            return raster
        rechunk = {d: c for d, c in rechunk.items() if d in raster.dims}
        return raster.chunk(rechunk) if rechunk else raster

    def close(self):
        """Close the files opened from disk by `__init__`.

        A dataset passed directly as ``l2a_obj`` is left open. Lazy arrays
        derived from the closed datasets reopen their file if accessed again.
        """
        for ds in self._opened:
            ds.close()
        self._opened = []
