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

    Reads an L2A product (NetCDF, Zarr, or a directory with a main and an
    ancillary NetCDF file) into an `xarray.Dataset`, and, for products using
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
        Ancillary dataset; only loaded for directory inputs, otherwise None.
    """


    def __init__(self, l2a_obj, chunks=None):
        """Load the L2A product.

        Parameters
        ----------
        chunks : dict, optional
            Dask chunks for NetCDF inputs. Default
            ``{'wl': -1, 'y': 1024, 'x': 1024}``: one chunk along ``wl``
            (band maths and ``sel`` need all bands) and tiles in space, so
            work is parallel and memory per task is bounded. Not applied to
            Zarr inputs, which keep their stored chunks.

        l2a_obj : str or xarray.Dataset
            Input product, one of:

            - path to a ``.nc`` file: opened with ``xr.open_dataset``,
              chunked with a single chunk along ``wl``;
            - path to a Zarr store (extension containing ``zarr``): opened
              with ``xr.open_zarr``;
            - path to a directory ``<dir>/`` containing ``<dir>.nc`` (main
              product) and ``<dir>_anc.nc`` (ancillary data), where the
              file names match the directory's base name;
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
        if chunks is None:
            chunks = {'wl': -1, 'y': 1024, 'x': 1024}

        ##################################
        # Get image data
        ##################################
        if isinstance(l2a_obj, str):
            logging.info('Load L2A from files')
            # get extension
            extension = l2a_obj.split('.')[-1]
            if extension == 'nc':
                self.raster = xr.open_dataset(l2a_obj, decode_coords='all', chunks=chunks)
                self.ancillary = None
            elif 'zarr' in extension:
                self.raster = xr.open_zarr(l2a_obj, decode_coords='all')
                self.ancillary = None

            elif os.path.isdir(l2a_obj):

                basename = os.path.basename(l2a_obj)
                main_file = opj(l2a_obj, basename + '.nc')
                ancillary_file = opj(l2a_obj, basename + '_anc.nc')

                self.raster = xr.open_dataset(main_file, decode_coords='all', chunks=chunks)
                self.ancillary = xr.open_dataset(ancillary_file, decode_coords='all')


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
        self.raster = xr.merge([raster, Rrs, Rrs_g])




