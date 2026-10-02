"""Level-2B product assembly and export to compressed, packed NetCDF."""

import os
import dask
import shutil
import numcodecs

import numpy as np
import xarray as xr
import rioxarray as rio
import logging
import datetime

from . import __package__, __version__


class L2bProduct():
    """Level-2B product built from an L2A product and derived L2 rasters.

    Merges the derived variables (e.g. Chl-a, SPM) with the ``flags`` and
    ``mask`` layers of the input product, sets the processing metadata, and
    exports the result to a NetCDF file in which the geophysical variables
    are packed into 16-bit integers.

    Parameters
    ----------
    prod : Product
        L2A product object; its ``raster`` dataset must provide ``Rrs`` and
        may provide ``flags`` and ``mask``.
    l2_raster_list : list of xarray.DataArray or xarray.Dataset
        L2 rasters to merge into the L2B product (all on the same grid).

    Attributes
    ----------
    processor : str
        Processor identifier, ``'<package>_<version>'``.
    prod : Product
        Input L2A product.
    l2b_raster_list : list
        Input L2 rasters.
    variables : list of str
        Names of the variables of the L2B dataset.
    l2b_prod : xarray.Dataset
        The assembled L2B dataset.
    complevel : int
        zlib compression level used on export (default 5).
    """

    def __init__(self, prod, l2_raster_list):
        self.processor = __package__ + '_' + __version__
        self.prod = prod
        self.l2b_raster_list = l2_raster_list
        self.variables = None
        self.l2b_prod = None
        self.complevel = 5
        self.construct_l2b()

    def construct_l2b(self):
        """Assemble the L2B dataset and set ``self.l2b_prod`` and ``self.variables``.

        Steps:

        1. Merge the L2 rasters with ``compat='override'`` (for conflicting
           values the first object wins).
        2. Add ``flags`` and ``mask`` from the input product; if absent,
           create them as ``uint8`` arrays of zeros on the image grid.
        3. Copy the attributes of the input raster and add
           ``processing_time`` (local time, ISO 8601) and ``processor``.

        Notes
        -----
        ``l2b_prod.attrs`` is the same dict object as ``prod.raster.attrs``,
        so the two new attributes are also written to the input product.
        """

        logging.info('construct l2b')

        l2b_prod = xr.merge(self.l2b_raster_list, compat='override')

        if 'flags' in self.prod.raster.keys():
            l2b_prod['flags'] = self.prod.raster['flags']
        else:
            l2b_prod['flags'] = xr.zeros_like(self.prod.raster.Rrs.isel(wl=0, drop=True).squeeze().astype(np.uint8))
        if 'mask' in self.prod.raster.keys():
            l2b_prod['mask'] = self.prod.raster['mask']
        else:
            l2b_prod['mask'] = xr.zeros_like(self.prod.raster.Rrs.isel(wl=0, drop=True).squeeze().astype(np.uint8))

        l2b_prod.attrs = self.prod.raster.attrs
        l2b_prod.attrs['processing_time'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        l2b_prod.attrs['processor'] = self.processor
        self.variables = list(l2b_prod.keys())
        self.l2b_prod = l2b_prod

    @staticmethod
    def compute_scale_and_offset(array, nbit=16):
        """Compute NetCDF packing parameters for an integer encoding.

        The data range ``[min, max]`` is stretched over the ``2**nbit - 1``
        steps of an ``nbit`` signed integer, centred on zero, so that
        ``packed = (value - add_offset) / scale_factor`` and
        ``value = packed * scale_factor + add_offset``.

        Parameters
        ----------
        array : array-like
            Data to pack; NaN values are ignored.
        nbit : int, optional
            Number of bits of the packed integer type (default 16).

        Returns
        -------
        scale_factor : float
            ``(max - min) / (2**nbit - 1)``.
        add_offset : float
            ``min + 2**(nbit - 1) * scale_factor``.

        Notes
        -----
        With this formula the minimum maps to ``-2**(nbit - 1)`` (-32768 for
        16 bits), which is also the ``_FillValue`` used in
        `export_to_netcdf`, so the smallest value would be read back as
        missing. The scale factor is 0 for constant arrays and NaN for
        all-NaN arrays.
        """
        return L2bProduct.scale_and_offset_from_range(np.nanmin(array),
                                                      np.nanmax(array),
                                                      nbit=nbit)

    @staticmethod
    def scale_and_offset_from_range(min_, max_, nbit=16):
        """Packing parameters from a known data range (see `compute_scale_and_offset`).

        Lets the range of many (dask) variables be computed in a single pass
        and converted afterwards.

        Parameters
        ----------
        min_, max_ : float
            Minimum and maximum of the data.
        nbit : int, optional
            Number of bits of the packed integer type (default 16).

        Returns
        -------
        scale_factor, add_offset : float
        """
        # stretch/compress data to the available packed range
        scale_factor = (max_ - min_) / (2 ** nbit - 1)
        # translate the range to be symmetric about zero
        add_offset = min_ + 2 ** (nbit - 1) * scale_factor
        return scale_factor, add_offset

    def export(self, ofile, persist=False):
        """Export to Zarr or NetCDF depending on the extension of `ofile`.

        Parameters
        ----------
        ofile : str
            Output path. A path ending in ``.zarr`` is written as a Zarr
            store (`export_to_zarr`), anything else as NetCDF
            (`export_to_netcdf`).
        persist : bool, optional
            See `export_to_netcdf`.
        """
        if str(ofile).rstrip('/\\').endswith('.zarr'):
            return self.export_to_zarr(ofile, persist=persist)
        return self.export_to_netcdf(ofile, persist=persist)

    def _packing_encoding(self, persist=False):
        """Build the packing part of the output encoding (format independent).

        ``flags`` and ``mask`` keep their dtype. All other variables are
        packed into ``int16`` with per-variable ``scale_factor`` and
        ``add_offset`` (see `compute_scale_and_offset`) and
        ``_FillValue = -32768``. For lazy (dask) variables the minimum and
        maximum of all packed variables are obtained in a single pass.

        Parameters
        ----------
        persist : bool, optional
            Compute the whole dataset once and keep it in memory (replaces
            ``self.l2b_prod``) before the ranges are determined.

        Returns
        -------
        dict
            ``{variable: encoding}``, each entry including
            ``grid_mapping = 'spatial_ref'``.
        """
        if persist:
            self.l2b_prod = self.l2b_prod.persist()

        packed = [v for v in self.variables if v not in ['mask', 'flags']]
        stats = {}
        if packed:
            ranges = dask.compute(*[(self.l2b_prod[v].min(skipna=True),
                                     self.l2b_prod[v].max(skipna=True))
                                    for v in packed])
            stats = {v: (float(lo), float(hi)) for v, (lo, hi) in zip(packed, ranges)}

        encoding = {}
        for variable in self.variables:
            if variable in ['mask', 'flags']:
                encoding[variable] = {"grid_mapping": "spatial_ref"}
            else:
                scale_factor, add_offset = self.scale_and_offset_from_range(*stats[variable], nbit=16)
                encoding[variable] = {
                    'dtype': 'int16',
                    'scale_factor': scale_factor,
                    'add_offset': add_offset,
                    '_FillValue': -32768,
                    "grid_mapping": "spatial_ref"
                }
        return encoding

    @staticmethod
    def _prepare_output(ofile):
        """Delete an existing output file or directory and create the parent directory."""
        if os.path.isdir(ofile):
            shutil.rmtree(ofile)
        elif os.path.exists(ofile):
            os.remove(ofile)

        odir = os.path.dirname(str(ofile).rstrip('/\\'))
        if odir == '':
            odir = './'
        if not os.path.exists(odir):
            os.makedirs(odir)

    def export_to_netcdf(self,
                         ofile,
                         zarr=False,
                         persist=False):
        """Write the L2B dataset to a compressed NetCDF file.

        ``flags`` and ``mask`` are written with their own dtype. All other
        variables are packed into ``int16`` with a per-variable
        ``scale_factor`` and ``add_offset`` (see `compute_scale_and_offset`),
        ``_FillValue = -32768``. All variables are zlib-compressed at level
        `complevel` and linked to the ``spatial_ref`` grid mapping. An
        existing `ofile` is deleted and missing output directories are
        created. The dataset is closed after writing.

        Parameters
        ----------
        ofile : str
            Output file path.
        zarr : bool, optional
            Currently unused.
        persist : bool, optional
            If True, the whole dataset is computed once and held in memory
            before the packing ranges are determined and the file is written
            (the lazy graph is evaluated once instead of twice). Needs enough
            RAM for all variables (default False).

        Notes
        -----
        With lazy (dask) variables the minimum and maximum of all packed
        variables are computed together in one pass, then the file is
        written in a second pass. The NetCDF chunk sizes follow the dask
        chunks.

        Returns
        -------
        None
        """
        if zarr:
            return self.export_to_zarr(ofile, persist=persist)

        logging.info('export into encoded netcdf')
        encoding = self._packing_encoding(persist=persist)
        for variable, enc in encoding.items():
            enc.update({"zlib": True, "complevel": self.complevel})
            # align NetCDF chunks with the dask chunks
            chunks = self.l2b_prod[variable].chunks
            if chunks is not None:
                enc['chunksizes'] = tuple(c[0] for c in chunks)

        # write file
        self._prepare_output(ofile)
        self.l2b_prod.to_netcdf(ofile, encoding=encoding)
        self.l2b_prod.close()

        return

    def export_to_zarr(self,
                       ofile,
                       persist=False,
                       chunks=None):
        """Write the L2B dataset to a Zarr store.

        Same content and packing as `export_to_netcdf` (``int16`` with
        ``scale_factor``/``add_offset`` and ``_FillValue = -32768`` for the
        geophysical variables, ``flags`` and ``mask`` unpacked), written with
        the default Zarr compressor. The store can be read back with
        ``xr.open_zarr`` (or ``Product``), which unpacks the values
        automatically. An existing `ofile` is deleted first.

        Parameters
        ----------
        ofile : str
            Output store path, typically ending in ``.zarr``.
        persist : bool, optional
            Compute the whole dataset once and keep it in memory before the
            ranges are determined and the store is written (needs enough
            RAM; default False).
        chunks : dict, optional
            Chunk sizes per dimension for the store, e.g.
            ``{'y': 1024, 'x': 1024}``. Default: the dask chunk size of the
            dataset (first block per dimension).

        Notes
        -----
        Zarr requires dask chunks to be regular and aligned with the store
        chunks, so the dataset is rechunked to one uniform chunk size per
        dimension before writing. Encodings inherited from the input
        product (e.g. NetCDF chunking or compression settings) are
        discarded on the data variables.

        Returns
        -------
        None
        """
        logging.info('export into encoded zarr')
        encoding = self._packing_encoding(persist=persist)

        # work on a shallow copy so the input product is not modified
        ds = self.l2b_prod.copy(deep=False)
        for variable in self.variables:
            ds[variable].encoding = {}

        # regular chunks, aligned with the zarr chunks
        if chunks is None:
            chunks = {}
            for variable in self.variables:
                if ds[variable].chunks is not None:
                    for dim, c in zip(ds[variable].dims, ds[variable].chunks):
                        chunks.setdefault(dim, c[0])
        if chunks:
            ds = ds.chunk(chunks)

        for variable, enc in encoding.items():
            if ds[variable].chunks is not None:
                enc['chunks'] = tuple(c[0] for c in ds[variable].chunks)

        self._prepare_output(ofile)
        ds.to_zarr(ofile, mode='w', encoding=encoding)
        self.l2b_prod.close()

        return

