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
        logging.info('export into encoded netcdf')
        complevel = self.complevel
        encoding = {}

        if persist:
            self.l2b_prod = self.l2b_prod.persist()

        # one pass over the data for the ranges of all packed variables
        packed = [v for v in self.variables if v not in ['mask', 'flags']]
        stats = {}
        if packed:
            ranges = dask.compute(*[(self.l2b_prod[v].min(skipna=True),
                                     self.l2b_prod[v].max(skipna=True))
                                    for v in packed])
            stats = {v: (float(lo), float(hi)) for v, (lo, hi) in zip(packed, ranges)}
        for variable in self.variables:

            if variable in ['mask', 'flags']:
                encoding[variable] = {
                    "zlib": True,
                    "complevel": complevel,
                    "grid_mapping": "spatial_ref"
                }
            else:
                scale_factor, add_offset = self.scale_and_offset_from_range(*stats[variable], nbit=16)
                # offset = np.mean(p.range)
                # range = float(np.diff(p.range))
                # scale_factor = round(range / 60000, 6)
                encoding[variable] = {
                    'dtype': 'int16',
                    'scale_factor': scale_factor,
                    'add_offset': add_offset,
                    '_FillValue': -32768,
                    "zlib": True,
                    "complevel": complevel,
                    "grid_mapping": "spatial_ref"
                }

        # align NetCDF chunks with the dask chunks
        for variable, enc in encoding.items():
            chunks = self.l2b_prod[variable].chunks
            if chunks is not None:
                enc['chunksizes'] = tuple(c[0] for c in chunks)

        # write file
        if os.path.exists(ofile):
            os.remove(ofile)

        odir = os.path.dirname(ofile)
        if odir == '':
            odir = './'
        if not os.path.exists(odir):
            os.makedirs(odir)

        self.l2b_prod.to_netcdf(ofile, encoding=encoding)
        self.l2b_prod.close()

        return



    def export_to_netcdf_beta(self, ofile, zarr=False):
        """Write the L2B dataset to a compressed NetCDF file."""

        logging.info("export into encoded netcdf")

        complevel = self.complevel
        encoding = {}

        # Variables that should retain their native dtype
        native_vars = {"mask", "flags", "spatial_ref"}

        for variable in self.l2b_prod.data_vars:
            if variable in native_vars:
                encoding[variable] = {
                    "zlib": True,
                    "complevel": complevel,
                }
                continue

            p = self.l2b_prod[variable]

            scale_factor, add_offset = self.compute_scale_and_offset(
                p.values,
                nbit=16,
            )

            encoding[variable] = {
                "dtype": "int16",
                "scale_factor": scale_factor,
                "add_offset": add_offset,
                "_FillValue": -32768,
                "zlib": True,
                "complevel": complevel,
            }

        # Ensure output directory exists
        odir = os.path.dirname(os.path.abspath(ofile))
        os.makedirs(odir, exist_ok=True)

        # Remove existing file
        if os.path.exists(ofile):
            os.remove(ofile)

        self.l2b_prod.to_netcdf(
            ofile,
            encoding=encoding,
        )

        self.l2b_prod.close()

    def export_to_zarr(self, odir, overwrite=True):
        """Write the L2B dataset to a compressed Zarr store.

        ``flags`` and ``mask`` are written with their own dtype. All other
        variables are packed into ``int16`` with a per-variable
        ``scale_factor`` and ``add_offset``.

        Parameters
        ----------
        odir : str
            Output Zarr store path.
        overwrite : bool, optional
            If True, an existing Zarr store is replaced.

        Returns
        -------
        None
        """
        logging.info("export into encoded zarr")

        # Remove existing Zarr store
        if overwrite and os.path.exists(odir):
            shutil.rmtree(odir)

        # Create a copy so that we don't modify self.l2b_prod
        ds = self.l2b_prod.copy()

        native_vars = {"mask", "flags", "spatial_ref"}

        # Pack variables into int16
        for variable in ds.data_vars:

            if variable in native_vars:
                continue

            p = ds[variable]

            scale_factor, add_offset = self.compute_scale_and_offset(
                p.values,
                nbit=16,
            )

            # Encode the data explicitly as int16
            ds[variable] = (
                p.dims,
                ((p.values - add_offset) / scale_factor).round().astype("int16"),
            )

            # Store packing information as attributes
            ds[variable].attrs["scale_factor"] = scale_factor
            ds[variable].attrs["add_offset"] = add_offset
            ds[variable].attrs["_FillValue"] = -32768

        # Zarr compressor
        compressor = numcodecs.Blosc(
            cname="zstd",
            clevel=self.complevel,
            shuffle=numcodecs.Blosc.BITSHUFFLE,
        )

        encoding = {}

        for variable in ds.data_vars:

            if variable in native_vars:
                encoding[variable] = {
                    "compressor": compressor,
                }
            else:
                encoding[variable] = {
                    "compressor": compressor,
                }

        # Write Zarr
        ds.to_zarr(
            odir,
            mode="w",
            encoding=encoding,
        )

        ds.close()
