"""Level-2B product assembly and export to compressed, packed NetCDF."""

import os
import dask
import shutil

import numpy as np
import xarray as xr
import zarr
import logging
import datetime

from . import __package__, __version__
from .mask import l2b_flags

# integer variables: not packed, nearest-neighbour resampling in the pyramid
CATEGORICAL = ('flags', 'flags_l2b', 'mask')
# OWT class IDs (owt_index_<database>, 1-based, NaN where not classified):
# written as uint8 with _FillValue 0, nearest-neighbour resampling in the pyramid
CLASS_ID_PREFIX = 'owt_index'


def _is_categorical(variable):
    """True for the variables that must keep exact values (flags, mask, class IDs)."""
    return variable in CATEGORICAL or variable.startswith(CLASS_ID_PREFIX)


def _json_safe(value):
    """Convert numpy scalars/arrays (recursively) to plain JSON-serializable Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


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
    flags : dict, optional
        L2B processing flags {name: boolean DataArray}, written as the bits
        of ``flags_l2b`` (see `GRSl2bgen.mask.L2B_FLAGS`).
    flag_descriptions : dict, optional
        {name: description} replacing the default description of these
        flags (e.g. with the thresholds used).
    complevel : int, optional
        zlib compression level of the NetCDF export (default 1). On a full
        Sentinel-2 tile, level 5 writes about 1.5 times slower for files about
        10 % smaller. Not used for Zarr output.

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
        zlib compression level of the NetCDF export.
    """

    def __init__(self, prod, l2_raster_list, flags=None, flag_descriptions=None, complevel=1):
        self.processor = __package__ + '_' + __version__
        self.prod = prod
        self.l2b_raster_list = l2_raster_list
        self.l2b_flags = flags or {}
        self.l2b_flag_descriptions = flag_descriptions or {}
        self.variables = None
        self.l2b_prod = None
        self.complevel = complevel
        self.construct_l2b()

    def construct_l2b(self):
        """Assemble the L2B dataset and set ``self.l2b_prod`` and ``self.variables``.

        Steps:

        1. Merge the L2 rasters with ``compat='override'`` (for conflicting
           values the first object wins).
        2. Add ``flags`` and ``mask`` from the input product; if absent,
           create them as ``uint8`` arrays of zeros on the image grid. Add
           ``flags_l2b``, the bit field of the L2B processing flags
           (`GRSl2bgen.mask.l2b_flags`).
        3. Convert the ``float64`` geophysical variables to ``float32``: they
           are packed into ``int16`` on export, far coarser than ``float32``,
           and the conversion halves the memory of the product.
        4. Copy the attributes of the input raster and add
           ``processing_time`` (local time, ISO 8601) and ``processor``.

        Notes
        -----
        ``l2b_prod.attrs`` is the same dict object as ``prod.raster.attrs``,
        so the two new attributes are also written to the input product.
        """

        logging.info('construct l2b')

        l2b_prod = xr.merge(self.l2b_raster_list, compat='override')
        for v in l2b_prod.data_vars:
            if l2b_prod[v].dtype == np.float64:
                l2b_prod[v] = l2b_prod[v].astype(np.float32)

        if 'flags' in self.prod.raster.keys():
            l2b_prod['flags'] = self.prod.raster['flags']
        else:
            l2b_prod['flags'] = xr.zeros_like(self.prod.raster.Rrs.isel(wl=0, drop=True).squeeze().astype(np.uint8))
        l2b_prod['flags_l2b'] = l2b_flags(self.prod.raster.Rrs.isel(wl=0, drop=True).squeeze(), self.l2b_flags,
                                          self.l2b_flag_descriptions)
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

        The data range ``[min, max]`` is stretched over the packed values
        ``[-(2**(nbit - 1) - 1), 2**(nbit - 1) - 1]`` (``[-32767, 32767]``
        for 16 bits), so that
        ``packed = (value - add_offset) / scale_factor`` and
        ``value = packed * scale_factor + add_offset``. The lowest integer,
        ``-2**(nbit - 1)`` (-32768), is left free for the ``_FillValue``
        used on export.

        Parameters
        ----------
        array : array-like
            Data to pack; NaN values are ignored.
        nbit : int, optional
            Number of bits of the packed integer type (default 16).

        Returns
        -------
        scale_factor : float
            ``(max - min) / (2**nbit - 2)``.
        add_offset : float
            ``min + (2**(nbit - 1) - 1) * scale_factor``.

        Notes
        -----
        For a constant array, ``scale_factor = 1`` and ``add_offset = min``
        (all values packed to 0). For an all-NaN array, ``scale_factor = 1``
        and ``add_offset = 0`` (all values packed to the fill value).
        """
        # fmin/fmax ignore NaN like nanmin/nanmax, without the 'All-NaN slice'
        # warning: an all-NaN array gives NaN, handled in scale_and_offset_from_range
        array = np.asarray(array)
        return L2bProduct.scale_and_offset_from_range(np.fmin.reduce(array, axis=None),
                                                      np.fmax.reduce(array, axis=None),
                                                      nbit=nbit)

    @staticmethod
    def scale_and_offset_from_range(min_, max_, nbit=16):
        """Packing parameters from a known data range (see `compute_scale_and_offset`).

        Lets the range of many (dask) variables be computed in a single pass
        and converted afterward.

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
        if not (np.isfinite(min_) and np.isfinite(max_)):
            # no valid data: any non-zero scale factor works
            return 1., 0.
        if max_ == min_:
            # constant field: avoid a zero scale factor (division by zero)
            return 1., float(min_)
        # stretch the data over [-(2**(nbit-1) - 1), 2**(nbit-1) - 1]; the lowest
        # integer, -2**(nbit-1), is kept for the _FillValue
        scale_factor = (max_ - min_) / (2 ** nbit - 2)
        add_offset = min_ + (2 ** (nbit - 1) - 1) * scale_factor
        return scale_factor, add_offset

    def export(self, ofile, persist=False, pyramid=False):
        """Export to Zarr or NetCDF depending on the extension of `ofile`.

        Parameters
        ----------
        ofile : str
            Output path. A path ending in ``.zarr`` is written as a Zarr
            store, anything else as NetCDF (`export_to_netcdf`).
        persist : bool, optional
            See `export_to_netcdf`.
        pyramid : bool, optional
            For Zarr output, write a multiscale (pyramid) store with
            `export_to_zarr_pyramid` instead of a single-resolution store
            (`export_to_zarr`). Ignored, with a warning, for NetCDF.

        Notes
        -----
        The product is written under a temporary hidden name
        (``.<name>.part<pid><ext>``) in the output directory and renamed to
        `ofile` once complete, replacing any existing `ofile`. An exception
        removes the temporary output; a killed process may leave it behind,
        but never an incomplete `ofile`.
        """
        ofile = str(ofile).rstrip('/\\')
        if pyramid and not ofile.endswith('.zarr'):
            logging.warning('pyramids are only written for Zarr output; ignored for NetCDF')

        # written under a temporary name in the same directory, renamed at the
        # end: an interrupted run never leaves an incomplete file under `ofile`
        root, ext = os.path.splitext(os.path.basename(ofile))
        tmp = os.path.join(os.path.dirname(ofile), f'.{root}.part{os.getpid()}{ext}')
        try:
            if not ofile.endswith('.zarr'):
                self.export_to_netcdf(tmp, persist=persist)
            elif pyramid:
                self.export_to_zarr_pyramid(tmp, persist=persist)
            else:
                self.export_to_zarr(tmp, persist=persist)
        except BaseException:
            self._remove(tmp)
            raise
        self._remove(ofile)
        os.replace(tmp, ofile)

    def _packing_encoding(self, persist=False):
        """Build the packing part of the output encoding (format independent).

        ``flags``, ``flags_l2b`` and ``mask`` keep their dtype. The OWT class IDs
        (``owt_index_*``) are written as ``uint8`` with ``_FillValue = 0``
        (class IDs start at 1), so that they read back as exact integers. All
        other variables are
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

        packed = [v for v in self.variables if not _is_categorical(v)]
        stats = {}
        if packed:
            # NaN replaced by +/-inf: same result as min/max(skipna=True) without
            # the 'All-NaN slice' warning raised for every all-NaN chunk (land,
            # clouds); an all-NaN variable gives +/-inf, i.e. no valid data
            ranges = dask.compute(*[(self.l2b_prod[v].fillna(np.inf).min(skipna=False),
                                     self.l2b_prod[v].fillna(-np.inf).max(skipna=False))
                                    for v in packed])
            stats = {v: (float(lo), float(hi)) for v, (lo, hi) in zip(packed, ranges)}

        encoding = {}
        for variable in self.variables:
            if variable in CATEGORICAL:
                encoding[variable] = {"grid_mapping": "spatial_ref"}
            elif variable.startswith(CLASS_ID_PREFIX):
                encoding[variable] = {'dtype': 'uint8',
                                      '_FillValue': 0,
                                      "grid_mapping": "spatial_ref"}
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
    def _remove(path):
        """Delete a file or directory if it exists."""
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _prepare_output(ofile):
        """Delete an existing output file or directory and create the parent directory."""
        L2bProduct._remove(ofile)

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

        ``flags``, ``flags_l2b`` and ``mask`` are written with their own dtype. All other
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

        # NetCDF has no boolean attribute type (e.g. ``gas_absorption_correction``
        # inherited from the L2A Rrs): write them as 0/1 integers
        ds = self.l2b_prod.copy(deep=False)
        for attrs in [ds.attrs] + [v.attrs for v in ds.variables.values()]:
            for key, value in attrs.items():
                if isinstance(value, (bool, np.bool_)):
                    attrs[key] = np.int8(value)

        # write file
        self._prepare_output(ofile)
        ds.to_netcdf(ofile, encoding=encoding)
        self.l2b_prod.close()

        return

    def _write_zarr_group(self, ds, ofile, encoding, chunks,
                          group=None, mode='w', consolidated=None):
        """Write `ds` into a Zarr store (or one group of it) with regular chunks.

        Works on a shallow copy, discards encodings inherited from the input
        product on the data variables, rechunks the dataset to the uniform
        sizes in `chunks` (Zarr needs dask chunks aligned with the store
        chunks) and writes it with `encoding` plus the matching ``chunks``.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing at least the variables in ``self.variables``.
        ofile : str
            Store path.
        encoding : dict
            Packing encoding from `_packing_encoding`.
        chunks : dict
            Chunk size per dimension (dimensions not in `ds` are ignored).
        group : str, optional
            Group inside the store (default: root).
        mode : str, optional
            ``to_zarr`` write mode (default ``'w'``).
        consolidated : bool, optional
            Passed to ``to_zarr``.
        """
        ds = ds.copy(deep=False)
        for variable in self.variables:
            ds[variable].encoding = {}
        # coordinates read from a Zarr L2A keep its (zarr v2) compressors,
        # refused by the zarr v3 writer
        for coord in ds.coords.values():
            for key in ('compressor', 'compressors', 'filters'):
                coord.encoding.pop(key, None)

        chunks = {d: s for d, s in chunks.items() if d in ds.dims}
        if chunks:
            ds = ds.chunk(chunks)

        enc = {}
        for variable in self.variables:
            enc[variable] = dict(encoding[variable])
            if ds[variable].chunks is not None:
                enc[variable]['chunks'] = tuple(c[0] for c in ds[variable].chunks)

        ds.to_zarr(ofile, group=group, mode=mode, encoding=enc, consolidated=consolidated)

    def export_to_zarr(self,
                       ofile,
                       persist=False,
                       chunks=None):
        """Write the L2B dataset to a Zarr store.

        Same content and packing as `export_to_netcdf` (``int16`` with
        ``scale_factor``/``add_offset`` and ``_FillValue = -32768`` for the
        geophysical variables, ``flags``, ``flags_l2b`` and ``mask`` unpacked), written with
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

        if chunks is None:
            chunks = {}
            for variable in self.variables:
                var_chunks = self.l2b_prod[variable].chunks
                if var_chunks is not None:
                    for dim, c in zip(self.l2b_prod[variable].dims, var_chunks):
                        chunks.setdefault(dim, c[0])

        self._prepare_output(ofile)
        self._write_zarr_group(self.l2b_prod, ofile, encoding, chunks)
        self.l2b_prod.close()

        return

    @staticmethod
    def _downsample(ds, dims=('y', 'x'), factor=2, categorical=None):
        """Reduce the spatial resolution of `ds` by an integer `factor`.

        Continuous variables are block-averaged ignoring NaN. Categorical
        variables (``flags``, ``flags_l2b``, ``mask``, ``owt_index_*``) are taken from the top-left pixel of
        each block (nearest neighbour), which preserves their values and
        dtype. Edge pixels that do not fill a complete block are trimmed, so
        the origin is unchanged and the pixel size is multiplied by `factor`.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset with the spatial dimensions `dims`.
        dims : tuple of str, optional
            Names of the (y, x) dimensions.
        factor : int, optional
            Block size (default 2).
        categorical : tuple of str, optional
            Variables resampled by nearest neighbour. Default: the flags, mask
            and OWT class IDs of `ds`.

        Returns
        -------
        xarray.Dataset
        """
        if categorical is None:
            categorical = [v for v in ds.data_vars if _is_categorical(v)]
        window = {d: factor for d in dims}
        continuous = [v for v in ds.data_vars if v not in categorical]
        coarse = ds[continuous].coarsen(window, boundary='trim').mean()
        for v in continuous:
            coarse[v].attrs = ds[v].attrs

        for v in categorical:
            if v in ds.data_vars:
                sub = ds[v].isel({d: slice(0, coarse.sizes[d] * factor, factor) for d in dims})
                sub = sub.assign_coords({d: coarse[d] for d in dims})
                coarse[v] = sub
                coarse[v].attrs = ds[v].attrs

        coarse.attrs = ds.attrs
        return coarse

    @staticmethod
    def _update_geotransform(ds, dims=('y', 'x')):
        """Rewrite the ``GeoTransform`` of the ``spatial_ref`` variable from the coordinates.

        Needed after resampling so that GIS readers see the right pixel size.
        Failures (e.g. no CRS or non-georeferenced data) only log a warning.
        """
        try:
            ds = ds.rio.set_spatial_dims(x_dim=dims[1], y_dim=dims[0])
            ds = ds.rio.write_transform(ds.rio.transform(recalc=True))
        except Exception as e:
            logging.warning(f'could not update geotransform: {e}')
        return ds

    def export_to_zarr_pyramid(self,
                               ofile,
                               persist=False,
                               levels=None,
                               min_size=256,
                               chunks=None,
                               dims=('y', 'x'),
                               factor=2):
        """Write a cloud-ready multiscale (pyramid) Zarr store.

        Layout: one group per resolution level, ``0`` = full resolution,
        ``1`` = `factor` times coarser, and so on, each holding the full
        L2B dataset with the same packing as `export_to_zarr`. The root
        group carries the product attributes and a ``multiscales`` attribute
        (level layout, scale factors, resampling method, following the
        evolving Zarr "multiscales" convention) and the store metadata is
        consolidated, so a viewer or client reads a single small file to
        discover everything and then fetches only the chunks it needs.

        Resampling: block average (NaN-aware) for continuous variables,
        nearest neighbor for ``flags``, ``flags_l2b``, ``mask`` and ``owt_index_*``. The geotransform is
        updated at every level. All levels use the packing parameters
        (``scale_factor``/``add_offset``) of level 0.

        Parameters
        ----------
        ofile : str
            Output store path, typically ending in ``.zarr``. An existing
            store is deleted first.
        persist : bool, optional
            Compute the full-resolution dataset once and keep it in memory
            (needs enough RAM; default False).
        levels : int, optional
            Number of extra levels below level 0. Default: as many as keep
            the smaller image side at least `min_size` pixels.
        min_size : int, optional
            Stop when the next level would have a side smaller than this
            (default 256).
        chunks : dict, optional
            Chunk sizes per dimension, same for all levels. Default 512 x 512,
            a good size for web access.
        dims : tuple of str, optional
            Names of the (y, x) dimensions (default ``('y', 'x')``).
        factor : int, optional
            Downsampling factor between successive levels (default 2).

        Notes
        -----
        Level 0 is the only pass over the lazy (dask) graph. Each further
        level is computed from the level just written (read back from the
        store, hence from the packed int16 values), so the upstream
        processing is never repeated. Open a level with
        ``xr.open_zarr(ofile, group='2')``; the root group itself holds no
        data variables.

        Returns
        -------
        None
        """
        logging.info('export into encoded multiscale zarr')
        for d in dims:
            if d not in self.l2b_prod.dims:
                raise ValueError(f'dimension {d!r} not found in the L2B product')
        if chunks is None:
            chunks = {dims[0]: 512, dims[1]: 512}

        encoding = self._packing_encoding(persist=persist)
        self._prepare_output(ofile)

        # level 0: the only evaluation of the processing graph
        self._write_zarr_group(self.l2b_prod, ofile, encoding, chunks,
                               group='0', mode='a', consolidated=False)
        self.l2b_prod.close()

        layout = [{'asset': '0'}]
        level = 0
        while levels is None or level < levels:
            prev = xr.open_zarr(ofile, group=str(level), consolidated=False,
                                decode_coords='all')
            small_side = min(prev.sizes[d] for d in dims)
            if small_side // factor < max(min_size, 1):
                prev.close()
                break
            coarse = self._downsample(prev, dims=dims, factor=factor)
            coarse = self._update_geotransform(coarse, dims=dims)
            level += 1
            self._write_zarr_group(coarse, ofile, encoding, chunks,
                                   group=str(level), mode='a', consolidated=False)
            prev.close()
            layout.append({'asset': str(level),
                           'derived_from': str(level - 1),
                           'transform': {'scale': [float(factor), float(factor)]},
                           'resampling_method': 'average'})

        # root group: product attributes + multiscale description

        root = zarr.open_group(ofile, mode='r+')
        root.attrs.update(_json_safe(dict(self.l2b_prod.attrs)))
        root.attrs['multiscales'] = {
            'layout': layout,
            'resampling_method': 'average',
            'categorical_variables': [v for v in self.variables if _is_categorical(v)],
            'categorical_resampling_method': 'nearest',
            'dims': list(dims),
        }
        # one small metadata file for the whole hierarchy: fewer requests in the cloud
        try:
            zarr.consolidate_metadata(ofile)
        except Exception as e:
            logging.warning(f'could not consolidate zarr metadata: {e}')

        return

