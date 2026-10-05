"""Spectral Angle Mapper (SAM) between pixel spectra and Optical Water Type (OWT) spectra."""

import numpy as np
import xarray as xr
from numba import njit, prange


def _sam_core(Rrs, owt_unit, Nsaved):
    """Per-pixel angles to the `Nsaved` closest classes.

    Parameters
    ----------
    Rrs : ndarray, shape (Ny, Nx, Nwl)
        Reflectance spectra, wavelength axis last (C-contiguous is fastest).
    owt_unit : ndarray, shape (Nclasses, Nwl), float64
        Class spectra normalised to unit Euclidean norm.
    Nsaved : int
        Number of closest classes to keep.

    Returns
    -------
    arr_dist : ndarray, shape (Ny, Nx, Nsaved), float32
        Spectral angles in radians, ascending; NaN for invalid pixels or
        when ``Nsaved > Nclasses``.
    arr_index : ndarray, shape (Ny, Nx, Nsaved), float32
        1-based class IDs matching `arr_dist`; NaN where `arr_dist` is NaN.
    """
    Ny, Nx, Nwl = Rrs.shape
    Nclasses = owt_unit.shape[0]
    Nk = min(Nsaved, Nclasses)

    arr_dist = np.full((Ny, Nx, Nsaved), np.nan, dtype=np.float32)
    arr_index = np.full((Ny, Nx, Nsaved), np.nan, dtype=np.float32)

    for iy in prange(Ny):
        cosines = np.empty(Nclasses, dtype=np.float64)  # private to each row/thread
        for ix in range(Nx):
            # pixel norm; skip pixels with any non-finite band or a null spectrum
            norm2 = 0.
            valid = True
            for iwl in range(Nwl):
                v = float(Rrs[iy, ix, iwl])
                if not np.isfinite(v):
                    valid = False
                    break
                norm2 += v * v
            if (not valid) or norm2 == 0.:
                continue
            inv_norm = 1. / np.sqrt(norm2)

            # cosine of the angle to every class (class spectra are unit vectors)
            for iowt in range(Nclasses):
                dot = 0.
                for iwl in range(Nwl):
                    dot += float(Rrs[iy, ix, iwl]) * owt_unit[iowt, iwl]
                cosines[iowt] = dot * inv_norm

            # the angle decreases as the cosine increases: select on the cosine
            # and call arccos only for the classes that are kept
            for k in range(Nk):
                best = 0
                bestval = -np.inf
                for iowt in range(Nclasses):
                    if cosines[iowt] > bestval:
                        bestval = cosines[iowt]
                        best = iowt
                arr_dist[iy, ix, k] = np.arccos(min(1., max(-1., bestval)))
                arr_index[iy, ix, k] = best + 1
                cosines[best] = -np.inf  # exclude from the next pass

    return arr_dist, arr_index


# Same source compiled twice. Parallel version: for one large in-memory array.
# Serial version: for use inside dask tasks (dask already provides the
# parallelism; concurrent parallel numba calls from several threads are not
# safe with numba's default threading layer).
_sam_parallel = njit(parallel=True)(_sam_core)
_sam_serial = njit()(_sam_core)


def _unit_classes(Rrs_owt):
    """Normalise class spectra to unit Euclidean norm (float64)."""
    owt = np.asarray(Rrs_owt, dtype=np.float64)
    norm = np.sqrt((owt ** 2).sum(axis=1, keepdims=True))
    if not np.all(np.isfinite(norm)) or np.any(norm == 0.):
        raise ValueError('OWT spectra must be finite and non-zero')
    return owt / norm


def sam(Rrs, Rrs_owt, Nclasses_to_be_saved=3, parallel=True):
    """Spectral angles between pixel spectra and OWT spectra (numpy arrays).

    Parameters
    ----------
    Rrs : ndarray, shape (Ny, Nx, Nwl)
        Reflectance, wavelength axis last.
    Rrs_owt : ndarray, shape (Nclasses, Nwl)
        Class spectra at the same wavelengths.
    Nclasses_to_be_saved : int, optional
        Number of closest classes kept per pixel (default 3).
    parallel : bool, optional
        Use all CPU cores (default True).

    Returns
    -------
    dist, index : ndarray, shape (Nclasses_to_be_saved, Ny, Nx), float32
        Angles (radians, ascending) and 1-based class IDs; NaN for invalid
        pixels. Non-contiguous views of the internal (Ny, Nx, K) result.
    """
    if Rrs.ndim != 3 or Rrs.shape[-1] != np.shape(Rrs_owt)[1]:
        raise ValueError('Rrs must have shape (Ny, Nx, Nwl) matching Rrs_owt (Nclasses, Nwl)')
    kernel = _sam_parallel if parallel else _sam_serial
    dist, index = kernel(np.ascontiguousarray(Rrs), _unit_classes(Rrs_owt),
                         int(Nclasses_to_be_saved))
    return np.moveaxis(dist, -1, 0), np.moveaxis(index, -1, 0)


def sam_dataarray(Rrs, Rrs_owt, Nclasses_to_be_saved=3, wl_dim='wl', name_dim='Nowt'):
    """Spectral angles for an xarray reflectance cube; lazy if `Rrs` is dask-backed.

    Parameters
    ----------
    Rrs : xarray.DataArray
        Reflectance with dimensions (y, x, wl) in any order. For dask arrays
        the wavelength dimension is rechunked to a single chunk.
    Rrs_owt : array-like, shape (Nclasses, Nwl)
        Class spectra at the wavelengths of `Rrs`.
    Nclasses_to_be_saved : int, optional
        Number of closest classes kept per pixel (default 3).
    wl_dim : str, optional
        Name of the wavelength dimension.
    name_dim : str, optional
        Name of the new class-rank dimension (default ``'Nowt'``).

    Returns
    -------
    dist, index : xarray.DataArray
        Dimensions (Nowt, y, x): angles in radians (ascending) and 1-based
        class IDs, float32. Ready for ``Chl.owt_blending`` (``owt_dist`` and
        ``owt_index``).
    """
    if Rrs.ndim != 3:
        raise ValueError('Rrs must have exactly three dimensions (y, x, wl)')
    unit = _unit_classes(Rrs_owt)
    if unit.shape[1] != Rrs.sizes[wl_dim]:
        raise ValueError('Rrs_owt and Rrs have a different number of wavelengths')
    nsaved = int(Nclasses_to_be_saved)

    is_dask = Rrs.chunks is not None
    kernel = _sam_serial if is_dask else _sam_parallel
    spatial = [d for d in Rrs.dims if d != wl_dim]
    Rrs = Rrs.transpose(*spatial, wl_dim)
    if is_dask:
        Rrs = Rrs.chunk({wl_dim: -1})

    def _block(block):
        return kernel(np.ascontiguousarray(block), unit, nsaved)

    dist, index = xr.apply_ufunc(
        _block, Rrs,
        input_core_dims=[[wl_dim]],
        output_core_dims=[[name_dim], [name_dim]],
        dask='parallelized',
        output_dtypes=[np.float32, np.float32],
        dask_gufunc_kwargs={'output_sizes': {name_dim: nsaved}},
    )
    dist = dist.transpose(name_dim, *spatial).rename('owt_dist')
    index = index.transpose(name_dim, *spatial).rename('owt_index')
    return dist, index
