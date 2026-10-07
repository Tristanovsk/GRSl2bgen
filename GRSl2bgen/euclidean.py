"""Euclidean distance between pixel spectra and Optical Water Type (OWT) spectra."""

import numpy as np
import xarray as xr
from numba import njit, prange


def _euclid_core(Rrs, owt, Nsaved):
    """Per-pixel Euclidean distances to the `Nsaved` closest classes.

    Parameters
    ----------
    Rrs : ndarray, shape (Ny, Nx, Nwl)
        Reflectance spectra, wavelength axis last (C-contiguous is fastest).
    owt : ndarray, shape (Nclasses, Nwl), float64
        Class spectra (used as is, no normalisation).
    Nsaved : int
        Number of closest classes to keep.

    Returns
    -------
    arr_dist : ndarray, shape (Ny, Nx, Nsaved), float32
        Euclidean distances (same units as Rrs), ascending; NaN for invalid
        pixels or when ``Nsaved > Nclasses``.
    arr_index : ndarray, shape (Ny, Nx, Nsaved), float32
        1-based class IDs matching `arr_dist`; NaN where `arr_dist` is NaN.
    """
    Ny, Nx, Nwl = Rrs.shape
    Nclasses = owt.shape[0]
    Nk = min(Nsaved, Nclasses)

    arr_dist = np.full((Ny, Nx, Nsaved), np.nan, dtype=np.float32)
    arr_index = np.full((Ny, Nx, Nsaved), np.nan, dtype=np.float32)

    for iy in prange(Ny):
        sq = np.empty(Nclasses, dtype=np.float64)  # private to each row/thread
        for ix in range(Nx):
            # skip pixels with any non-finite band
            valid = True
            for iwl in range(Nwl):
                if not np.isfinite(Rrs[iy, ix, iwl]):
                    valid = False
                    break
            if not valid:
                continue

            # squared distance to every class
            for iowt in range(Nclasses):
                acc = 0.
                for iwl in range(Nwl):
                    d = float(Rrs[iy, ix, iwl]) - owt[iowt, iwl]
                    acc += d * d
                sq[iowt] = acc

            # the distance increases with the squared distance: select on the
            # squared value and call sqrt only for the classes that are kept
            for k in range(Nk):
                best = 0
                bestval = np.inf
                for iowt in range(Nclasses):
                    if sq[iowt] < bestval:
                        bestval = sq[iowt]
                        best = iowt
                arr_dist[iy, ix, k] = np.sqrt(bestval)
                arr_index[iy, ix, k] = best + 1
                sq[best] = np.inf  # exclude from the next pass

    return arr_dist, arr_index


# Same source compiled twice. Parallel version: for one large in-memory array.
# Serial version: for use inside dask tasks (dask already provides the
# parallelism; concurrent parallel numba calls from several threads are not
# safe with numba's default threading layer).
_euclid_parallel = njit(parallel=True)(_euclid_core)
_euclid_serial = njit()(_euclid_core)


def _check_classes(Rrs_owt):
    """Return class spectra as a finite float64 array."""
    owt = np.asarray(Rrs_owt, dtype=np.float64)
    if not np.all(np.isfinite(owt)):
        raise ValueError('OWT spectra must be finite')
    return owt


def euclidean(Rrs, Rrs_owt, Nclasses_to_be_saved=3, parallel=True):
    """Euclidean distances between pixel spectra and OWT spectra (numpy arrays).

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
        Distances (ascending) and 1-based class IDs; NaN for invalid
        pixels. Non-contiguous views of the internal (Ny, Nx, K) result.
    """
    if Rrs.ndim != 3 or Rrs.shape[-1] != np.shape(Rrs_owt)[1]:
        raise ValueError('Rrs must have shape (Ny, Nx, Nwl) matching Rrs_owt (Nclasses, Nwl)')
    kernel = _euclid_parallel if parallel else _euclid_serial
    dist, index = kernel(np.ascontiguousarray(Rrs), _check_classes(Rrs_owt),
                         int(Nclasses_to_be_saved))
    return np.moveaxis(dist, -1, 0), np.moveaxis(index, -1, 0)


def euclidean_dataarray(Rrs, Rrs_owt, Nclasses_to_be_saved=3, wl_dim='wl', name_dim='Nowt'):
    """Euclidean distances for an xarray reflectance cube; lazy if `Rrs` is dask-backed.

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
        Dimensions (Nowt, y, x): distances (ascending) and 1-based class
        IDs, float32, named ``owt_dist`` and ``owt_index``.
    """
    if Rrs.ndim != 3:
        raise ValueError('Rrs must have exactly three dimensions (y, x, wl)')
    owt = _check_classes(Rrs_owt)
    if owt.shape[1] != Rrs.sizes[wl_dim]:
        raise ValueError('Rrs_owt and Rrs have a different number of wavelengths')
    nsaved = int(Nclasses_to_be_saved)

    is_dask = Rrs.chunks is not None
    kernel = _euclid_serial if is_dask else _euclid_parallel
    spatial = [d for d in Rrs.dims if d != wl_dim]
    Rrs = Rrs.transpose(*spatial, wl_dim)
    if is_dask:
        Rrs = Rrs.chunk({wl_dim: -1})

    def _block(block):
        return kernel(np.ascontiguousarray(block), owt, nsaved)

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