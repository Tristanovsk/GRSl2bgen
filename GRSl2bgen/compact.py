"""Processing restricted to the valid pixels of an image (lazy, dask-friendly).

The retrievals of the L2B chain are pixel-wise, so they give the same values
whether the pixels are laid out on the image grid or side by side on any other
grid. `CompactGrid` packs the valid pixels (water, L2A ``mask == 0``) of the
image into a grid of one row, the chain runs on that grid, and the results are
put back on the image grid. The pixels left out are not computed at all: on a
tile with little water (coasts, rivers, lakes: often less than 10 % of the
pixels) most of the processing is saved.

The packing is done block by block (the spatial dask chunks of the image): the
chunk ``k`` of the compact grid holds the valid pixels of the ``k``-th block
that has any, in row-major order. Both directions are therefore made of
independent per-block tasks, and nothing is read or computed before the result
is.
"""

import numpy as np
import dask.array as da
import xarray as xr

# spatial chunk of an image given as numpy arrays (see product.SPATIAL_CHUNK)
DEFAULT_CHUNK = 1024


def _take(block, valid):
    """Valid pixels of an image block, as one row: (..., ny, nx) -> (..., 1, n)."""
    return block[..., valid][..., None, :]


def _put(block, valid, fill):
    """Image block of the pixels of a compact chunk: (..., 1, n) -> (..., ny, nx)."""
    out = np.full(block.shape[:-2] + valid.shape, fill, dtype=block.dtype)
    out[..., valid] = block[..., 0, :]
    return out


def _fill_value(dtype):
    """Value of the pixels left out: NaN, False or 0 depending on `dtype`."""
    dtype = np.dtype(dtype)
    if dtype.kind in 'fc':
        return np.nan
    if dtype.kind == 'b':
        return False
    return 0


def _regular_chunks(size, chunk):
    return tuple(min(chunk, size - start) for start in range(0, size, chunk))


class CompactGrid():
    """Valid pixels of an image packed block by block into a grid of one row.

    Parameters
    ----------
    valid : numpy.ndarray of bool, shape (ny, nx)
        Pixels to keep.
    chunks : tuple of two tuples of int, optional
        Spatial blocks of the image, as the dask chunks along ``y`` and ``x``.
        Default: regular blocks of `DEFAULT_CHUNK` pixels.
    y, x : str, optional
        Names of the spatial dimensions (default ``'y'`` and ``'x'``).

    Attributes
    ----------
    size : int
        Number of pixels of the compact grid.
    counts : tuple of int
        Number of pixels of each compact chunk.

    Notes
    -----
    If `valid` has no pixel, the first pixel of the image is kept anyway, so
    that the compact grid is never empty: the chain then runs on a single
    (masked) pixel instead of a zero-sized array.

    The pixels left out are given NaN (float), False (bool) or 0 (integer) by
    `unpack`. This is what the chain gives on the image grid for masked pixels
    (all inputs NaN); ``tests/test_compact.py`` checks it on the test product.

    Examples
    --------
    >>> grid = CompactGrid((raster['mask'] == 0).values, raster['mask'].chunks)
    >>> out = grid.unpack(algorithm(grid.pack(raster)))
    """

    def __init__(self, valid, chunks=None, y='y', x='x'):
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim != 2:
            raise ValueError('valid must be a 2-D array (y, x)')
        if not valid.any():
            valid = valid.copy()
            valid[0, 0] = True
        if chunks is None:
            chunks = tuple(_regular_chunks(n, DEFAULT_CHUNK) for n in valid.shape)
        chunks = tuple(tuple(int(c) for c in cs) for cs in chunks)
        if tuple(sum(cs) for cs in chunks) != valid.shape:
            raise ValueError(f'chunks {chunks} do not match the shape {valid.shape}')
        self.y, self.x = y, x
        self.shape = valid.shape
        self.chunks = chunks
        # blocks[iy][ix] = (mask of the block, index of its compact chunk or None)
        self.blocks = []
        counts = []
        y0 = 0
        for cy in chunks[0]:
            row = []
            x0 = 0
            for cx in chunks[1]:
                block = valid[y0:y0 + cy, x0:x0 + cx]
                n = int(block.sum())
                if n:
                    row.append((block, len(counts)))
                    counts.append(n)
                else:
                    row.append((block, None))
                x0 += cx
            self.blocks.append(row)
            y0 += cy
        self.counts = tuple(counts)
        self.size = sum(counts)

    def _is_spatial(self, var):
        return self.y in var.dims and self.x in var.dims

    def _spatial_last(self, var):
        """`var` with its spatial dimensions last, and the order to restore."""
        lead = [d for d in var.dims if d not in (self.y, self.x)]
        return var.transpose(*lead, self.y, self.x), var.dims

    def _other_coords(self, var):
        """Coordinates of `var` that do not depend on the spatial dimensions."""
        return {k: c for k, c in var.coords.items()
                if self.y not in c.dims and self.x not in c.dims}

    def pack(self, obj):
        """Valid pixels of `obj` on the compact grid (lazy).

        Parameters
        ----------
        obj : xarray.DataArray or xarray.Dataset
            Image (dask or numpy backed) with the spatial dimensions of the
            grid. Variables without both of them are kept as they are.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Same variables, attributes and dimension order; the spatial
            dimensions have sizes 1 (``y``) and `size` (``x``), without
            coordinates.
        """
        if isinstance(obj, xr.Dataset):
            out = xr.Dataset({k: self.pack(v) if self._is_spatial(v) else v
                              for k, v in obj.data_vars.items()},
                             coords={k: c for k, c in obj.coords.items()
                                     if self.y not in c.dims and self.x not in c.dims},
                             attrs=obj.attrs)
            return out
        var, dims = self._spatial_last(obj)
        if var.shape[-2:] != self.shape:
            raise ValueError(f'{obj.name}: spatial shape {var.shape[-2:]} is not {self.shape}')
        data = var.data
        lead = var.shape[:-2]
        if isinstance(data, da.Array):
            data = data.rechunk(tuple(-1 for _ in lead) + self.chunks)
        else:
            data = da.from_array(data, chunks=lead + self.chunks)
        parts = []
        for iy, row in enumerate(self.blocks):
            for ix, (valid, k) in enumerate(row):
                if k is None:
                    continue
                block = data.blocks[(0,) * len(lead) + (iy, ix)]
                parts.append(block.map_blocks(_take, valid=valid, dtype=data.dtype,
                                              chunks=lead + (1, self.counts[k])))
        packed = da.concatenate(parts, axis=-1)
        out = xr.DataArray(packed, dims=var.dims, coords=self._other_coords(var),
                           attrs=var.attrs, name=var.name)
        return out.transpose(*dims)

    def unpack(self, obj, coords=None):
        """Results on the compact grid put back on the image grid (lazy).

        Parameters
        ----------
        obj : xarray.DataArray or xarray.Dataset
            Variables on the compact grid (as returned by an algorithm
            applied to `pack`). Variables without both spatial dimensions are
            kept as they are.
        coords : mapping, optional
            Coordinates of the image along the spatial dimensions (``x``,
            ``y``, ...), given to the spatial variables of the result.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Same variables, attributes and dimension order on the image grid,
            chunked as the image; the pixels left out are NaN, False or 0
            depending on the dtype, with the spatial `coords`.
        """
        if isinstance(obj, xr.Dataset):
            return xr.Dataset({k: self.unpack(v, coords) if self._is_spatial(v) else v
                               for k, v in obj.data_vars.items()},
                              coords={k: c for k, c in obj.coords.items()
                                      if self.y not in c.dims and self.x not in c.dims},
                              attrs=obj.attrs)
        var, dims = self._spatial_last(obj)
        if var.shape[-2:] != (1, self.size):
            raise ValueError(f'{obj.name}: spatial shape {var.shape[-2:]} is not (1, {self.size})')
        data = var.data
        lead = var.shape[:-2]
        if not isinstance(data, da.Array):
            data = da.from_array(data, chunks=-1)
        # chunks of the other dimensions, restored at the end
        lead_chunks = data.chunks[:-2]
        data = data.rechunk(tuple(-1 for _ in lead) + ((1,), self.counts))
        fill = _fill_value(data.dtype)
        rows = []
        for iy, row in enumerate(self.blocks):
            blocks = []
            for ix, (valid, k) in enumerate(row):
                shape = lead + valid.shape
                if k is None:
                    blocks.append(da.full(shape, fill, dtype=data.dtype, chunks=shape))
                else:
                    block = data.blocks[(0,) * len(lead) + (0, k)]
                    blocks.append(block.map_blocks(_put, valid=valid, fill=fill,
                                                   dtype=data.dtype, chunks=shape))
            rows.append(blocks)
        full = da.block(rows)
        if lead:
            full = full.rechunk(lead_chunks + full.chunks[-2:])
        out = xr.DataArray(full, dims=var.dims, coords=self._other_coords(var),
                           attrs=var.attrs, name=var.name)
        if coords:
            out = out.assign_coords(dict(coords))
        return out.transpose(*dims)
