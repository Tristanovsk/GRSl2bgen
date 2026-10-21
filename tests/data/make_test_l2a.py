"""Crop a GRS L2A Zarr pyramid into the small test product of `tests/data`.

The crop keeps levels 0 and 1 of the pyramid (same area), the variables,
attributes and int16 packing of the source, so that the test product is read
by `GRSl2bgen.product.Product` exactly as a full scene. The ancillary group is
not copied (not used by GRSl2bgen).

Default window: 128 x 128 pixels (2.56 km) over the Camargue lagoons (tile
31TFJ, 2021-02-11), mixing clear to very turbid water, land and shoreline.

Usage::

    python tests/data/make_test_l2a.py <L2A product dir or .zarr> [y0 x0 size]
"""

import os
import sys

import xarray as xr

from GRSl2bgen.product import Product

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'S2A_MSIL2AGRS_20210211_T31TFJ_camargue_subset.zarr')
NLEVELS = 2
# packing and compression of the source are kept, not its chunks and shards
KEPT_ENCODING = ('dtype', 'scale_factor', 'add_offset', '_FillValue', 'compressors')


def _geotransform(ds):
    """GDAL GeoTransform string of the (cropped) grid of `ds`."""
    dx = float(ds.x[1] - ds.x[0])
    dy = float(ds.y[1] - ds.y[0])
    x0 = float(ds.x[0]) - dx / 2
    y0 = float(ds.y[0]) - dy / 2
    return f'{x0} {dx} 0.0 {y0} 0.0 {dy}'


def crop_level(src, level, y0, x0, size):
    """Window of pyramid `level` covering the level-0 window (y0, x0, size)."""
    scale = 2 ** level
    raster = Product(src, level=level).raster
    sub = raster.isel(y=slice(y0 // scale, (y0 + size) // scale),
                      x=slice(x0 // scale, (x0 + size) // scale)).load()
    sub['spatial_ref'].attrs['GeoTransform'] = _geotransform(sub)
    for var in sub.variables.values():
        var.encoding = {k: v for k, v in var.encoding.items() if k in KEPT_ENCODING}
    return sub


def main(src, y0=4128, x0=1056, size=128):
    if size % 2 ** (NLEVELS - 1) or y0 % 2 ** (NLEVELS - 1) or x0 % 2 ** (NLEVELS - 1):
        raise ValueError(f'y0, x0 and size must be multiples of {2 ** (NLEVELS - 1)}')
    store = src
    if not src.rstrip('/').endswith('.zarr'):
        store = os.path.join(src, os.path.basename(src.rstrip('/')) + '.zarr')
    root = xr.open_zarr(store)
    attrs = dict(root.attrs)
    multiscales = [dict(attrs['multiscales'][0])]
    multiscales[0]['layout'] = [{'group': str(k), 'scale': 2 ** k,
                                 'shape': [size // 2 ** k, size // 2 ** k]}
                                for k in range(NLEVELS)]
    attrs['multiscales'] = multiscales
    attrs['test_subset'] = (f'crop of {os.path.basename(store)}, level-0 window '
                            f'y=[{y0}, {y0 + size}), x=[{x0}, {x0 + size}), '
                            f'made with tests/data/make_test_l2a.py')
    root.close()

    xr.Dataset(attrs=attrs).to_zarr(OUTPUT, mode='w', zarr_format=3, consolidated=True)
    for level in range(NLEVELS):
        crop_level(src, level, y0, x0, size).to_zarr(OUTPUT, group=str(level), mode='a',
                                                     zarr_format=3, consolidated=True)
    print(f'written {OUTPUT}')


if __name__ == '__main__':
    if len(sys.argv) not in (2, 5):
        sys.exit(__doc__)
    main(sys.argv[1], *map(int, sys.argv[2:]))
