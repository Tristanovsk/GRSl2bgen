"""Quick look of an L2B product: a PNG of a few parameters, written next to the product.

The image, ``<name>_ql_l2b.png``, shows Chl-a, SPM and Kd(PAR) over the valid water
pixels, on fixed logarithmic colour scales so that the quick looks of different dates
compare. It is read from a coarse level of a Zarr pyramid, or from one pixel out of
``step`` of the other products, so that a full tile is rendered in a few seconds.
`GRSl2bgen.stac.write_stac_item` writes it before the Item, as its ``thumbnail`` asset.
"""

import logging
import math
import os

import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

# records of the package logger (GRSl2bgen.<module>), configured by the application
logger = logging.getLogger(__name__)

# variable, title, colormap, (min, max) of the logarithmic colour scale
PANELS = (
    ('Chla_OC2nasa', 'Chl-a, OC2', 'Greens', (0.1, 100.)),
    ('SPM_nechad', 'SPM, Nechad', 'YlOrBr', (0.1, 100.)),
    ('Kd_par', 'Kd(PAR)', 'Blues', (0.05, 5.)),
)
# pixels that are not valid water (land, clouds, outside the swath)
NO_WATER_COLOR = '#d9d9d9'
# largest side (pixels) of the image read for the quick look
MAX_SIZE = 800


def quicklook_path(ofile):
    """Path of the quick look of the product `ofile`: ``<name>_ql_l2b.png`` next to it."""
    return os.path.splitext(str(ofile).rstrip('/\\'))[0] + '_ql_l2b.png'


def _levels(attrs):
    """Groups of the levels of a multiscale Zarr store, finest first (empty if not a pyramid).

    GRSl2bgen writes ``{'layout': [{'asset': '0'}, ...]}``, GRS ``[{'layout': [{'group': '0'}, ...]}]``.
    """
    multiscales = attrs.get('multiscales')
    if isinstance(multiscales, list):
        multiscales = multiscales[0] if multiscales else None
    layout = (multiscales or {}).get('layout') or []
    return [str(level.get('asset', level.get('group'))) for level in layout]


def open_coarse(ofile, max_size=MAX_SIZE):
    """Open the product `ofile` at a resolution whose largest side is about `max_size` pixels.

    For a Zarr pyramid, the finest level not larger than `max_size` (else the coarsest);
    for the other products, one pixel out of ``step`` in x and y.

    Returns
    -------
    xarray.Dataset
        Lazy dataset, to be closed by the caller.
    """
    path = str(ofile).rstrip('/\\')
    if path.endswith('.zarr'):
        with xr.open_zarr(path) as root:
            groups = [] if root.data_vars else _levels(root.attrs)
        if groups:
            for group in groups:
                ds = xr.open_zarr(path, group=group, decode_coords='all')
                if max(ds.sizes['x'], ds.sizes['y']) <= max_size or group == groups[-1]:
                    return ds
                ds.close()
        ds = xr.open_zarr(path, decode_coords='all')
    else:
        ds = xr.open_dataset(path, decode_coords='all')
    step = math.ceil(max(ds.sizes['x'], ds.sizes['y']) / max_size)
    return ds.isel(x=slice(None, None, step), y=slice(None, None, step)) if step > 1 else ds


def write_quicklook(ofile, path=None, max_size=MAX_SIZE, dpi=150):
    """Write the quick look PNG of the L2B product `ofile`.

    Parameters
    ----------
    ofile : str
        L2B product (NetCDF file or Zarr store, possibly a pyramid).
    path : str, optional
        PNG file (default: `quicklook_path` of `ofile`).
    max_size : int, optional
        Largest side (pixels) of the image read (default `MAX_SIZE`).
    dpi : int, optional
        Resolution of the PNG (default 150).

    Returns
    -------
    str or None
        Path of the PNG; None, with a warning, if the product has none of the
        variables of `PANELS`.
    """
    path = os.path.abspath(path or quicklook_path(ofile))
    ds = open_coarse(ofile, max_size)
    try:
        panels = [panel for panel in PANELS if panel[0] in ds]
        if not panels:
            logger.warning(f'no quick look for {ofile}: none of {[p[0] for p in PANELS]} in the product')
            return None
        water = (ds['mask'] == 0) if 'mask' in ds else True
        # north up whatever the order of the y coordinate
        origin = 'upper' if ds['y'].size < 2 or float(ds['y'][0]) > float(ds['y'][-1]) else 'lower'
        name = os.path.splitext(os.path.basename(str(ofile).rstrip('/\\')))[0]

        fig = Figure(figsize=(4.2 * len(panels), 3.5), dpi=dpi, layout='constrained')
        for i, (var, title, cmap, (vmin, vmax)) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1)
            values = ds[var].where(water).squeeze().values.astype('float32')
            # zero or negative values are below the logarithmic scale: shown at its minimum
            values = np.where(np.isfinite(values), np.clip(values, vmin, None), np.nan)
            cm = colormaps[cmap].copy()
            cm.set_bad(NO_WATER_COLOR)
            image = ax.imshow(values, cmap=cm, norm=LogNorm(vmin, vmax), origin=origin,
                              interpolation='nearest')
            ax.set_axis_off()
            units = ds[var].attrs.get('units')
            fig.colorbar(image, ax=ax, shrink=0.8, extend='both',
                         label=f'{title} ({units})' if units else title)
        fig.suptitle(name, fontsize=9)
        tmp = os.path.join(os.path.dirname(path), f'.{os.path.basename(path)}.part{os.getpid()}')
        fig.savefig(tmp, format='png')
        os.replace(tmp, path)
    finally:
        ds.close()
    return path
