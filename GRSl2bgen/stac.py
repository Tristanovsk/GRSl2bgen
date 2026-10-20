"""STAC Item of an L2B product.

The Item (``<name>.json``) is written next to the product and its log, with
relative asset links, so that the product folder can be moved or uploaded as a
whole (e.g. to an S3 bucket). It is written last, once the product and its log are
complete: an existing Item marks a finished product.

Extensions: projection (grid of the product), processing (software versions,
lineage), file (size and checksum of the assets), eo (cloud cover of the L1C) and
grid (MGRS tile).
"""

import datetime
import hashlib
import json
import os

import numpy as np
import pystac
import xarray as xr
from pyproj import Transformer

from . import __package__, __version__

STAC_EXTENSIONS = [
    'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
    'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
    'https://stac-extensions.github.io/file/v2.1.0/schema.json',
    'https://stac-extensions.github.io/eo/v1.1.0/schema.json',
    'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
]

MEDIA_TYPES = {'.nc': 'application/x-netcdf', '.zarr': 'application/vnd+zarr',
               '.log': 'text/plain'}


def item_path(ofile):
    """Path of the STAC Item of the product `ofile`: ``<name>.json`` next to it."""
    return os.path.splitext(str(ofile).rstrip('/\\'))[0] + '.json'


def _file_info(path):
    """``file:size`` and ``file:checksum`` (sha2-256 multihash) of a file."""
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            sha.update(block)
    return {'file:size': os.path.getsize(path), 'file:checksum': '1220' + sha.hexdigest()}


def _open(ofile):
    """Open the product, level 0 for a Zarr pyramid (attributes of the root group)."""
    if not str(ofile).rstrip('/\\').endswith('.zarr'):
        return xr.open_dataset(ofile, decode_coords='all'), None
    root = xr.open_zarr(ofile, decode_coords='all')
    if root.data_vars:
        return root, None
    return xr.open_zarr(ofile, group='0', decode_coords='all'), root


def _footprint(ds, npts=2):
    """Polygon (lon/lat) of the product grid edges, `npts` points per side.

    Two points (the corners) are enough for a UTM tile: a 110 km side is straight
    to a few metres in longitude/latitude. Counter-clockwise ring (RFC 7946).
    """
    xmin, ymin, xmax, ymax = ds.rio.bounds()
    t = np.linspace(0, 1, npts)[:-1]
    xs = np.concatenate([xmin + (xmax - xmin) * t, np.full_like(t, xmax),
                         xmax - (xmax - xmin) * t, np.full_like(t, xmin)])
    ys = np.concatenate([np.full_like(t, ymin), ymin + (ymax - ymin) * t,
                         np.full_like(t, ymax), ymax - (ymax - ymin) * t])
    lon, lat = Transformer.from_crs(ds.rio.crs, 'EPSG:4326', always_xy=True).transform(xs, ys)
    ring = [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon, lat)]
    ring.append(ring[0])
    bbox = [min(lon), min(lat), max(lon), max(lat)]
    return {'type': 'Polygon', 'coordinates': [ring]}, [round(float(v), 7) for v in bbox]


def _datetime(value):
    return datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def make_stac_item(ofile, l2a=None, log_file=None):
    """Build the STAC Item of the L2B product `ofile`.

    Parameters
    ----------
    ofile : str
        L2B product (NetCDF file or Zarr store).
    l2a : str, optional
        L2A input: ``derived_from`` link and lineage.
    log_file : str, optional
        Log of the run, added as a ``metadata`` asset.

    Returns
    -------
    pystac.Item
        Item with absolute asset links (made relative by `write_stac_item`).
    """
    ofile = os.path.abspath(str(ofile).rstrip('/\\'))
    name, ext = os.path.splitext(os.path.basename(ofile))
    ds, root = _open(ofile)
    attrs = dict(root.attrs if root is not None else ds.attrs)
    geometry, bbox = _footprint(ds)

    transform = ds.rio.transform()
    properties = {
        'constellation': 'sentinel-2' if attrs.get('constellation') == 'Sentinel-2'
        else str(attrs.get('constellation', '')).lower() or None,
        'platform': str(attrs.get('DATATAKE_1_SPACECRAFT_NAME', attrs.get('satellite', ''))).lower() or None,
        'instruments': [str(attrs['instrument']).lower()] if 'instrument' in attrs else None,
        'proj:code': ds.rio.crs.to_string(),
        'proj:shape': [int(ds.rio.height), int(ds.rio.width)],
        'proj:transform': [transform.a, transform.b, transform.c,
                           transform.d, transform.e, transform.f],
        'proj:bbox': list(ds.rio.bounds()),
        'processing:level': 'L2B',
        'processing:software': {__package__: __version__},
        # processing_time is written in local time without offset
        'processing:datetime': _datetime(attrs['processing_time']).astimezone(
            datetime.timezone.utc).isoformat() if 'processing_time' in attrs else None,
        'processing:lineage': (f'{__package__} {__version__} applied to the GRS L2A product '
                               f'{os.path.basename(str(l2a).rstrip("/"))}') if l2a else None,
        'grid:code': f"MGRS-{attrs['tile']}" if 'tile' in attrs else None,
    }
    if 'version' in attrs:
        properties['processing:software']['GRS'] = str(attrs['version'])
    if 'CLOUD_COVERAGE_ASSESSMENT' in attrs:
        properties['eo:cloud_cover'] = float(attrs['CLOUD_COVERAGE_ASSESSMENT'])
    properties = {k: v for k, v in properties.items() if v is not None}

    item = pystac.Item(id=name, geometry=geometry, bbox=bbox,
                       datetime=_datetime(attrs['acquisition_date']),
                       properties=properties, stac_extensions=list(STAC_EXTENSIONS))
    ds.close()
    if root is not None:
        root.close()

    # no size and checksum for a Zarr store (a directory)
    item.add_asset('data', pystac.Asset(href=ofile, media_type=MEDIA_TYPES.get(ext),
                                        title=f'{__package__} L2B water quality parameters',
                                        roles=['data'],
                                        extra_fields=_file_info(ofile) if os.path.isfile(ofile) else None))
    if log_file and os.path.isfile(log_file):
        item.add_asset('log', pystac.Asset(href=os.path.abspath(log_file), media_type='text/plain',
                                           title=f'{__package__} processing log', roles=['metadata'],
                                           extra_fields=_file_info(log_file)))
    if l2a:
        item.add_link(pystac.Link('derived_from', os.path.abspath(str(l2a).rstrip('/')),
                                  title='GRS L2A product'))
    return item


def write_stac_item(ofile, l2a=None, log_file=None, item_file=None):
    """Write the STAC Item of `ofile` (see `make_stac_item`), atomically.

    Asset links are relative to the Item; the ``derived_from`` link stays as given.

    Parameters
    ----------
    ofile, l2a, log_file : str
        See `make_stac_item`.
    item_file : str, optional
        Item path (default: `item_path` of `ofile`).

    Returns
    -------
    str
        Path of the Item.
    """
    item_file = os.path.abspath(item_file or item_path(ofile))
    item = make_stac_item(ofile, l2a=l2a, log_file=log_file)
    item_dir = os.path.dirname(item_file)
    for asset in item.assets.values():
        asset.href = './' + os.path.relpath(asset.href, item_dir)
    tmp = os.path.join(item_dir, f'.{os.path.basename(item_file)}.part{os.getpid()}')
    with open(tmp, 'w') as f:
        json.dump(item.to_dict(include_self_link=False, transform_hrefs=False), f, indent=2)
    os.replace(tmp, item_file)
    return item_file
