"""Shared configuration and helpers of the Tarasenko et al. notebooks (OWT update, 30TWT).

Imported by every notebook of this folder (``from ta_common import *``), so that the paths, the band sets, the
bloom criterion and the spectral-angle classification are defined once.

Layout
------
notebook/tarasenko/
    ta_common.py              this module
    00_..07_*.ipynb           the notebooks (run in order)
    data/                     OWT libraries written by 01-03 (NetCDF), small tables
    cache/<tile>/             per-image reductions of the L2A products (04), sensitivity results (05)
    figures/                  figures of the notebooks; 07 copies the article ones to ARTICLE_FIG

Data
----
GRS L2A products (Zarr pyramids) of the bucket ``s3://grstbx-s3/sentinel-2/l2a/<tile>/YYYY/MM/DD/<name>/<name>.zarr``,
read from the local copy ``/data/satellite/s3/grstbx-s3`` when it exists, otherwise through s3fs (AWS profile
``grstbx-local``) and the STAC catalogue ``s3://grstbx-s3/stac/catalog.json``.
"""

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

# ---------------------------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
FIG = HERE / 'figures'
TILE = os.environ.get('TA_TILE', '30TWT')
CACHE = HERE / 'cache' / TILE
for _d in (DATA, FIG, CACHE / 'img'):
    _d.mkdir(parents=True, exist_ok=True)

BUCKET = 'grstbx-s3'
AWS_PROFILE = os.environ.get('AWS_PROFILE', 'grstbx-local')
STAC_CATALOG = f's3://{BUCKET}/stac/catalog.json'
STAC_COLLECTION = 'grs-l2a'
LOCAL_BUCKET = Path('/data/satellite/s3') / BUCKET / 'sentinel-2'     # local copy of s3://grstbx-s3/sentinel-2
GRS_VERSION = '3.1.1'

# manuscript (LaTeX project) and the analysis of the current version (L2B based, kept for comparison)
ARTICLE = Path('/home/harmel/Dropbox/Dropbox/work/git/satellite_app/grstbx/notebook/case_study/tarasenko')
ARTICLE_FIG = ARTICLE / 'figures'
ARTICLE_PDF_V1 = Path('/home/harmel/Dropbox/Dropbox/work/ARTICLES/tarasenko/article_tarasenko_etal_obt.pdf')
BATHY_FILE = ARTICLE / 'analysis' / 'aux' / 'emodnet_dtm_30TWT.tif'   # EMODnet DTM 2024 (elevation / LAT)

# OWT libraries
import GRSl2bgen                                                        # noqa: E402

GRSL2BGEN_DATA = Path(GRSl2bgen.__file__).parent / 'data'
TA25_V1_FILE = GRSL2BGEN_DATA / 'owt_tarasenko2025.nc'                  # current database of GRSl2bgen
TA25_V2_FILE = DATA / 'owt_tarasenko2025_v2.nc'                         # 01
HABLAB_OWT_FILE = DATA / 'owt_hablab_invRrs_s2.nc'                      # 02 (S2 bands)
HABLAB_OWT_HYPER_FILE = DATA / 'owt_hablab_invRrs_hyper.nc'             # 02 (hyperspectral)
WT_OWT_FILE = DATA / 'owt_water_types_invRrs_s2.nc'                     # 03 (S2 bands)
WT_OWT_HYPER_FILE = DATA / 'owt_water_types_invRrs_hyper.nc'            # 03 (hyperspectral)
CORDEIRO_FILE = DATA / 'owt_cordeiro2022_spm.nc'    # removed from GRSl2bgen (replaced by invRrsWT), kept for 08

HABLAB_SIM = Path('/DATA/git/laboratory/hablab_tbx/notebook/invRrs/data/hablab_bloom_Rrs_simulations.nc')

# ---------------------------------------------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------------------------------------------
BANDS = [443, 490, 560, 665, 705, 740, 783, 842, 865]   # MSI bands of the L2A used here (B1-B8A)
BANDS_TA25 = [443, 492, 560, 665, 704, 740, 783, 834, 865]  # wavelengths of owt_tarasenko2025.nc (as v1)
BANDS_GRSL2BGEN = BANDS[:7]                             # wl_range 350-800 nm of GRSl2bgen.OWT
PIXEL_KM2 = 20 * 20 / 1e6

# spectral ranges of the sensitivity analysis (05): name -> MSI bands
SPECTRAL_RANGES = {
    'B1-B7 443-783 (GRSl2bgen)': [443, 490, 560, 665, 705, 740, 783],
    'B1-B8A 443-865 (v1, 9 bands)': BANDS,
    'B2-B7 490-783 (no 443)': [490, 560, 665, 705, 740, 783],
    'B1-B5 443-705 (visible + red edge)': [443, 490, 560, 665, 705],
    'B2-B6 490-740': [490, 560, 665, 705, 740],
    'B3-B7 560-783 (red edge)': [560, 665, 705, 740, 783],
}

# ---------------------------------------------------------------------------------------------------------------
# Ta2025 classes: labels and colours of the article
# ---------------------------------------------------------------------------------------------------------------
TA25_SHORT = {'Cyanobacteria': 'Cyano', 'Mesodinium': 'M. rubrum', 'Dinoflagellates2': 'Dino2', 'red': 'N. red',
              'Dinoflagellates1': 'Dino1 (L. polyedra)', 'Lepidodinium': 'L. chlorophorum'}
TA25_BIO = list(range(11, 17))                          # bloom types of Ta2025 (1-based)
HAB_TYPES = [12, 16, 15, 13]                            # M. rubrum, L. chlorophorum, Dino1, Dino2

# Table tab_owt_spectra of the manuscript (x 1e-3 sr-1, wavelengths BANDS_TA25), classes 11-16 of Ta2025
TA25_TABLE = {
    'Cyanobacteria':    [7.347, 12.105, 22.158, 14.372, 45.543, 91.918, 93.941, 92.142, 88.447],
    'Mesodinium':       [1.505, 2.275, 2.388, 2.507, 9.695, 4.162, 3.644, 2.653, 1.172],
    'Dinoflagellates2': [0.637, 1.705, 6.440, 2.731, 10.553, 3.796, 3.704, 2.818, 2.029],
    'red':              [6.847, 8.341, 25.695, 27.815, 27.387, 26.743, 27.227, 27.744, 24.269],
    'Dinoflagellates1': [0.099, 1.106, 4.919, 2.465, 16.576, 8.143, 7.326, 6.064, 3.588],
    'Lepidodinium':     [1.246, 3.101, 12.438, 2.824, 12.382, 2.445, 2.285, 2.124, 1.219],
}

# HABLAB cultures -> Ta2025 bloom type they should be classified as (None: no Ta2025 counterpart)
HABLAB_TO_TA25 = {'MES': 'Mesodinium', 'TEL': 'Mesodinium', 'LEP': 'Lepidodinium', 'LIN': 'Dinoflagellates1',
                  'ALM': 'Dinoflagellates1', 'ALX': 'Dinoflagellates1', 'PRO': 'Dinoflagellates1',
                  'HEC': 'Dinoflagellates2', 'KAR': 'Dinoflagellates2', 'MIC': 'Cyanobacteria',
                  'EMI': None, 'CHA': None, 'HET': None, 'PRY': None}


def ta25_style(owt_names):
    """Labels (1-based dict) and colours of the Ta2025 classes, as in the figures of the article."""
    import matplotlib.pyplot as plt
    label = {i + 1: TA25_SHORT.get(str(o), str(o)) for i, o in enumerate(owt_names)}
    color = dict(zip(range(1, 11), plt.get_cmap('Spectral_r')(np.linspace(0.02, 0.6, 10))[::-1]))
    color.update({11: '#9400d3', 12: 'red', 13: 'green', 14: 'cyan', 15: 'olive', 16: 'yellowgreen'})
    return label, color


# ---------------------------------------------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------------------------------------------
NAME_RE = re.compile(r'(S2[ABC])_MSIL(?:2B|2AGRS|1C)_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\w{5})_(\d{8}T\d{6})(?:_V([\d.]+))?')


def inventory_local(level='l2a', tile=TILE, root=LOCAL_BUCKET):
    """L2A (or L2B) products of the local copy of the bucket: one row per acquisition (last processing)."""
    rows = []
    for p in sorted(glob.glob(str(root / level / tile / '*/*/*/*/*.zarr'))):
        m = NAME_RE.match(Path(p).name)
        if m:
            sat, dt, base, orbit, _, proc, ver = m.groups()
            rows.append(dict(path=p, name=Path(p).name.removesuffix('.zarr'), sat=sat, time=pd.Timestamp(dt),
                             baseline=base, orbit=orbit, processed=pd.Timestamp(proc), version=ver))
    df = pd.DataFrame(rows, columns=['path', 'name', 'sat', 'time', 'baseline', 'orbit', 'processed', 'version'])
    return (df.sort_values(['version', 'processed']).drop_duplicates(['sat', 'time'], keep='last')
            .sort_values('time').reset_index(drop=True))


def inventory_stac(tile=TILE, start='2015-01-01', end='2030-12-31', max_cloud_cover=None, version=GRS_VERSION):
    """L2A products of the STAC catalogue of the bucket (paths are s3:// URLs)."""
    os.environ.setdefault('AWS_PROFILE', AWS_PROFILE)
    from grstbx.app import search_items
    from grstbx import batch
    found = search_items(STAC_CATALOG, collections=STAC_COLLECTION, start=start, end=end,
                         max_cloud_cover=max_cloud_cover, version=version)
    found = found[found.tile == tile]
    return batch.select_versions(found.set_index('time')).reset_index()


def inventory(tile=TILE, start='2015-01-01', end='2030-12-31', source='auto'):
    """L2A products of `tile`: local copy if present (source='local'), else the STAC catalogue ('stac')."""
    if source == 'local' or (source == 'auto' and (LOCAL_BUCKET / 'l2a' / tile).exists()):
        df = inventory_local('l2a', tile)
    else:
        df = inventory_stac(tile, start, end)
    df = df[(df.time >= pd.Timestamp(start)) & (df.time <= pd.Timestamp(end) + pd.Timedelta(days=1))]
    return df.reset_index(drop=True)


def open_l2a(path, level=0):
    """GRS L2A Zarr image (pyramid level `level`), local path or s3:// URL."""
    os.environ.setdefault('AWS_PROFILE', AWS_PROFILE)
    from grstbx.driver import open_zarr_image
    return open_zarr_image(str(path), level=level)


def product_of_day(products, day):
    """Row of `products` (inventory) acquired on `day` ('YYYY-MM-DD'), or None."""
    sel = products[products.time.dt.strftime('%Y-%m-%d') == day]
    return None if sel.empty else sel.iloc[0]


# ---------------------------------------------------------------------------------------------------------------
# OWT libraries
# ---------------------------------------------------------------------------------------------------------------
def load_owt(path):
    """OWT library (DataArray owt x wl) and its label / colour / group coordinates when present."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    var = [v for v in ds.data_vars if ds[v].dims == ('owt', 'wl')][0]
    da = ds[var]
    for c in ('label', 'color', 'group', 'chl', 'species', 'ta25_class'):
        if c in ds and c not in da.coords:
            da = da.assign_coords({c: ds[c]})
    da['owt'] = [str(o) for o in da.owt.values]
    return da


def to_bands(owt, bands=BANDS):
    """Library `owt` (owt x wl) on the MSI bands: linear interpolation in wavelength, as GRSl2bgen.OWT does."""
    return owt.interp(wl=bands)


def write_owt(da, path, title, **attrs):
    """Write an OWT library with the layout read by GRSl2bgen.OWT (xowt) and by `load_owt`."""
    ds = da.to_dataset(name=da.name or 'm_Rrs')
    ds.attrs.update(title=title, Conventions='CF-1.8', **{k: str(v) for k, v in attrs.items()})
    ds.wl.attrs.update(units='nm', long_name='wavelength')
    enc = {v: {'zlib': True} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    return path


# ---------------------------------------------------------------------------------------------------------------
# spectral angle classification
# ---------------------------------------------------------------------------------------------------------------
def unit(a):
    a = np.asarray(a, dtype=np.float64)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def angles(a, b):
    """Spectral angles (rad) between the rows of a (n, nb) and b (m, nb): (n, m)."""
    return np.arccos(np.clip(unit(a) @ unit(b).T, -1, 1))


def sam_topk(rrs, ref, k=2):
    """k best classes of each spectrum: (index (n, k) 1-based, angle (n, k) rad); NaN rows of rrs give NaN."""
    rrs = np.asarray(rrs, dtype=np.float64)
    ok = np.all(np.isfinite(rrs), axis=1) & np.any(rrs != 0, axis=1)
    idx = np.full((len(rrs), k), np.nan)
    ang = np.full((len(rrs), k), np.nan)
    if ok.any():
        a = angles(rrs[ok], ref)
        o = np.argsort(a, axis=1)[:, :k]
        idx[ok] = o + 1
        ang[ok] = np.take_along_axis(a, o, axis=1)
    return idx, ang


def classify(rrs, ref, bands_rrs, bands, k=2):
    """SAM of spectra `rrs` (n, len(bands_rrs)) against `ref` (owt x wl DataArray) over `bands`.

    Returns a DataFrame: cls1, ang1 (deg), cls2, ang2 (deg), gap = ang2 - ang1 (deg).
    """
    j = [list(bands_rrs).index(b) for b in bands]
    r = to_bands(ref, bands).values
    idx, ang = sam_topk(np.asarray(rrs)[:, j], r, k=k)
    out = pd.DataFrame({'cls1': idx[:, 0], 'ang1': np.degrees(ang[:, 0])})
    if k > 1:
        out['cls2'] = idx[:, 1]
        out['ang2'] = np.degrees(ang[:, 1])
        out['gap'] = out.ang2 - out.ang1
    return out


def classify_two_step(rrs, bands_rrs, ref1, bands1, ref2, bands2, trigger, map2=None):
    """Two-step SAM: a first classification on the general library, a second one on a bloom library.

    Step 1: SAM against `ref1` over `bands1` (e.g. the 16 Ta2025 classes, 443-783 nm).
    Step 2: where `trigger(step1, rrs)` is True (e.g. a HAB type or a massive bloom by the superbloom criterion),
    a second SAM against `ref2` only (e.g. the 6 bloom types, or the HABLAB species x Chl-a) over `bands2`
    (e.g. a red-edge range). `map2` maps the 1-based classes of ref2 to classes of ref1 (labels comparable).

    Returns the DataFrame of step 1 with the columns of step 2 (cls2nd, ang2nd, gap2nd) and `cls` (final).
    """
    s1 = classify(rrs, ref1, bands_rrs, bands1)
    trig = np.asarray(trigger(s1, rrs), bool)
    s1['second'] = trig
    s1['cls2nd'] = np.nan
    s1['ang2nd'] = np.nan
    s1['gap2nd'] = np.nan
    if trig.any():
        s2 = classify(np.asarray(rrs)[trig], ref2, bands_rrs, bands2)
        s1.loc[trig, 'cls2nd'] = s2.cls1.values
        s1.loc[trig, 'ang2nd'] = s2.ang1.values
        s1.loc[trig, 'gap2nd'] = s2.gap.values
    final = s1.cls1.copy()
    if trig.any():
        c2 = s1.loc[trig, 'cls2nd']
        final.loc[trig] = c2.map(map2) if map2 is not None else c2
    s1['cls'] = final
    return s1


# ---------------------------------------------------------------------------------------------------------------
# bloom criterion, products of the L2A (same formulas as GRSl2bgen), spatial filter
# ---------------------------------------------------------------------------------------------------------------
BLOOM = dict(r705_min=0.005, ratio_min=1.8, r665_min=5e-4, min_pixels=100)     # Eq. superbloom of the article
CAND = dict(r705_min=0.003, ratio_min=1.2)                                       # loose criterion of the cache
PATCH = dict(min_bloom_px=100, min_class_px=9)


def superbloom(r705, r665, p=BLOOM):
    r705, r665 = np.asarray(r705), np.asarray(r665)
    ok = (r705 > p['r705_min']) & (r705 > p['ratio_min'] * r665)
    if p.get('r665_min') is not None:
        ok &= r665 > p['r665_min']
    return ok


def chla_nirb(r705, r443):
    """Chla_NIRB of GRSl2bgen: 11.2 (Rrs705/Rrs443)^1.7, bounded to 0-1200 mg m-3."""
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 11.2 * (np.asarray(r705) / np.asarray(r443)) ** 1.7
    return np.where((np.asarray(r705) >= 0) & (np.asarray(r443) > 0) & (c <= 1200), c, np.nan)


def spm_nechad(r665, A=342.1, C=0.1956):
    """SPM_nechad of GRSl2bgen (665 nm), g m-3."""
    rho = np.pi * np.asarray(r665)
    with np.errstate(divide='ignore', invalid='ignore'):
        return A * rho / (1 - rho / C)


def groups(iy, ix):
    """8-connected groups of sparse pixels: (label of each pixel, size of its group)."""
    iy, ix = np.asarray(iy, int), np.asarray(ix, int)
    if not len(iy):
        return np.zeros(0, int), np.zeros(0, int)
    y0, x0 = iy.min(), ix.min()
    g = np.zeros((iy.max() - y0 + 1, ix.max() - x0 + 1), bool)
    g[iy - y0, ix - x0] = True
    lab, _ = ndimage.label(g, structure=np.ones((3, 3), bool))
    lab = lab[iy - y0, ix - x0]
    return lab, np.bincount(lab)[lab]


def spatial_filter(px, cls_col='cls', p=PATCH):
    """Bloom patches and identified pixels of a table of superbloom pixels (columns name, iy, ix, `cls_col`).

    Adds patch, patch_px, class_px, in_patch (patch >= min_bloom_px) and identified (class group >= min_class_px).
    """
    n = len(px)
    patch, patch_px, class_px = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    iy, ix, cls = px.iy.values.astype(int), px.ix.values.astype(int), px[cls_col].values
    for _, pos in px.groupby('name', sort=False, observed=True).indices.items():
        patch[pos], patch_px[pos] = groups(iy[pos], ix[pos])
        for k in pd.unique(cls[pos]):
            sel = pos[cls[pos] == k]
            class_px[sel] = groups(iy[sel], ix[sel])[1]
    out = px.assign(patch=patch, patch_px=patch_px, class_px=class_px)
    out['in_patch'] = out.patch_px >= p['min_bloom_px']
    out['identified'] = out.in_patch & (out.class_px >= p['min_class_px'])
    return out


def events(px, label, cls_col='cls'):
    """One row per bloom patch: time, surface, dominant class of the identified pixels and its share."""
    b = px[px.in_patch]
    rows = []
    for (name, pid), g in b.groupby(['name', 'patch'], observed=True):
        cnt = g.loc[g.identified, cls_col].value_counts()
        rows.append(dict(name=name, time=g.time.iloc[0] if 'time' in g else pd.NaT, km2=len(g) * PIXEL_KM2,
                         iy=g.iy.mean(), ix=g.ix.mean(), identified=g.identified.mean(),
                         dominant=label.get(cnt.index[0], str(cnt.index[0])) if len(cnt) else 'unidentified',
                         dominant_share=cnt.iloc[0] / max(cnt.sum(), 1) if len(cnt) else np.nan,
                         ang_med=g.ang1.median() if 'ang1' in g else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# shallow-water mask (EMODnet DTM, LAT)
# ---------------------------------------------------------------------------------------------------------------
def depth_grid(template, path=BATHY_FILE):
    """Depth below LAT (m, > 0 at sea) on the grid of `template` (DataArray y, x of a product), bilinear."""
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling
    z = xr.open_dataarray(path, engine='rasterio').squeeze(drop=True)
    z = z if z.rio.crs else z.rio.write_crs(4326)
    if not template.rio.crs:
        template = template.rio.write_crs(int(template.attrs.get('epsg', 32630)) if 'epsg' in template.attrs else 32630)
    return -z.rio.reproject_match(template, resampling=Resampling.bilinear).values.astype(np.float32)


def shallow_mask(depth, min_depth_m=0.0, buffer_px=0):
    """True where the pixel is shallower than `min_depth_m` below LAT (NaN depth: masked), dilated by `buffer_px`."""
    m = ~(np.nan_to_num(depth, nan=-1e3) >= min_depth_m)
    if buffer_px:
        m = ndimage.binary_dilation(m, iterations=buffer_px)
    return m


# coastal sea (100 m cells) from the water frequency, as in the current version of the article
MARINE = dict(min_water_frac=0.5, window_cells=51, estuary_cut=(-2.03, 47.12),
              exclude_boxes={'Guerande salt marshes and Traict du Croisic': (-2.52, 47.285, -2.42, 47.35)})


def valid_maps(names, cache=None):
    """Valid 20 m pixels per 100 m cell summed over the images, and number of images where the cell is mostly valid."""
    cache = cache or CACHE
    total, n_ok = None, None
    for n in names:
        with np.load(cache / 'img' / f'{n}.npz', allow_pickle=True) as f:
            v = f['valid_lr'].astype(np.int32)
        total = v if total is None else total + v
        n_ok = (v >= 13).astype(np.int32) if n_ok is None else n_ok + (v >= 13)
    return total, n_ok


def marine_mask(names, ref, p=MARINE, epsg=32630):
    """Coastal sea (bool, 100 m cells): permanent water connected to the open sea, without the upper estuary.

    `ref` is a product of the tile (for the coordinates of the 20 m grid)."""
    import pyproj
    _, n_ok = valid_maps(names)
    frac = n_ok / np.maximum(ndimage.maximum_filter(n_ok, size=p['window_cells']), 1)
    lab, _ = ndimage.label(frac >= p['min_water_frac'])
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    xs = ref.x.values[2::5][:n_ok.shape[1]]; ys = ref.y.values[2::5][:n_ok.shape[0]]
    lon, lat = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True).transform(*np.meshgrid(xs, ys))
    sea = (lab == sizes.argmax()) & ~((lon > p['estuary_cut'][0]) & (lat > p['estuary_cut'][1]))
    for x0, y0, x1, y1 in p.get('exclude_boxes', {}).values():
        sea &= ~((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
    return sea, frac


def in_cells(mask_lr, iy, ix):
    """Value of a 100 m cell mask at 20 m pixels (iy, ix)."""
    cy = np.minimum(np.asarray(iy, int) // 5, mask_lr.shape[0] - 1)
    cx = np.minimum(np.asarray(ix, int) // 5, mask_lr.shape[1] - 1)
    return mask_lr[cy, cx]


def load_cache(names, what='cand', depth=None, rec=None):
    """Pixels of the 04 cache as one table: R443..R865, iy, ix, name, time, depth, chl_nirb, spm_nechad, superbloom."""
    frames = []
    for n in names:
        with np.load(CACHE / 'img' / f'{n}.npz', allow_pickle=True) as f:
            if f[f'{what}_iy'].size == 0:
                continue
            d = pd.DataFrame(f[f'{what}_rrs'], columns=[f'R{b}' for b in BANDS])
            d['iy'], d['ix'], d['name'] = f[f'{what}_iy'], f[f'{what}_ix'], n
            frames.append(d)
    px = pd.concat(frames, ignore_index=True)
    px['name'] = px.name.astype('category')
    if rec is not None:
        px['time'] = px.name.map(rec.set_index('name').time).astype('datetime64[ns]')
    if depth is not None:
        px['depth'] = depth[px.iy.astype(int), px.ix.astype(int)]
    px['chl_nirb'] = chla_nirb(px.R705, px.R443)
    px['spm_nechad'] = spm_nechad(px.R665)
    px['superbloom'] = superbloom(px.R705, px.R665)
    return px


RCOLS = [f'R{b}' for b in BANDS]


# ---------------------------------------------------------------------------------------------------------------
# invRrs helpers
# ---------------------------------------------------------------------------------------------------------------
def s2_convolution(wl, R, sat='s2a', bands=BANDS):
    """Band-averaged Rrs on the MSI bands B1-B8A (invRrs / Py6S spectral responses); R (..., wl)."""
    from invRrs.spectral_integration import spectral_responses
    srf = spectral_responses(sat, band_idx=list(range(9)))
    R = np.asarray(R)
    flat = R.reshape(-1, R.shape[-1])
    out = np.array([srf.convolution(np.asarray(wl), r) for r in flat])
    return out.reshape(R.shape[:-1] + (len(bands),))


def add_noise(R, rel=0.05, abs_=2e-4, ac_offset=0.0, ac_slope=1.0, bands=BANDS, n=50, seed=0):
    """Noisy copies of spectra R (m, nb): relative + absolute Gaussian noise and an atmospheric-correction-like
    offset ac_offset * (lambda / 865)^-ac_slope drawn per copy in N(0, ac_offset). Returns (m, n, nb)."""
    rng = np.random.default_rng(seed)
    R = np.asarray(R)[:, None, :]
    lam = (np.asarray(bands, float) / 865.) ** (-ac_slope)
    noisy = R * (1 + rel * rng.standard_normal((R.shape[0], n, R.shape[2]))) \
        + abs_ * rng.standard_normal((R.shape[0], n, R.shape[2]))
    if ac_offset:
        noisy = noisy + ac_offset * rng.standard_normal((R.shape[0], n, 1)) * lam
    return noisy


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
    return path
