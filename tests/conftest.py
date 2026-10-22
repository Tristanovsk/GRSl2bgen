import os

import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
# made with tests/data/make_test_l2a.py
L2A_SUBSET = os.path.join(DATA_DIR, 'S2A_MSIL2AGRS_20210211_T31TFJ_camargue_subset.zarr')


@pytest.fixture(scope='session')
def l2a_path():
    """Path of the test L2A product (Zarr pyramid, 128 x 128 pixels, 2 levels)."""
    return L2A_SUBSET


@pytest.fixture(scope='session')
def l2a_netcdf_dir(tmp_path_factory, l2a_path):
    """The test L2A as a GRS NetCDF product: folder <name>/ with <name>.nc and <name>_anc.nc."""
    import numpy as np
    import xarray as xr

    name = 'S2A_MSIL2A_20210211T103021_N0500_R108_T31TFJ_20230101T000000_V2.2.1'
    pdir = tmp_path_factory.mktemp('l2a_nc') / name
    pdir.mkdir()
    with xr.open_zarr(l2a_path, group='0', decode_coords='all') as raster:
        raster = raster.load()
    # Zarr encodings (compressors, chunks) are not valid NetCDF ones, and NetCDF has
    # no boolean attribute type
    for obj in [raster, *raster.variables.values()]:
        obj.attrs = {k: int(v) if isinstance(v, (bool, np.bool_)) else v for k, v in obj.attrs.items()}
    for var in raster.variables.values():
        var.encoding = {}
    raster.to_netcdf(pdir / f'{name}.nc')
    # ancillary data on a coarse grid, as in the GRS products
    raster[['aot550']].coarsen(x=16, y=16).mean().to_netcdf(pdir / f'{name}_anc.nc')
    return str(pdir)


@pytest.fixture(scope='session')
def l2b_nc(tmp_path_factory, l2a_path):
    """L2B NetCDF produced once by the full processing chain from the test L2A."""
    from GRSl2bgen import Process

    ofile = str(tmp_path_factory.mktemp('l2b') / 'l2b.nc')
    Process(l2a_path, l2b_path=ofile).run()
    return ofile


@pytest.fixture(scope='session')
def l2b_pyramid(tmp_path_factory, l2a_path):
    """L2B multiscale Zarr store (levels 0 to 2) produced once from the test L2A."""
    from GRSl2bgen import Process

    ofile = str(tmp_path_factory.mktemp('l2b') / 'l2b.zarr')
    proc = Process(l2a_path)
    proc.execute()
    # the default min_size (256) leaves the 128 x 128 test image with level 0 only
    proc.l2b.export_to_zarr_pyramid(ofile, min_size=32)
    proc.l2a.close()
    return ofile
