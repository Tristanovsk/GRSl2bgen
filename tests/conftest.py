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
