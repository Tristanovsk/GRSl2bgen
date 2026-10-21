"""End-to-end tests on the L2A test product of `tests/data` (see conftest.py)."""

import numpy as np
import pytest
import xarray as xr

from GRSl2bgen.product import Product

EXPECTED_VARIABLES = (
    'owt_dist_Spyrakos2018', 'owt_index_Spyrakos2018',
    'owt_dist_Cordeiro2022SPM', 'owt_index_Cordeiro2022SPM',
    'Chla_OWTblend_Ta2025', 'SPM_OWTblend_Ta2025', 'TURB_dogliotti',
    'adg443_QAAv6Chl', 'Kd_par', 'Zsd_Lee15',
    'Kd490_KD2', 'Kd490_Lee13', 'Kd_par_Mo07', 'Zeu_Lee05', 'Kd_par_Lee05',
    'flags', 'flags_l2b', 'mask',
)


@pytest.fixture(scope='module')
def l2b(l2b_nc):
    with xr.open_dataset(l2b_nc) as ds:
        yield ds.load()


@pytest.fixture(scope='module')
def l2a(l2a_path):
    prod = Product(l2a_path)
    yield prod.raster.load()
    prod.close()


@pytest.mark.parametrize('level, size', [(0, 128), (1, 64)])
def test_product_reads_pyramid_level(l2a_path, level, size):
    prod = Product(l2a_path, level=level)
    assert dict(prod.raster.sizes) == {'y': size, 'x': size, 'wl': 11}
    assert prod.raster['Rrs'].dims == ('wl', 'y', 'x')
    assert prod.ancillary is None
    prod.close()


def test_product_rejects_missing_level(l2a_path):
    with pytest.raises(ValueError):
        Product(l2a_path, level=2)


def test_l2b_variables_and_grid(l2b, l2a):
    missing = [v for v in EXPECTED_VARIABLES if v not in l2b]
    assert not missing
    np.testing.assert_array_equal(l2b.x, l2a.x)
    np.testing.assert_array_equal(l2b.y, l2a.y)


def test_l2b_scalar_coordinates(l2b):
    # only the acquisition time and the grid mapping, no leftover of the
    # input (band) or of the wavelength selections (wl, central_wavelength)
    scalar_coords = {c for c in l2b.coords if c not in l2b.dims}
    assert scalar_coords <= {'time', 'spatial_ref'}
    for name in ('band', 'wl', 'central_wavelength',
                 'forecast_period', 'forecast_reference_time'):
        assert name not in l2b.variables, name


def test_l2b_coordinates_attribute(l2b_nc):
    with xr.open_dataset(l2b_nc, decode_cf=False) as ds:
        for var in ds.data_vars:
            coords = set(ds[var].attrs.get('coordinates', '').split())
            assert coords <= {'time', 'spatial_ref'}, (var, coords)


def test_l2b_only_water_pixels(l2b, l2a):
    land = (l2a['mask'] != 0).values
    water = ~land
    for var in ('Chla_OWTblend_Ta2025', 'SPM_OWTblend_Ta2025', 'Kd_par'):
        values = l2b[var].values
        assert np.all(np.isnan(values[land])), var
        # a few water pixels may be invalid for a given algorithm
        assert np.isfinite(values[water]).mean() > 0.95, var


def test_l2b_flags(l2b):
    flags = l2b['flags_l2b']
    assert flags.dtype == np.uint32
    assert len(flags.attrs['flag_names']) == 32
    assert len(flags.attrs['flag_descriptions']) == 32


# validity filters applied by the retrieval modules (docs/source/output_product.rst)
VALID_RANGES = {
    'Chla_OC2nasa': [0, 1200], 'Chla_M09B': [0, 1200], 'Chla_NIRB': [0, 1200],
    'Chla_OWTblend_Ta2025': [0, 1200],
    'SPM_obs2co': [0, 2000], 'TURB_dogliotti': [0, 2000], 'SPM_nechad': [0, 2000],
    'SPM_OWTblend_Ta2025': [0, 2000],
    'acdom_B15': [0, 30], 'adg443_QAAv6': [0, 30], 'adg443_QAAv6Chl': [0, 30],
    'Kd_par': [0, 3000], 'Zsd_Lee15': [0, 100],
    'Kd490_KD2': [0, 50], 'Kd490_Lee13': [0, 50], 'Kd_par_Mo07': [0, 50],
    'Zeu_Lee05': [0, 200], 'Kd_par_Lee05': [0, 50],
}


def test_l2b_range_attributes(l2b):
    """The ``range`` attribute gives the bounds of the validity filter applied."""
    for var, expected in VALID_RANGES.items():
        assert list(l2b[var].attrs['range']) == expected, var


def test_l2b_values_within_range_attribute(l2b):
    """Every valid value lies within the ``range`` attribute of its variable."""
    with_range = [v for v in l2b.data_vars if 'range' in l2b[v].attrs]
    for var in with_range:
        lo, hi = l2b[var].attrs['range']
        values = l2b[var].values
        valid = values[np.isfinite(values)]
        # one int16 packing step of tolerance
        tol = float(l2b[var].encoding.get('scale_factor', 0))
        assert valid.min() >= lo - tol and valid.max() <= hi + tol, var


def _assert_class_ids(ds):
    """OWT class IDs must read back as exact integers (1-based), NaN outside water."""
    for var in [v for v in ds.data_vars if v.startswith('owt_index')]:
        values = ds[var].values
        valid = values[np.isfinite(values)]
        assert valid.size, var
        np.testing.assert_array_equal(valid, np.round(valid), err_msg=var)
        assert valid.min() >= 1, var


def test_l2b_owt_index_are_class_ids(l2b):
    _assert_class_ids(l2b)


@pytest.mark.parametrize('level', [0, 1, 2])
def test_l2b_pyramid_owt_index_are_class_ids(l2b_pyramid, level):
    with xr.open_zarr(l2b_pyramid, group=str(level)) as ds:
        _assert_class_ids(ds.load())
