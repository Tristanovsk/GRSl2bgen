"""Processing on the valid pixels only (GRSl2bgen.compact)."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from GRSl2bgen import Process
from GRSl2bgen.compact import CompactGrid


def _image(shape=(7, 9), nwl=3, seed=0):
    rng = np.random.default_rng(seed)
    ny, nx = shape
    return xr.Dataset(
        {'Rrs': (('wl', 'y', 'x'), rng.random((nwl, ny, nx))),
         'sza': (('y', 'x'), rng.random((ny, nx))),
         'flag': (('y', 'x'), rng.random((ny, nx)) > 0.5),
         'scalar': ((), 1.5)},
        coords={'wl': [443., 560., 665.][:nwl], 'y': np.arange(ny) * 20., 'x': np.arange(nx) * 20.,
                'time': np.datetime64('2021-02-11')},
        attrs={'title': 'test'})


@pytest.mark.parametrize('chunks', [None, ((4, 3), (5, 4)), ((7,), (9,))])
@pytest.mark.parametrize('lazy', [False, True])
def test_pack_unpack_roundtrip(chunks, lazy):
    ds = _image()
    valid = np.random.default_rng(1).random((7, 9)) > 0.6
    valid[:4, :5] = False  # one block without any valid pixel
    if lazy:
        ds = ds.chunk({'y': 4, 'x': 5})
    grid = CompactGrid(valid, chunks)
    assert grid.size == valid.sum()

    packed = grid.pack(ds)
    assert dict(packed.sizes) == {'wl': 3, 'y': 1, 'x': int(valid.sum())}
    assert packed['scalar'].item() == 1.5
    assert packed.attrs == ds.attrs and 'time' in packed.coords
    assert isinstance(packed['Rrs'].data, da.Array)

    coords = {'y': ds.y, 'x': ds.x}
    out = grid.unpack(packed, coords).compute()
    expected = ds.compute()
    np.testing.assert_array_equal(out['Rrs'].values, np.where(valid, expected['Rrs'].values, np.nan))
    np.testing.assert_array_equal(out['sza'].values, np.where(valid, expected['sza'].values, np.nan))
    np.testing.assert_array_equal(out['flag'].values, valid & expected['flag'].values)
    assert out['Rrs'].dims == ('wl', 'y', 'x')
    np.testing.assert_array_equal(out.x, ds.x)
    np.testing.assert_array_equal(out.y, ds.y)


def test_pixel_order_follows_blocks():
    """Pixels are packed block by block, row-major inside each block."""
    valid = np.ones((2, 4), bool)
    values = xr.DataArray(np.arange(8.).reshape(2, 4), dims=('y', 'x'))
    grid = CompactGrid(valid, ((2,), (2, 2)))
    assert grid.counts == (4, 4)
    np.testing.assert_array_equal(grid.pack(values).values, [[0., 1., 4., 5., 2., 3., 6., 7.]])


def test_transposed_variable():
    ds = _image()
    valid = np.random.default_rng(2).random((7, 9)) > 0.5
    grid = CompactGrid(valid, ((4, 3), (5, 4)))
    rrs = ds['Rrs'].transpose('y', 'x', 'wl')
    packed = grid.pack(rrs)
    assert packed.dims == ('y', 'x', 'wl')
    out = grid.unpack(packed)
    assert out.dims == ('y', 'x', 'wl')
    np.testing.assert_array_equal(out.values, np.where(valid[..., None], rrs.values, np.nan))


def test_no_valid_pixel_keeps_one():
    grid = CompactGrid(np.zeros((5, 5), bool))
    assert grid.size == 1


def test_shape_mismatch():
    grid = CompactGrid(np.ones((3, 3), bool))
    with pytest.raises(ValueError):
        grid.pack(xr.DataArray(np.zeros((4, 3)), dims=('y', 'x')))
    with pytest.raises(ValueError):
        CompactGrid(np.ones((3, 3), bool), ((2,), (3,)))


def _run(l2a, tmp_path, name, **kwargs):
    proc = Process(l2a, l2b_path=str(tmp_path / name), **kwargs)
    proc.run()
    return xr.open_dataset(tmp_path / name, mask_and_scale=False)


@pytest.mark.parametrize('input_fixture', ['l2a_path', 'l2a_netcdf_dir'])
def test_compact_product_is_identical(request, tmp_path, input_fixture):
    """Same L2B product, values, attributes and encoding, with and without the compact grid."""
    l2a = request.getfixturevalue(input_fixture)
    with _run(l2a, tmp_path, 'full.nc', compact=False) as full, \
            _run(l2a, tmp_path, 'compact.nc', compact=True) as compact:
        full.attrs.pop('processing_time')
        compact.attrs.pop('processing_time')
        assert full.identical(compact)
        for v in full.variables:
            assert full[v].encoding.get('chunksizes') == compact[v].encoding.get('chunksizes')


def test_compact_product_without_water(tmp_path, l2a_path):
    """An image without any water pixel gives the same (empty) product."""
    from GRSl2bgen.product import Product

    prod = Product(l2a_path)
    raster = prod.raster.load()
    prod.close()
    raster['mask'] = xr.ones_like(raster['mask'])
    with _run(raster, tmp_path, 'full.nc', compact=False) as full, \
            _run(raster, tmp_path, 'compact.nc', compact=True) as compact:
        full.attrs.pop('processing_time')
        compact.attrs.pop('processing_time')
        assert full.identical(compact)
