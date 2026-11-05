"""Atomic export: the output only appears under its final name once complete."""

import os

import pytest

from GRSl2bgen.output import L2bProduct


class _Fake(L2bProduct):
    """L2bProduct whose writers only create a file (no dataset needed)."""

    def __init__(self, fail=False):
        self.fail = fail
        self.written = []

    def export_to_netcdf(self, ofile, persist=False):
        self.written.append(ofile)
        with open(ofile, 'w') as f:
            f.write('partial')
        if self.fail:
            raise RuntimeError('interrupted')
        with open(ofile, 'w') as f:
            f.write('complete')


def test_export_renames_complete_file(tmp_path):
    ofile = tmp_path / 'l2b.nc'
    ofile.write_text('old')
    prod = _Fake()
    prod.export(ofile)
    assert ofile.read_text() == 'complete'
    assert os.path.basename(prod.written[0]) != 'l2b.nc'
    assert os.listdir(tmp_path) == ['l2b.nc']


def test_export_failure_leaves_no_partial_file(tmp_path):
    ofile = tmp_path / 'l2b.nc'
    with pytest.raises(RuntimeError):
        _Fake(fail=True).export(ofile)
    assert os.listdir(tmp_path) == []


def test_netcdf_persist_writes_numpy_same_file(tmp_path, l2a_path, monkeypatch):
    """persist=True writes the in-memory product as numpy arrays, same file as the lazy write."""
    import dask.array as da
    import xarray as xr

    from GRSl2bgen import Process

    written = {}
    to_netcdf = xr.Dataset.to_netcdf

    def spy(ds, path, *args, **kwargs):
        written[os.path.basename(path)] = any(isinstance(v.data, da.Array)
                                              for v in ds.data_vars.values())
        return to_netcdf(ds, path, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, 'to_netcdf', spy)

    proc = Process(l2a_path)
    proc.execute()
    lazy, persisted = tmp_path / 'lazy.nc', tmp_path / 'persisted.nc'
    proc.l2b.export_to_netcdf(str(lazy), persist=False)
    proc.l2b.export_to_netcdf(str(persisted), persist=True)
    proc.l2a.close()

    assert written == {'lazy.nc': True, 'persisted.nc': False}
    with xr.open_dataset(lazy, mask_and_scale=False) as a, \
            xr.open_dataset(persisted, mask_and_scale=False) as b:
        # packed values, attributes and NetCDF chunking/compression
        assert a.identical(b)
        for v in a.variables:
            for key in ('chunksizes', 'zlib', 'complevel', 'shuffle'):
                assert a[v].encoding.get(key) == b[v].encoding.get(key)
