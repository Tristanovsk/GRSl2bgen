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
