import io
import unittest

import numpy as np
import xarray as xr

from GRSl2bgen.output import L2bProduct


def _roundtrip(values):
    """Pack `values` as on export (int16, _FillValue -32768) and read them back."""
    values = np.asarray(values, dtype=np.float64)
    scale_factor, add_offset = L2bProduct.compute_scale_and_offset(values, nbit=16)
    encoding = {'v': {'dtype': 'int16', 'scale_factor': scale_factor,
                      'add_offset': add_offset, '_FillValue': -32768}}
    raw = xr.Dataset({'v': ('i', values)}).to_netcdf(encoding=encoding)
    with xr.open_dataset(io.BytesIO(raw), mask_and_scale=False) as ds:
        packed = ds['v'].values
    with xr.open_dataset(io.BytesIO(raw)) as ds:
        unpacked = ds['v'].values
    return scale_factor, packed, unpacked


class TestPacking(unittest.TestCase):
    """int16 packing of the L2B variables."""

    def test_min_and_max_are_not_fill_value(self):
        values = [0., 0.01, 600., 1200., np.nan]
        scale_factor, packed, unpacked = _roundtrip(values)
        self.assertEqual(packed[0], -32767)
        self.assertEqual(packed[3], 32767)
        self.assertEqual(packed[4], -32768)
        np.testing.assert_allclose(unpacked[:4], values[:4], atol=scale_factor / 2)
        self.assertTrue(np.isnan(unpacked[4]))

    def test_constant_field(self):
        _, packed, unpacked = _roundtrip([3., 3., np.nan])
        np.testing.assert_array_equal(packed, [0, 0, -32768])
        np.testing.assert_allclose(unpacked[:2], [3., 3.])

    def test_all_nan_field(self):
        _, packed, unpacked = _roundtrip([np.nan, np.nan])
        np.testing.assert_array_equal(packed, [-32768, -32768])
        self.assertTrue(np.all(np.isnan(unpacked)))


if __name__ == '__main__':
    unittest.main()
