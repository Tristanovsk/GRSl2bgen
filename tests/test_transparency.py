"""Unit tests of the transparency algorithms (GRSl2bgen.transparency)."""

import numpy as np
import xarray as xr

from GRSl2bgen.transparency import Transparency


def test_kd490_kd2():
    # Rrs(490) / Rrs(560) = 1, 2, 0.5, then a negative Rrs(490); the other bands
    # are set so that a wrong band selection changes the result
    ratio = np.array([1., 2., 0.5, -1.])
    rrs = np.stack([np.full(4, 0.5), 0.01 * ratio, np.full(4, 0.01), np.full(4, 0.5)])
    raster = xr.Dataset({'Rrs': (('wl', 'x'), rrs)}, coords={'wl': [443, 490, 560, 665]})
    kd = Transparency(raster).kd490_KD2().values
    # MERIS KD2E of the NASA OBPG Kd_490 ATBD
    X = np.log10(ratio[:3])
    expected = 0.0166 + 10 ** (-0.8641 - 1.6549 * X + 2.0112 * X ** 2
                               - 2.5174 * X ** 3 - 1.1035 * X ** 4)
    np.testing.assert_allclose(kd[:3], expected)
    np.testing.assert_allclose(kd[0], 0.0166 + 10 ** -0.8641)
    assert np.isnan(kd[3])


def test_depth_lee05_solves_the_equation():
    K1 = np.array([0.05, 0.2, 1.0, 3.0])
    K2 = np.array([0.1, 0.05, 0.3, 0.0])
    tau = np.log(100)
    z = Transparency._depth_lee05(K1, K2, tau)
    np.testing.assert_allclose(K1 * z + K2 * z / np.sqrt(1 + z), tau, rtol=1e-6)
    # K2 = 0: constant Kpar
    np.testing.assert_allclose(z[3], tau / 3.0, rtol=1e-6)


def test_depth_lee05_invalid():
    z = Transparency._depth_lee05(np.array([-0.01, np.nan, 1e-6]),
                                  np.array([0.1, 0.1, 0.0]), np.log(100))
    # K1 <= 0, NaN input, root beyond 2 km
    assert np.all(np.isnan(z))


def test_zeu_lee05_dask():
    a490 = xr.DataArray(np.array([[0.03, 0.1], [0.5, 2.0]]), dims=('y', 'x')).chunk(1)
    bb490 = xr.DataArray(np.array([[0.002, 0.005], [0.02, 0.1]]), dims=('y', 'x')).chunk(1)
    out = Transparency.zeu_Lee05(a490, bb490, sza=30.).compute()
    zeu = out['Zeu'].values
    # euphotic depth decreases with absorption and backscattering
    assert np.all(np.diff(zeu.ravel()) < 0)
    np.testing.assert_allclose(out['Kd_par'].values, np.log(100) / zeu)


def test_kd_par_mo07():
    kd490 = xr.DataArray([0.02, 0.1, 1.0, 0.0])
    kd = Transparency.kd_par_Mo07(kd490).values
    np.testing.assert_allclose(kd[:3], 0.0864 + 0.884 * kd490[:3] - 0.00137 / kd490[:3])
    assert np.isnan(kd[3])
