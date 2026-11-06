from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import xarray as xr
from GRSl2bgen.chlorophyll_a import Chl

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def chl():
    return Chl.__new__(Chl)

@pytest.fixture
def Rrs():
    return xr.DataArray(
        np.array([
            # 443 nm
            [
                [0.006, 0.004],
                [0.008, 0.003]
            ],
            # 490 nm
            [
                [0.005, 0.007],
                [0.006, 0.002]
            ],
            # 560 nm
            [
                [0.010, 0.010],
                [0.012, 0.008]
            ],
            # 665 nm
            [
                [0.010, 0.012],
                [0.008, 0.010]
            ],
            # 705 nm
            [
                [0.020, 0.018],
                [0.016, 0.015]
            ],
            # 740 nm
            [
                [0.012, 0.010],
                [0.014, 0.009]
            ]
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [443, 490, 560, 665, 705, 740],
            "y": [0, 1],
            "x": [0, 1]
        },
        name="Rrs"
    )
    
# ============================================================
# Tests for set_range
# ============================================================    

def test_set_range():
    chl = Chl.__new__(Chl)
    param = xr.DataArray(
        np.array([
            [0.0, np.nan, 168.0, 50.0],
            [0.01,99.9,-0.01,100.0]
        ]),
        dims=("y", "x"),
        coords={
            "y": [0.0, 1.0],
            "x": [0.0, 1.0, 2.0, 3.0],
        },
        name="Chl-a",
    )
    minval = 0.0
    maxval= 100.0
    
    result = chl.set_range(param=param, minval=minval, maxval=maxval)
    expected = np.array([
        [np.nan, np.nan, np.nan, 50.0],
        [0.01,99.9,np.nan, np.nan]
    ])
    
    np.testing.assert_array_equal(result, expected)

# ============================================================
# Tests for OCX_chl
# ============================================================

def test_OCX_chl(chl):

    ratio = xr.DataArray(
        [[0.1, 0.2], [0.3, 0.4]],
        dims=("y", "x")
    )

    acoef = [0.2424, -2.7423, 1.8017, 0.0015, -1.2280]

    result = chl.OCX_chl(ratio, acoef)

    expected = 10 ** (
        acoef[0]
        + acoef[1] * ratio.values
        + acoef[2] * ratio.values**2
        + acoef[3] * ratio.values**3
        + acoef[4] * ratio.values**4
    )

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("y", "x")

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


def test_OCX_chl_ndarray(chl):

    ratio = np.array([0.1, 0.2, 0.3])
    acoef = [1.0, 2.0, 3.0]

    result = chl.OCX_chl(ratio, acoef)

    expected = 10 ** (
        1.0 + 2.0 * ratio + 3.0 * ratio**2
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)


# ============================================================
# Tests for OC2_ratio
# ============================================================

def test_OC2_ratio(chl, Rrs):

    result = chl.OC2_ratio(Rrs)

    R490 = Rrs.sel(wl=490)
    R560 = Rrs.sel(wl=560)

    expected = np.log10(R490.values / R560.values)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("y", "x")

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


def test_OC2_ratio_masks_non_positive_bands(chl):

    Rrs = xr.DataArray(
        np.array([
            [[0.005, -0.005], [0.005, 0.005]],
            [[0.010, 0.010], [0.000, -0.010]]
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [490, 560],
            "y": [0, 1],
            "x": [0, 1]
        }
    )

    result = chl.OC2_ratio(Rrs)

    expected = np.array([
        [-0.3010299957, np.nan],
        [np.nan, np.nan]
    ])

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )


# ============================================================
# Tests for OC3_ratio
# ============================================================

def test_OC3_ratio(chl, Rrs):

    result = chl.OC3_ratio(Rrs)

    blue = np.maximum(
        Rrs.sel(wl=443).values,
        Rrs.sel(wl=490).values
    )

    R560 = Rrs.sel(wl=560).values

    expected = np.log10(blue / R560)

    assert result.dims == ("y", "x")

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


def test_OC3_ratio_masks_non_positive_bands(chl):

    Rrs = xr.DataArray(
        np.array([
            [[0.006, -0.004], [0.008, 0.003]],
            [[0.005, 0.007], [-0.006, 0.002]],
            [[0.010, 0.010], [0.012, 0.008]]
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [443, 490, 560],
            "y": [0, 1],
            "x": [0, 1]
        }
    )

    result = chl.OC3_ratio(Rrs)

    expected = np.array([
        [
            np.log10(0.006 / 0.010),
            np.log10(0.007 / 0.010)
        ],
        [
            np.log10(0.008 / 0.012),
            np.log10(0.003 / 0.008)
        ]
    ])

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )


# ============================================================
# Tests for OC2
# ============================================================

def test_OC2(chl, Rrs):

    acoef = [0.2424, -2.7423, 1.8017, 0.0015, -1.2280]

    result = chl.OC2(Rrs, acoef)

    ratio = np.log10(
        Rrs.sel(wl=490).values /
        Rrs.sel(wl=560).values
    )

    logchl = sum(
        acoef[i] * ratio**i
        for i in range(len(acoef))
    )

    expected = 10**logchl

    assert result.dims == ("y", "x")

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


# ============================================================
# Tests for OC3
# ============================================================

def test_OC3(chl, Rrs):

    acoef = [0.2424, -2.7423, 1.8017, 0.0015, -1.2280]

    result = chl.OC3(Rrs, acoef)

    # The implementation uses the global maximum of the blue bands.
    blue = np.max(
        Rrs.sel(wl=[443, 490]).values
    )

    R560 = Rrs.sel(wl=560).values

    ratio = np.log10(blue / R560)

    logchl = sum(
        acoef[i] * ratio**i
        for i in range(len(acoef))
    )

    expected = 10**logchl

    assert result.dims == ("y", "x")

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


# ============================================================
# Tests for RED2
# ============================================================

def test_RED2(chl, Rrs):

    result = chl.RED2(Rrs)

    R665 = Rrs.sel(wl=665).values
    R705 = Rrs.sel(wl=705).values

    expected = R705 / R665

    assert result.dims == ("y", "x")

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


def test_RED2_masks_non_positive_bands(chl):

    Rrs = xr.DataArray(
        np.array([
            [[0.010, 0.010], [0.000, -0.010]],
            [[0.020, -0.010], [0.020, 0.020]]
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [665, 705],
            "y": [0, 1],
            "x": [0, 1]
        }
    )

    result = chl.RED2(Rrs)

    expected = np.array([
        [2.0, np.nan],
        [np.nan, np.nan]
    ])

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )


# ============================================================
# Tests for RED3
# ============================================================

def test_RED3(chl, Rrs):

    result = chl.RED3(Rrs)

    R665 = Rrs.sel(wl=665).values
    R705 = Rrs.sel(wl=705).values
    R740 = Rrs.sel(wl=740).values

    expected = (1 / R665 - 1 / R705) * R740

    assert result.dims == ("y", "x")

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


# ============================================================
# Tests for M09B
# ============================================================

def test_M09B(chl, Rrs):

    acoef = np.array([232.329, 23.174])

    result = chl.M09B(Rrs, acoef)

    R665 = Rrs.sel(wl=665).values
    R705 = Rrs.sel(wl=705).values
    R740 = Rrs.sel(wl=740).values

    index = (1 / R665 - 1 / R705) * R740

    expected = acoef[0] * index + acoef[1]

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


# ============================================================
# Tests for NIRB
# ============================================================

def test_NIRB(chl, Rrs):

    acoef = np.array([11.2, 1.7])

    result = chl.NIRB(Rrs, acoef)

    R443 = Rrs.sel(wl=443).values
    R705 = Rrs.sel(wl=705).values

    index = R705 / R443

    expected = acoef[0] * index**acoef[1]

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


def test_NIRB_masks_non_positive_bands(chl):

    Rrs = xr.DataArray(
        np.array([
            [[0.005, -0.005], [0.000, 0.010]],
            [[0.010, 0.010], [0.010, -0.010]]
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [443, 705],
            "y": [0, 1],
            "x": [0, 1]
        }
    )

    result = chl.NIRB(Rrs)

    expected = np.array([
        [11.2 * (0.010 / 0.005)**1.7, np.nan],
        [np.nan, np.nan]
    ])

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )


# ============================================================
# Tests for G10B
# ============================================================

def test_G10B(chl, Rrs):

    acoef = np.array([113.36, -16.45, 1.124])

    result = chl.G10B(Rrs, acoef)

    R665 = Rrs.sel(wl=665).values
    R705 = Rrs.sel(wl=705).values
    R740 = Rrs.sel(wl=740).values

    index = (1 / R665 - 1 / R705) * R740
    base = acoef[0] * index + acoef[1]

    expected = np.where(
        base >= 0,
        base**acoef[2],
        np.nan
    )

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )


def test_G10B_masks_negative_base(chl):

    Rrs = xr.DataArray(
        np.array([
            # Rrs at 665 nm
            [[0.010, 0.010], [0.010, 0.010]],

            # Rrs at 705 nm
            [[0.020, 0.020], [0.020, 0.020]],

            # Rrs at 740 nm
            [[0.001, 0.001], [0.001, 0.001]]
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [665, 705, 740],
            "y": [0, 1],
            "x": [0, 1]
        }
    )

    result = chl.G10B(Rrs)

    expected = np.full((2, 2), np.nan)

    np.testing.assert_allclose(
        result.values,
        expected,
        equal_nan=True
    )


# ============================================================
# Tests for G11B
# ============================================================

def test_G11B(chl, Rrs):

    acoef = np.array([315.5, 215.95, -25.66])

    result = chl.G11B(Rrs, acoef)

    R665 = Rrs.sel(wl=665).values
    R705 = Rrs.sel(wl=705).values
    R740 = Rrs.sel(wl=740).values

    index = (1 / R665 - 1 / R705) * R740

    expected = (
        acoef[0] * index**2
        + acoef[1] * index
        + acoef[2]
    )

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


# ============================================================
# Tests for A14B
# ============================================================

def test_A14B(chl, Rrs):

    acoef = np.array([581.1, 25.5])

    result = chl.A14B(Rrs, acoef)

    R665 = Rrs.sel(wl=665).values
    R705 = Rrs.sel(wl=705).values
    R740 = Rrs.sel(wl=740).values

    index = (1 / R665 - 1 / R705) * R740

    expected = acoef[0] * index + acoef[1]

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


# ============================================================
# Tests for B16B
# ============================================================

def test_B16B(chl, Rrs):

    acoef = np.array([98.773, 34.763])

    result = chl.B16B(Rrs, acoef)

    R665 = Rrs.sel(wl=665).values
    R705 = Rrs.sel(wl=705).values
    R740 = Rrs.sel(wl=740).values

    index = (1 / R665 - 1 / R705) * R740

    expected = acoef[0] * index + acoef[1]

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8
    )


# ============================================================
# Tests for chl_gons
# ============================================================

def test_chl_gons():
    chl = Chl.__new__(Chl)
    rrs = xr.DataArray(
        np.array([
            [
                [0.010, 0.010, 0.010],
                [0.010, 0.000, 0.010]
            ],
            [
                [0.020, -0.010, 0.020],
                [0.020, 0.020, 0.020]
            ],
            [
                [0.010, 0.010, 0.010],
                [0.200, 0.010, 0.010]
            ]
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [650,700,750],
            "y": [0,1],
            "x": [0,1,2],
        }
    )
    
    result = chl.chl_gons(Rrs=rrs)
    assert result.dims == ("y", "x")
    assert np.isclose(result.sel(y=0, x=0), 75.91, atol=0.02)
    # R705 < 0
    assert np.isnan(result.sel(y=0, x=1))
    # bb < 0
    assert np.isnan(result.sel(y=1, x=0))
    # R665 = 0
    assert np.isnan(result.sel(y=1, x=1))
    assert np.isfinite(result.sel(y=0, x=2))

# ============================================================
# Tests for chl_ndci
# ============================================================

def test_chl_ndci():
    chl = Chl.__new__(Chl)
    rrs = xr.DataArray(
        np.array([
            [
                [3,2,5],
                [-0.5,1,6]
            ],
            [
                [-1,2,0],
                [4,0.5,1]
            ],
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [660,700],
            "y": [0,1],
            "x": [0,1,2],
        }
    )
    
    expected = np.array([
        [619.109, 14.039, 122.249],
        [445.989408, 6.925667, 51.673694],
    ])
    result = chl.chl_ndci(Rrs=rrs)
    assert result.dims == ("y", "x")
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-5)

# ============================================================
# Tests for gilerson2
# ============================================================

def test_chl_gilerson2():
    chl = Chl.__new__(Chl)
    R650 = np.array([
        [0.010, 0.010, 0.010],
        [0.010, 0.000, 0.010]
    ])

    R700 = np.array([
        [0.020, -0.010, 0.005],
        [0.020, 0.020, (0.4245 / 0.7864) * 0.010]
    ])
    Rrs = xr.DataArray(
        np.array([
            R650,
            R700
        ]),
        dims=("wl", "y", "x"),
        coords={
            "wl": [650, 700],
            "y": [0, 1],
            "x": [0, 1, 2]
        }
    )

    result = chl.chl_gilerson2(Rrs)

    expected = np.array([
        [85.235, np.nan, np.nan],
        [85.235, np.nan, 0.0]
    ])

    assert result.dims == ("y", "x")
    assert result.shape == (2, 3)

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-3,
        atol=1e-6,
        equal_nan=True
    )

# ============================================================
# Tests for create_mask_from_owt
# ============================================================

def test_create_mask_from_owt():
    chl = Chl.__new__(Chl)
    owt_raster = xr.DataArray(
        np.array([
                [1,2,3],
                [5,6,7]
        ]),
        dims=("y", "x"),
        coords={
            "y": [0,1],
            "x": [0,1,2],
        }
    )
    
    owt_index = np.array([1,3,4,7,11])

    expected = np.array([
        [1,0,1],
        [0,0,1]
    ])

    result = chl.create_mask_from_owt(xowt_raster=owt_raster, owt_index_to_keep=owt_index)
    
    np.testing.assert_array_equal(result, expected)


# ============================================================
# Tests for owt_blending
# ============================================================

def test_owt_blending(chl, monkeypatch):

    # ---------------------------------------------------
    # Mock chlorophyll algorithms
    # ---------------------------------------------------

    chl.Rrs = xr.DataArray(
        np.ones((3, 1, 3)),
        dims=("wl", "y", "x"),
        coords={
            "wl": [665, 705, 783],
            "y": [0],
            "x": [0, 1, 2]
        }
    )

    def fake_gons(Rrs):
        return xr.DataArray(
            [[10.0, np.nan, np.nan]],
            dims=("y", "x"),
            coords={"y": [0], "x": [0, 1, 2]}
        )

    def fake_ndci(Rrs):
        return xr.DataArray(
            [[20.0, 20.0, np.nan]],
            dims=("y", "x"),
            coords={"y": [0], "x": [0, 1, 2]}
        )

    def fake_gilerson2(Rrs):
        return xr.DataArray(
            [[30.0, 30.0, np.nan]],
            dims=("y", "x"),
            coords={"y": [0], "x": [0, 1, 2]}
        )

    def fake_OC2(Rrs, acoef):
        return xr.DataArray(
            [[40.0, 40.0, np.nan]],
            dims=("y", "x"),
            coords={"y": [0], "x": [0, 1, 2]}
        )

    monkeypatch.setattr(chl, "chl_gons", fake_gons)
    monkeypatch.setattr(chl, "chl_ndci", fake_ndci)
    monkeypatch.setattr(chl, "chl_gilerson2", fake_gilerson2)
    monkeypatch.setattr(chl, "OC2", fake_OC2)

    # ---------------------------------------------------
    # Mock OWT mask creation
    # ---------------------------------------------------

    monkeypatch.setattr(
        chl,
        "create_mask_from_owt",
        lambda owt_index, classes: owt_index.isin(classes)
    )

    # ---------------------------------------------------
    # Create OWT product
    # ---------------------------------------------------

    prod = xr.Dataset(
        data_vars={
            "owt_dist": (
                ("Nclasses", "y", "x"),
                np.array([
                    [[0.1, 0.1, 0.1]],
                    [[0.2, 0.2, 0.2]],
                    [[0.3, 0.3, 0.3]],
                    [[0.4, 0.4, 0.4]]
                ])
            ),
            "owt_index": (
                ("Nclasses", "y", "x"),
                np.array([
                    [[1, 1, 1]],
                    [[2, 2, 2]],
                    [[7, 7, 7]],
                    [[3, 3, 3]]
                ])
            )
        },
        coords={
            "Nclasses": np.arange(4),
            "y": [0],
            "x": [0, 1, 2]
        }
    )

    # ---------------------------------------------------
    # Run algorithm
    # ---------------------------------------------------

    result = chl.owt_blending(prod, Nclasses=10)

    # ---------------------------------------------------
    # Calculate expected values
    # ---------------------------------------------------

    eps = 1e-6

    weights = 1 / (np.array([0.1, 0.2, 0.3, 0.4]) + eps)

    # Pixel (0, 0): all four algorithms provide valid estimates.
    expected_0 = (
        10.0 * weights[0]
        + 20.0 * weights[1]
        + 30.0 * weights[2]
        + 40.0 * weights[3]
    ) / np.sum(weights)

    # Pixel (0, 1): Gons returns NaN.
    # Its weight must therefore be excluded from the normalization.
    expected_1 = (
        20.0 * weights[1]
        + 30.0 * weights[2]
        + 40.0 * weights[3]
    ) / np.sum(weights[1:])

    # Pixel (0, 2): all algorithms return NaN.
    expected = np.array([
        [expected_0, expected_1, np.nan]
    ])

    # ---------------------------------------------------
    # Assertions
    # ---------------------------------------------------

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("y", "x")
    assert result.shape == (1, 3)

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )

    # Check output metadata
    assert result.name == "Chla_OWTblend_Ta2025"
    assert result.attrs["units"] == "mg m-3"
    assert result.attrs["n_classes_blended"] == 4
    assert "description" in result.attrs
    assert "references" in result.attrs

# ============================================================
# Tests for process
# ============================================================

class TestProcess:
    
    def test_without_owt(self,monkeypatch):

        # Create input raster
        Rrs = xr.DataArray(
            np.ones((6, 2, 2)),
            dims=("wl", "y", "x"),
            coords={
                "wl": [443, 490, 560, 665, 705, 740],
                "y": [0, 1],
                "x": [0, 1]
            },
            name="Rrs"
        )

        raster = xr.Dataset({"Rrs": Rrs})

        chl = Chl(raster)

        # Mock standalone algorithms
        def fake_OC2(Rrs, acoef):
            return xr.DataArray(
                [[10.0, -1.0], [1200.0, 50.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]}
            )

        def fake_M09B(Rrs, acoef):
            return xr.DataArray(
                [[20.0, 30.0], [40.0, 50.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]}
            )

        def fake_NIRB(Rrs, acoef):
            return xr.DataArray(
                [[30.0, 40.0], [50.0, 60.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]}
            )

        monkeypatch.setattr(chl, "OC2", fake_OC2)
        monkeypatch.setattr(chl, "M09B", fake_M09B)
        monkeypatch.setattr(chl, "NIRB", fake_NIRB)

        # Run process
        chl.process()

        # Expected output
        expected_OC2 = np.array([
            [10.0, np.nan],
            [np.nan, 50.0]
        ])

        expected_M09B = np.array([
            [20.0, 30.0],
            [40.0, 50.0]
        ])

        expected_NIRB = np.array([
            [30.0, 40.0],
            [50.0, 60.0]
        ])

        # Check output type and variables
        assert isinstance(chl.output, xr.Dataset)

        assert set(chl.output.data_vars) == {
            "Chla_OC2nasa",
            "Chla_M09B",
            "Chla_NIRB"
        }

        # Check output dimensions
        assert chl.output["Chla_OC2nasa"].dims == ("y", "x")
        assert chl.output["Chla_M09B"].dims == ("y", "x")
        assert chl.output["Chla_NIRB"].dims == ("y", "x")

        # Check chlorophyll values
        np.testing.assert_allclose(
            chl.output["Chla_OC2nasa"].values,
            expected_OC2,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True
        )

        np.testing.assert_allclose(
            chl.output["Chla_M09B"].values,
            expected_M09B,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True
        )

        np.testing.assert_allclose(
            chl.output["Chla_NIRB"].values,
            expected_NIRB,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True
        )

        # Check metadata
        assert chl.output["Chla_OC2nasa"].attrs["units"] == "mg m-3"
        assert chl.output["Chla_M09B"].attrs["units"] == "mg m-3"
        assert chl.output["Chla_NIRB"].attrs["units"] == "mg m-3"

        assert chl.output["Chla_OC2nasa"].attrs["range"] == [0, 1200]
        assert chl.output["Chla_M09B"].attrs["range"] == [0, 1200]
        assert chl.output["Chla_NIRB"].attrs["range"] == [0, 1200]

        # Check that the wavelength coordinate has been removed
        assert "wl" not in chl.output.coords

    def test_with_owt(self,monkeypatch):

        # Create input raster
        Rrs = xr.DataArray(
            np.ones((6, 2, 2)),
            dims=("wl", "y", "x"),
            coords={
                "wl": [443, 490, 560, 665, 705, 740],
                "y": [0, 1],
                "x": [0, 1]
            },
            name="Rrs"
        )

        raster = xr.Dataset({"Rrs": Rrs})

        # Create OWT product
        xowt_prod = xr.Dataset(
            data_vars={
                "owt_dist_Spyrakos2018": (
                    ("Nowt", "y", "x"),
                    np.ones((2, 2, 2))
                ),
                "owt_index_Spyrakos2018": (
                    ("Nowt", "y", "x"),
                    np.ones((2, 2, 2))
                )
            },
            coords={
                "Nowt": [0, 1],
                "y": [0, 1],
                "x": [0, 1]
            }
        )

        chl = Chl(
            raster,
            xowt_prod=xowt_prod,
            Nclasses=2
        )

        # Mock standalone algorithms
        def fake_OC2(Rrs, acoef):
            return xr.DataArray(
                [[10.0, 20.0], [30.0, 40.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]}
            )

        def fake_M09B(Rrs, acoef):
            return xr.DataArray(
                [[20.0, 30.0], [40.0, 50.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]}
            )

        def fake_NIRB(Rrs, acoef):
            return xr.DataArray(
                [[30.0, 40.0], [50.0, 60.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]}
            )

        monkeypatch.setattr(chl, "OC2", fake_OC2)
        monkeypatch.setattr(chl, "M09B", fake_M09B)
        monkeypatch.setattr(chl, "NIRB", fake_NIRB)

        # Mock OWT blending
        calls = {}

        def fake_owt_blending(
            xowt_prod,
            Nclasses=3,
            owt_dist_var="owt_dist",
            owt_index_var="owt_index",
            eps=1e-6
        ):
            calls["xowt_prod"] = xowt_prod
            calls["Nclasses"] = Nclasses
            calls["owt_dist_var"] = owt_dist_var
            calls["owt_index_var"] = owt_index_var

            return xr.DataArray(
                [[100.0, 200.0], [1300.0, -10.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]},
                name="Chla_OWTblend_Ta2025",
                attrs={
                    "units": "mg m-3",
                    "n_classes_blended": 2
                }
            )

        monkeypatch.setattr(chl, "owt_blending", fake_owt_blending)

        # Run process
        chl.process()

        # Expected OWT result after range filtering
        expected_OWT = np.array([
            [100.0, 200.0],
            [np.nan, np.nan]
        ])

        # Check output variables
        assert isinstance(chl.output, xr.Dataset)

        assert set(chl.output.data_vars) == {
            "Chla_OC2nasa",
            "Chla_M09B",
            "Chla_NIRB",
            "Chla_OWTblend_Ta2025"
        }

        # Check OWT blending result
        np.testing.assert_allclose(
            chl.output["Chla_OWTblend_Ta2025"].values,
            expected_OWT,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True
        )

        # Check OWT blending arguments
        assert calls["xowt_prod"] is xowt_prod
        assert calls["Nclasses"] == 2
        assert calls["owt_dist_var"] == "owt_dist_Spyrakos2018"
        assert calls["owt_index_var"] == "owt_index_Spyrakos2018"

        # Check OWT metadata
        assert chl.output["Chla_OWTblend_Ta2025"].attrs["units"] == "mg m-3"
        assert chl.output["Chla_OWTblend_Ta2025"].attrs["range"] == [0, 1200]
        assert chl.output["Chla_OWTblend_Ta2025"].attrs["n_classes_blended"] == 2

        # Check output dimensions
        assert chl.output["Chla_OWTblend_Ta2025"].dims == ("y", "x")

        # Check that the wavelength coordinate has been removed
        assert "wl" not in chl.output.coords
