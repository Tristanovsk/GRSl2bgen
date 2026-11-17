import numpy as np
import pandas as pd
import pytest
import xarray as xr

from GRSl2bgen.suspended_particulate_matter import Spm

OWT_DIST_VAR = 'owt_dist_invRrsWT'
OWT_INDEX_VAR = 'owt_index_invRrsWT'

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def Rrs():
    """Synthetic reflectance cube with all required wavelengths."""
    wavelengths = [443, 490, 560, 665, 705, 740, 865]

    values = np.array([
        [
            [0.010, 0.012, 0.015, 0.020, 0.025, 0.030, 0.035],
            [0.012, 0.014, 0.017, 0.025, 0.030, 0.035, 0.040],
        ],
        [
            [0.015, 0.017, 0.020, 0.040, 0.045, 0.050, 0.060],
            [0.020, 0.022, 0.025, 0.060, 0.070, 0.080, 0.090],
        ],
    ], dtype=np.float32)

    return xr.DataArray(
        values,
        dims=("y", "x", "wl"),
        coords={
            "y": [10, 20],
            "x": [30, 40],
            "wl": wavelengths,
        },
        name="Rrs",
    )

@pytest.fixture
def raster(Rrs):
    """Synthetic raster dataset."""
    return xr.Dataset({"Rrs": Rrs})

@pytest.fixture
def spm(raster):
    return Spm(raster)

@pytest.fixture
def xowt_prod():
    """Synthetic ranked OWT classification product."""
    return xr.Dataset(
        {
            OWT_DIST_VAR: (
                ("Nclasses", "y", "x"),
                np.array([
                    [
                        [0.1, 0.2],
                        [0.3, 0.4],
                    ],
                    [
                        [0.2, 0.3],
                        [0.4, 0.5],
                    ],
                ], dtype=np.float32),
            ),
            OWT_INDEX_VAR: (
                ("Nclasses", "y", "x"),
                np.array([
                    [
                        [1, 2],
                        [3, 4],
                    ],
                    [
                        [2, 3],
                        [4, 1],
                    ],
                ], dtype=np.float32),
            ),
        },
        coords={
            "Nclasses": [0, 1],
            "y": [10, 20],
            "x": [30, 40],
        },
    )


# ============================================================
# Initialization
# ============================================================

def test_init(raster):
    result = Spm(raster)

    assert result.raster is raster
    xr.testing.assert_identical(result.Rrs, raster["Rrs"])
    assert result.xowt_prod is None
    assert result.Nclasses == 2
    assert result.output is None



def test_init_custom_parameter(raster):
    raster["custom"] = raster["Rrs"] * 2

    result = Spm(
        raster,
        param="custom",
        Nclasses=3,
    )

    assert result.Rrs.identical(raster["custom"])
    assert result.Nclasses == 3


def test_init_with_owt_product(raster, xowt_prod):
    result = Spm(raster, xowt_prod=xowt_prod)

    assert result.xowt_prod is xowt_prod


# ============================================================
# Tests for set_range
# ============================================================

def test_set_range(spm):
    values = xr.DataArray(
        [0, 1, 5, 10, 20, 30],
        dims="x",
    )

    result = spm.set_range(values, minval=1, maxval=20)

    expected = np.array([
        np.nan, np.nan, 5, 10, np.nan, np.nan
    ])

    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_set_range_custom_thresholds(spm):
    values = xr.DataArray(
        [1, 5, 10, 15, 20],
        dims="x",
    )

    result = spm.set_range(
        values,
        minval=5,
        maxval=15,
    )

    expected = np.array([
        np.nan, np.nan, 10, np.nan, np.nan
    ])

    np.testing.assert_allclose(result.values, expected, equal_nan=True)


# ============================================================
# Tests for nechad_relationship
# ============================================================

def test_nechad_relationship():
    Rrs = xr.DataArray([0.01, 0.02, 0.03], dims="x")
    coefs = [342.1, 0.19563]

    result = Spm.nechad_relationship(Rrs, coefs)

    rho = np.pi * Rrs
    expected = coefs[0] * rho / (1 - rho / coefs[1])

    xr.testing.assert_allclose(result, expected)


def test_nechad_relationship_zero():
    Rrs = xr.DataArray([0.0], dims="x")

    result = Spm.nechad_relationship(
        Rrs,
        [342.1, 0.19563],
    )

    np.testing.assert_allclose(result.values, [0.0])


def test_nechad_relationship_preserves_dimensions():
    Rrs = xr.DataArray(
        np.ones((2, 3)) * 0.01,
        dims=("y", "x"),
    )

    result = Spm.nechad_relationship(
        Rrs,
        [342.1, 0.19563],
    )

    assert result.dims == ("y", "x")
    assert result.shape == (2, 3)


# ============================================================
# Tests for obs2co
# ============================================================

class TestObs2co:
    
    def test_nominal(self,spm):
        result = spm.obs2co()

        assert result.dims == ("y", "x")
        assert result.shape == (2, 2)

        assert np.all(
            (result.values[np.isfinite(result.values)] >= 0)
            & (result.values[np.isfinite(result.values)] <= 2000)
        )

    def test_low_reflectance(self):
        Rrs = xr.DataArray(
            np.array([[[0.01, 0.01]]]),
            dims=("y", "x", "wl"),
            coords={"wl": [665, 865]},
        )

        instance = Spm(xr.Dataset({"Rrs": Rrs}))

        result = instance.obs2co()

        expected = instance.nechad_relationship(
            Rrs.sel(wl=665),
            [610.94, 0.2324],
        )

        xr.testing.assert_allclose(result, expected)

    def test_high_reflectance(self):
        Rrs = xr.DataArray(
            np.array([[[0.15, 0.20]]]),
            dims=("y", "x", "wl"),
            coords={"wl": [665, 865]},
        )

        instance = Spm(xr.Dataset({"Rrs": Rrs}))

        result = instance.obs2co()

        red = Rrs.sel(wl=665)
        nir = Rrs.sel(wl=865)

        expected = 691.13 * (nir / red) ** 2.5411

        xr.testing.assert_allclose(
            result.drop_vars("wl", errors="ignore"),
            expected,
        )

    def test_negative_red_is_masked(self):
        Rrs = xr.DataArray(
            [[[-0.01, 0.02]]],
            dims=("y", "x", "wl"),
            coords={"wl": [665, 865]},
        )

        instance = Spm(xr.Dataset({"Rrs": Rrs}))

        result = instance.obs2co()

        assert np.isnan(result.values[0, 0])

    def test_negative_nir_is_masked(self):
        Rrs = xr.DataArray(
            [[[0.02, -0.01]]],
            dims=("y", "x", "wl"),
            coords={"wl": [665, 865]},
        )

        instance = Spm(xr.Dataset({"Rrs": Rrs}))

        result = instance.obs2co()

        assert np.isnan(result.values[0, 0])


# ============================================================
# Tests for turbi_D15
# ============================================================

def test_turbi_D15(spm):
    result = spm.turbi_D15()

    assert result.dims == ("y", "x")
    assert result.shape == (2, 2)

    assert np.all(
        (result.values[np.isfinite(result.values)] >= 0)
        & (result.values[np.isfinite(result.values)] <= 2000)
    )


def test_turbi_D15_low_reflectance():
    Rrs = xr.DataArray(
        [[[0.005]]],
        dims=("y", "x", "wl"),
        coords={"wl": [665]},
    )

    instance = Spm(xr.Dataset({"Rrs": Rrs}))

    result = instance.turbi_D15()

    expected = instance.nechad_relationship(
        Rrs.sel(wl=665),
        [228.1, 0.1641],
    )

    xr.testing.assert_allclose(result, expected)


def test_turbi_D15_high_reflectance():
    Rrs = xr.DataArray(
        [[[0.05]]],
        dims=("y", "x", "wl"),
        coords={"wl": [665]},
    )

    instance = Spm(xr.Dataset({"Rrs": Rrs}))

    result = instance.turbi_D15()

    expected = instance.nechad_relationship(
        Rrs.sel(wl=665),
        [3078.9, 0.2112],
    )

    xr.testing.assert_allclose(result, expected)

# ============================================================
# Tests for spm_N10
# ============================================================

def test_spm_N10(spm):
    result = spm.spm_N10()

    assert result.dims == ("y", "x")
    assert result.shape == (2, 2)

    assert np.all(
        (result.values[np.isfinite(result.values)] >= 0)
        & (result.values[np.isfinite(result.values)] <= 2000)
    )


def test_spm_N10_custom_coefficients(spm):
    result = spm.spm_N10(
        coefs=[100, 0.2],
    )

    expected = Spm.nechad_relationship(
        spm.Rrs.sel(wl=665),
        [100, 0.2],
    )

    xr.testing.assert_allclose(result, expected.where(expected >= 0, 0).where(expected <= 2000))


def test_spm_N10_valid_limit():
    Rrs = xr.DataArray(
        [[[0.15]]],
        dims=("y", "x", "wl"),
        coords={"wl": [665]},
    )

    instance = Spm(xr.Dataset({"Rrs": Rrs}))

    result = instance.spm_N10(
        valid_limit=[0, 10],
    )

    assert np.isnan(result.values[0, 0]) or result.values[0, 0] <= 10


# ============================================================
# Tests for _spm_jiang_green
# ============================================================

def test_spm_jiang_green():
    bands = [
        xr.DataArray([[v]], dims=("y", "x"))
        for v in [0.01, 0.012, 0.015, 0.02]
    ]

    result = Spm._spm_jiang_green(*bands)

    assert result.shape == (1, 1)
    assert result.dtype == np.float32


def test_spm_jiang_green_negative_sqrt_argument():
    bands = [
        xr.DataArray([[0.01]], dims=("y", "x")),
        xr.DataArray([[0.01]], dims=("y", "x")),
        xr.DataArray([[-0.1]], dims=("y", "x")),
        xr.DataArray([[0.01]], dims=("y", "x")),
    ]

    result = Spm._spm_jiang_green(*bands)

    assert np.isnan(result.values[0, 0])


def test_spm_jiang_green_preserves_dimensions():
    bands = [
        xr.DataArray(np.full((2, 3), v), dims=("y", "x"))
        for v in [0.01, 0.012, 0.015, 0.02]
    ]

    result = Spm._spm_jiang_green(*bands)

    assert result.dims == ("y", "x")
    assert result.shape == (2, 3)


# ============================================================
# Tests for _spm_jiang_red
# ============================================================

def test_spm_jiang_red():
    bands = [
        xr.DataArray([[v]], dims=("y", "x"))
        for v in [0.01, 0.012, 0.015, 0.02]
    ]

    result = Spm._spm_jiang_red(*bands)

    assert result.shape == (1, 1)
    assert result.dtype == np.float32


def test_spm_jiang_red_negative_red():
    bands = [
        xr.DataArray([[0.01]], dims=("y", "x")),
        xr.DataArray([[0.01]], dims=("y", "x")),
        xr.DataArray([[0.01]], dims=("y", "x")),
        xr.DataArray([[-0.01]], dims=("y", "x")),
    ]

    result = Spm._spm_jiang_red(*bands)

    assert np.isnan(result.values[0, 0])


def test_spm_jiang_red_negative_blue():
    bands = [
        xr.DataArray([[-0.02]], dims=("y", "x")),
        xr.DataArray([[0.01]], dims=("y", "x")),
        xr.DataArray([[0.01]], dims=("y", "x")),
        xr.DataArray([[0.02]], dims=("y", "x")),
    ]

    result = Spm._spm_jiang_red(*bands)

    assert np.isnan(result.values[0, 0])


# ============================================================
# Tests for _spm_zhang2014
# ============================================================

def test_spm_zhang2014():
    R705 = xr.DataArray(
        [[0.02, 0.03]],
        dims=("y", "x"),
    )

    result = Spm._spm_zhang2014(R705)

    expected = 362507.0 * R705 ** 2.3222

    np.testing.assert_allclose(result, expected, rtol=1e-6)

    assert result.dtype == np.float32


def test_spm_zhang2014_custom_coefficients():
    R705 = xr.DataArray(
        [[0.01, 0.02]],
        dims=("y", "x"),
    )

    result = Spm._spm_zhang2014(
        R705,
        acoef=[100, 2],
    )

    expected = 100 * R705 ** 2

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_spm_zhang2014_negative_values():
    R705 = xr.DataArray(
        [[-0.01, 0.02]],
        dims=("y", "x"),
    )

    result = Spm._spm_zhang2014(R705)

    assert np.isnan(result.values[0, 0])
    assert np.isfinite(result.values[0, 1])


# ============================================================
# Tests for _spm_binding2010
# ============================================================

def test_spm_binding2010():
    R740 = xr.DataArray(
        [[0.01, 0.02]],
        dims=("y", "x"),
    )

    result = Spm._spm_binding2010(R740)

    rho = R740 * np.pi
    scale = 51.162 / (0.554 * 0.019)
    denom = (0.54 * 128.123 / np.pi) + 0.48 * rho

    expected = scale * (2.8 * rho / denom - 0.00027)

    np.testing.assert_allclose(result, expected, rtol=1e-6)

    assert result.dtype == np.float32


def test_spm_binding2010_preserves_dimensions():
    R740 = xr.DataArray(
        np.ones((2, 3)) * 0.01,
        dims=("y", "x"),
    )

    result = Spm._spm_binding2010(R740)

    assert result.dims == ("y", "x")
    assert result.shape == (2, 3)


# ============================================================
# Tests for create_mask_from_owt
# ============================================================

class TestCreateMaskFromOwt:
    
    def test_single_class(self):
        classes = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
        )

        result = Spm.create_mask_from_owt(classes, [1])

        expected = np.array([
            [True, False],
            [False, False],
        ])

        np.testing.assert_array_equal(result.values, expected)
        assert result.dtype == bool

    def test_multiple_classes(self):
        classes = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
        )

        result = Spm.create_mask_from_owt(classes, [1, 3])

        expected = np.array([
            [True, False],
            [True, False],
        ])

        np.testing.assert_array_equal(result.values, expected)

    def test_no_matches(self):
        classes = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
        )

        result = Spm.create_mask_from_owt(classes, [5])

        assert not result.any()

    def test_dimensions(self):
        classes = xr.DataArray(
            np.ones((2, 3)),
            dims=("y", "x"),
        )

        result = Spm.create_mask_from_owt(classes, [1, 2])

        assert result.dims == ("y", "x")
        assert result.shape == (2, 3)


# ============================================================
# Tests for spm_blending
# ============================================================

class TestSpmBlending:
    
    def test_nominal(self,spm, xowt_prod):
        result = spm.spm_blending(xowt_prod,
            owt_dist_var=OWT_DIST_VAR,
            owt_index_var=OWT_INDEX_VAR,)

        assert result.name == "SPM_OWTblend_Ta2025"
        assert result.dims == ("y", "x")
        assert result.shape == (2, 2)
        assert result.dtype == np.float32

    def test_attributes(self,spm, xowt_prod):
        result = spm.spm_blending(xowt_prod,
            owt_dist_var=OWT_DIST_VAR,
            owt_index_var=OWT_INDEX_VAR,)

        assert result.attrs["units"] == "mg L-1"
        assert result.attrs["n_classes_blended"] == 2
        assert "Jiang" in result.attrs["references"]
        assert "Zhang" in result.attrs["references"]

    def test_custom_number_of_classes(self,spm, xowt_prod):
        result = spm.spm_blending(
            xowt_prod,
            owt_dist_var=OWT_DIST_VAR,
            owt_index_var=OWT_INDEX_VAR,
            Nclasses=1,
        )

        assert result.attrs["n_classes_blended"] == 1

    def test_caps_number_of_classes(self,spm, xowt_prod):
        result = spm.spm_blending(
            xowt_prod,
            owt_dist_var=OWT_DIST_VAR,
            owt_index_var=OWT_INDEX_VAR,
            Nclasses=10,
        )

        assert result.attrs["n_classes_blended"] == 2

    def test_without_Nclasses_dimension_raises_error(self,spm):
        product = xr.Dataset(
            {
                "owt_dist": (
                    ("y", "x"),
                    np.ones((2, 2)),
                ),
                "owt_index": (
                    ("y", "x"),
                    np.ones((2, 2)),
                ),
            }
        )

        with pytest.raises(ValueError, match="ranked OWT"):
            spm.spm_blending(
                product,
                owt_dist_var="owt_dist",
                owt_index_var="owt_index",
            )

    def test_custom_variable_names(self,spm):
        product = xr.Dataset(
            {
                "distance": (
                    ("Nclasses", "y", "x"),
                    np.ones((2, 2, 2)),
                ),
                "classification": (
                    ("Nclasses", "y", "x"),
                    np.ones((2, 2, 2)),
                ),
            }
        )

        result = spm.spm_blending(
            product,
            owt_dist_var="distance",
            owt_index_var="classification",
            limits=False,
        )

        assert result.shape == (2, 2)

    def test_limits_disabled(self,spm, xowt_prod):
        result = spm.spm_blending(
            xowt_prod,
            owt_dist_var=OWT_DIST_VAR,
            owt_index_var=OWT_INDEX_VAR,
            limits=False,
        )

        assert result.shape == (2, 2)

    def test_nan_distance(self,spm, xowt_prod):
        product = xowt_prod.copy()

        product[OWT_DIST_VAR][0, 0, 0] = np.nan

        result = spm.spm_blending(product,
            owt_dist_var=OWT_DIST_VAR,
            owt_index_var=OWT_INDEX_VAR,)

        assert result.shape == (2, 2)

    def test_nan_spectra(self,spm, xowt_prod):
        instance = Spm(
            xr.Dataset({
                "Rrs": spm.Rrs.where(spm.Rrs.wl != 443)
            })
        )

        result = instance.spm_blending(xowt_prod,
            owt_dist_var=OWT_DIST_VAR,
            owt_index_var=OWT_INDEX_VAR,)

        assert result.shape == (2, 2)


# ============================================================
# Tests for process
# ============================================================

class TestProcess:
    
    def test_nominal(self,spm):
        spm.process()

        assert spm.output is not None

        expected_variables = [
            "SPM_obs2co",
            "TURB_dogliotti",
            "SPM_nechad",
        ]

        for name in expected_variables:
            assert name in spm.output.data_vars


    def test_output_dimensions(self,spm):
        spm.process()

        assert spm.output.sizes["y"] == 2
        assert spm.output.sizes["x"] == 2


    def test_output_attributes(self,spm):
        spm.process()

        assert spm.output["SPM_obs2co"].attrs["units"] == "mg/L"
        assert spm.output["SPM_obs2co"].attrs["reference"] == "OBS2CO"

        assert spm.output["TURB_dogliotti"].attrs["units"] == "FNU"
        assert spm.output["TURB_dogliotti"].attrs["reference"] == "Dogliotti et al., 2015"

        assert spm.output["SPM_nechad"].attrs["units"] == "mg/l"
        assert spm.output["SPM_nechad"].attrs["reference"] == "Nechad et al., 2010"


    def test_output_ranges(self,spm):
        spm.process()

        for name in ["SPM_obs2co", "TURB_dogliotti", "SPM_nechad"]:
            assert spm.output[name].attrs["range"] == [0, 2000]


    def test_with_owt_blending(self,raster, xowt_prod):
        instance = Spm(
            raster,
            xowt_prod=xowt_prod,
            Nclasses=2,
        )

        instance.process()

        assert "SPM_OWTblend_Ta2025" in instance.output.data_vars

        assert instance.output["SPM_OWTblend_Ta2025"].attrs["range"] == [0, 2000]


    def test_without_owt_blending(self,raster):
        instance = Spm(raster)

        instance.process()

        assert "SPM_OWTblend_Ta2025" not in instance.output.data_vars


    def test_drops_wavelength_dimension(self,spm):
        spm.process()

        assert "wl" not in spm.output.dims


    def test_returns_none(self,spm):
        result = spm.process()

        assert result is None


# ============================================================
# Recipe of the invRrs water types and fallback
# ============================================================

def test_recipe_covers_the_14_water_types():
    from GRSl2bgen.suspended_particulate_matter import SPM_OWT_RECIPE
    classes = sorted(c for cls, _ in SPM_OWT_RECIPE.values() for c in cls)
    assert classes == list(range(1, 15))


def _single_class(spm, cls):
    # every pixel in class `cls` (both ranks), distance 0.1
    return xr.Dataset({OWT_DIST_VAR: (("Nclasses", "y", "x"), np.full((2, 2, 2), 0.1)),
                       OWT_INDEX_VAR: (("Nclasses", "y", "x"), np.full((2, 2, 2), float(cls)))},
                      coords={"Nclasses": [0, 1], "y": spm.Rrs.y, "x": spm.Rrs.x})


def test_recipe_algorithm_of_each_group(spm):
    R = {w: spm.Rrs.sel(wl=w) for w in (443, 490, 560, 665, 705)}
    green = Spm._spm_jiang_green(R[443], R[490], R[560], R[665])
    zhang = Spm._spm_zhang2014(R[705])
    out1 = spm.spm_blending(_single_class(spm, 1), owt_dist_var=OWT_DIST_VAR, owt_index_var=OWT_INDEX_VAR,
                            limits=False, fallback=False)
    out12 = spm.spm_blending(_single_class(spm, 12), owt_dist_var=OWT_DIST_VAR, owt_index_var=OWT_INDEX_VAR,
                             limits=False, fallback=False)
    np.testing.assert_allclose(out1.values, green.transpose(*out1.dims).values, rtol=1e-5)
    np.testing.assert_allclose(out12.values, zhang.transpose(*out12.dims).values, rtol=1e-5)


def test_fallback_outside_validity_range():
    # class 12 (Zhang 705, valid 20-1000): pixel 0 inside the range, pixel 1 below it (Rrs705 = 0.005)
    wl = [443, 490, 560, 665, 705, 740]
    rrs = np.array([[[0.01, 0.012, 0.015, 0.02, 0.025, 0.03],
                     [0.01, 0.012, 0.015, 0.02, 0.005, 0.003]]])               # (y, x, wl)
    spm = Spm(xr.Dataset({'Rrs': (('y', 'x', 'wl'), rrs)}, coords={'y': [0], 'x': [0, 1], 'wl': wl}))
    prod = xr.Dataset({OWT_DIST_VAR: (("Nclasses", "y", "x"), np.full((2, 1, 2), 0.1)),
                       OWT_INDEX_VAR: (("Nclasses", "y", "x"), np.full((2, 1, 2), 12.))},
                      coords={"Nclasses": [0, 1], "y": [0], "x": [0, 1]})
    zhang = Spm._spm_zhang2014(spm.Rrs.sel(wl=705)).values.ravel()
    assert 20 <= zhang[0] <= 1000 and 0 <= zhang[1] < 20
    without = spm.spm_blending(prod, owt_dist_var=OWT_DIST_VAR, owt_index_var=OWT_INDEX_VAR, fallback=False).values.ravel()
    with_ = spm.spm_blending(prod, owt_dist_var=OWT_DIST_VAR, owt_index_var=OWT_INDEX_VAR).values.ravel()
    np.testing.assert_allclose(without[0], zhang[0], rtol=1e-5)
    assert np.isnan(without[1])                                    # outside the range: no valid estimate
    np.testing.assert_allclose(with_, zhang, rtol=1e-5)            # fallback: algorithm of the best class


def test_fallback_not_applied_to_unclassified_pixels():
    wl = [443, 490, 560, 665, 705, 740]
    rrs = np.array([[[0.01, 0.012, 0.015, 0.02, 0.005, 0.003]]])
    spm = Spm(xr.Dataset({'Rrs': (('y', 'x', 'wl'), rrs)}, coords={'y': [0], 'x': [0], 'wl': wl}))
    prod = xr.Dataset({OWT_DIST_VAR: (("Nclasses", "y", "x"), np.full((2, 1, 1), np.nan)),
                       OWT_INDEX_VAR: (("Nclasses", "y", "x"), np.full((2, 1, 1), np.nan))},
                      coords={"Nclasses": [0, 1], "y": [0], "x": [0]})
    out = spm.spm_blending(prod, owt_dist_var=OWT_DIST_VAR, owt_index_var=OWT_INDEX_VAR)
    assert np.isnan(out.values).all()


def test_custom_recipe(spm):
    prod = _single_class(spm, 1)
    out = spm.spm_blending(prod, owt_dist_var=OWT_DIST_VAR, owt_index_var=OWT_INDEX_VAR, limits=False,
                           recipe={'binding2010': ([1], (0, 2000))}, fallback=False)
    binding = Spm._spm_binding2010(spm.Rrs.sel(wl=740))
    np.testing.assert_allclose(out.values, binding.transpose(*out.dims).values, rtol=1e-5)
