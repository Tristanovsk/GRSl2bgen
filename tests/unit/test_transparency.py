import numpy as np
import pytest
import xarray as xr
import dask.array as da

from GRSl2bgen.transparency import Transparency, KD2E


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def Rrs():
    """Synthetic reflectance cube (y, x, wl)."""
    return np.array([
        [
            [0.010, 0.012, 0.015, 0.020, 0.025, 0.030, 0.035],
            [0.012, 0.014, 0.018, 0.025, 0.030, 0.035, 0.040],
        ],
        [
            [0.008, 0.010, 0.013, 0.018, 0.022, 0.028, 0.032],
            [0.015, 0.018, 0.022, 0.040, 0.050, 0.060, 0.070],
        ],
    ], dtype=np.float64)


@pytest.fixture
def raster(Rrs):
    """Synthetic xarray raster."""
    return xr.Dataset(
        {
            "Rrs": (
                ("y", "x", "wl"),
                Rrs,
            ),
            "sza": (
                ("y", "x"),
                np.array([
                    [30.0, 35.0],
                    [40.0, 45.0],
                ]),
            ),
        },
        coords={
            "y": [10, 20],
            "x": [30, 40],
            "wl": [443, 490, 560, 665, 705, 740, 865],
        },
    )


@pytest.fixture
def transparency(raster):
    return Transparency(raster)


# ============================================================
# Initialization
# ============================================================

class TestInitialization:
    
    def test_nominal(self,raster):
        result = Transparency(raster)

        assert result.raster is raster
        xr.testing.assert_identical(result.Rrs, raster["Rrs"])
        xr.testing.assert_identical(result.sza, raster["sza"])

        assert result.output is None
        assert result.flags == {}


    def test_custom_parameter(self,raster):
        raster["reflectance"] = raster["Rrs"]

        result = Transparency(raster, param="reflectance")

        xr.testing.assert_identical(
            result.Rrs,
            raster["reflectance"],
        )


    def test_custom_sza_name(self,raster):
        raster["solar_angle"] = raster["sza"]

        result = Transparency(
            raster,
            sza="solar_angle",
        )

        xr.testing.assert_identical(
            result.sza,
            raster["solar_angle"],
        )


    def test_without_sza(self,raster):
        result = Transparency(raster, sza="missing")

        assert result.sza is None


# ============================================================
# set_range
# ============================================================

def test_set_range(transparency):
    values = xr.DataArray(
        [-1.0, 0.0, 1.0, 10.0, 20.0, np.nan],
        dims="pixel",
    )

    result = transparency.set_range(
        values,
        minval=0,
        maxval=10,
    )

    expected = np.array([
        np.nan,
        np.nan,
        1.0,
        np.nan,
        np.nan,
        np.nan,
    ])

    np.testing.assert_allclose(
        result.values,
        expected,
        equal_nan=True,
    )


# ============================================================
# Kd_par_RD22
# ============================================================

def test_Kd_par_RD22(transparency):
    result = transparency.Kd_par_RD22()

    R490 = transparency.Rrs.sel(wl=490)
    R665 = transparency.Rrs.sel(wl=665)

    expected = 3.09 * np.exp(-90.17 * (R490 - R665))

    xr.testing.assert_allclose(result, expected)


def test_Kd_par_RD22_custom_coefficients(transparency):
    coef = [2.0, -10.0]

    result = transparency.Kd_par_RD22(acoef=coef)

    R490 = transparency.Rrs.sel(wl=490)
    R665 = transparency.Rrs.sel(wl=665)

    expected = 2.0 * np.exp(-10.0 * (R490 - R665))

    xr.testing.assert_allclose(result, expected)


# ============================================================
# kd490_KD2
# ============================================================

class Test_kd490_KD2:
    
    def test_nominal(self,transparency):
        result = transparency.kd490_KD2()

        R490 = transparency.Rrs.sel(wl=490)
        R560 = transparency.Rrs.sel(wl=560)

        X = np.log10(R490 / R560)

        expected = (
            KD2E["Kw490"]
            + 10 ** sum(
                c * X ** i
                for i, c in enumerate(KD2E["coef"])
            )
        )

        xr.testing.assert_allclose(result, expected)


    def test_custom_coefficients(self,transparency):
        coef = [1.0, 2.0]
        wls = [443, 560]
        Kw490 = 0.02

        result = transparency.kd490_KD2(
            coef=coef,
            wls=wls,
            Kw490=Kw490,
        )

        R443 = transparency.Rrs.sel(wl=443, drop=True)
        R560 = transparency.Rrs.sel(wl=560, drop=True)

        X = np.log10(R443 / R560)

        expected = Kw490 + 10 ** (1.0 + 2.0 * X)

        xr.testing.assert_allclose(result, expected)


    def test_negative_reflectance(self,transparency):
        Rrs = transparency.Rrs.copy()

        Rrs.loc[dict(wl=490)] = -0.01

        instance = Transparency(
            xr.Dataset({"Rrs": Rrs})
        )

        result = instance.kd490_KD2()

        assert np.isnan(result.values).all()


    def test_zero_reflectance(self,transparency):
        Rrs = transparency.Rrs.copy()

        Rrs.loc[dict(wl=560)] = 0.0

        instance = Transparency(
            xr.Dataset({"Rrs": Rrs})
        )

        result = instance.kd490_KD2()

        assert np.isnan(result.values).all()


# ============================================================
# kd_par_Mo07
# ============================================================

def test_kd_par_Mo07():
    Kd490 = xr.DataArray(
        [0.1, 0.5, 1.0, 2.0, np.nan],
        dims="pixel",
    )

    result = Transparency.kd_par_Mo07(Kd490)

    expected = (
        0.0864
        + 0.884 * Kd490
        - 0.00137 / Kd490
    )

    xr.testing.assert_allclose(result, expected)


def test_kd_par_Mo07_invalid_values():
    Kd490 = xr.DataArray(
        [-1.0, 0.0, 1.0],
        dims="pixel",
    )

    result = Transparency.kd_par_Mo07(Kd490)

    assert np.isnan(result.values[:2]).all()
    assert np.isfinite(result.values[2])


# ============================================================
# _depth_lee05
# ============================================================

class TestDepthLee05:
    
    def test_valid_solution(self):
        K1 = np.array([0.1, 0.2])
        K2 = np.array([0.05, 0.1])
        tau = np.log(100)

        result = Transparency._depth_lee05(
            K1,
            K2,
            tau,
        )

        assert result.shape == (2,)
        assert np.isfinite(result).all()
        assert np.all(result > 0)


    def test_satisfies_equation(self):
        K1 = np.array([0.1, 0.2, 0.3])
        K2 = np.array([0.05, 0.1, 0.2])
        tau = np.log(100)

        result = Transparency._depth_lee05(
            K1,
            K2,
            tau,
        )

        reconstructed = (
            K1 * result
            + K2 * result / np.sqrt(1 + result)
        )

        np.testing.assert_allclose(
            reconstructed,
            tau,
            atol=1e-6,
        )


    def test_invalid_K1(self):
        K1 = np.array([0.0, -0.1, np.nan])
        K2 = np.array([0.1, 0.1, 0.1])

        result = Transparency._depth_lee05(
            K1,
            K2,
            np.log(100),
        )

        assert np.isnan(result).all()


    def test_invalid_K2(self):
        K1 = np.array([0.1, 0.2, 0.3])
        K2 = np.array([np.nan, np.inf, -np.inf])

        result = Transparency._depth_lee05(
            K1,
            K2,
            np.log(100),
        )

        assert np.isnan(result).all()


    def test_broadcasting(self):
        K1 = np.array([[0.1], [0.2]])
        K2 = np.array([0.05, 0.1])

        result = Transparency._depth_lee05(
            K1,
            K2,
            np.log(100),
        )

        assert result.shape == (2, 2)
        assert np.isfinite(result).all()


# ============================================================
# zeu_Lee05
# ============================================================

def test_zeu_Lee05():
    a490 = xr.DataArray(
        [[0.1, 0.2], [0.15, 0.25]],
        dims=("y", "x"),
    )

    bb490 = xr.DataArray(
        [[0.01, 0.02], [0.015, 0.025]],
        dims=("y", "x"),
    )

    sza = xr.DataArray(
        [[30.0, 35.0], [40.0, 45.0]],
        dims=("y", "x"),
    )

    result = Transparency.zeu_Lee05(
        a490,
        bb490,
        sza,
    )

    assert set(result.data_vars) == {"Zeu", "Kd_par"}
    assert result.Zeu.shape == (2, 2)
    assert result.Kd_par.shape == (2, 2)

    assert np.isfinite(result.Zeu.values).all()

    np.testing.assert_allclose(
        result.Kd_par,
        np.log(100) / result.Zeu,
    )


def test_zeu_Lee05_custom_percent():
    a490 = xr.DataArray(
        [[0.1]],
        dims=("y", "x"),
    )

    bb490 = xr.DataArray(
        [[0.01]],
        dims=("y", "x"),
    )

    result_1 = Transparency.zeu_Lee05(
        a490,
        bb490,
        30.0,
        percent=1,
    )

    result_10 = Transparency.zeu_Lee05(
        a490,
        bb490,
        30.0,
        percent=10,
    )

    assert result_10.Zeu.values[0, 0] < result_1.Zeu.values[0, 0]


def test_zeu_Lee05_dask():
    a490 = xr.DataArray(
        da.from_array(
            [[0.1, 0.2], [0.15, 0.25]],
            chunks=(1, 1),
        ),
        dims=("y", "x"),
    )

    bb490 = xr.DataArray(
        da.from_array(
            [[0.01, 0.02], [0.015, 0.025]],
            chunks=(1, 1),
        ),
        dims=("y", "x"),
    )

    result = Transparency.zeu_Lee05(
        a490,
        bb490,
        30.0,
    )

    assert isinstance(result.Zeu.data, da.Array)
    assert isinstance(result.Kd_par.data, da.Array)

    computed = result.compute()

    assert computed.Zeu.shape == (2, 2)


# ============================================================
# zsd_Lee15
# ============================================================

def test_zsd_Lee15_output(transparency):
    result = transparency.zsd_Lee15(
        transparency.Rrs,
        transparency.sza,
    )

    expected_variables = {
        "Zsd",
        "wl_tr",
        "Kd_tr",
        "a_below_aw",
        "Kd",
        "a",
        "bb",
    }

    assert set(result.data_vars) == expected_variables

    assert result.Zsd.shape == (2, 2)
    assert result.wl_tr.shape == (2, 2)
    assert result.Kd_tr.shape == (2, 2)

    assert result.a_below_aw.dtype == bool


def test_zsd_Lee15_transparent_wavelength(transparency):
    result = transparency.zsd_Lee15(
        transparency.Rrs,
        transparency.sza,
    )

    valid = np.isfinite(result.wl_tr.values)

    assert np.isin(
        result.wl_tr.values[valid],
        [443, 490, 560, 665],
    ).all()


def test_zsd_Lee15_bathymetry(transparency):
    bathymetry = xr.DataArray(
        np.full((2, 2), 1.0),
        dims=("y", "x"),
        coords={
            "y": [10, 20],
            "x": [30, 40],
        },
    )

    result = transparency.zsd_Lee15(
        transparency.Rrs,
        transparency.sza,
        bathymetry=bathymetry,
    )

    assert "bottom_limited" in result

    xr.testing.assert_equal(
        result.bottom_limited,
        result.Zsd > bathymetry,
    )


def test_zsd_Lee15_no_bathymetry(transparency):
    result = transparency.zsd_Lee15(
        transparency.Rrs,
        transparency.sza,
    )

    assert "bottom_limited" not in result


# ============================================================
# process
# ============================================================

class TestProcess:
    
    def test_without_sza(self,raster):
        raster = raster.drop_vars("sza")

        instance = Transparency(raster)
        instance.process()

        assert isinstance(instance.output, xr.Dataset)

        assert set(instance.output.data_vars) == {
            "Kd_par",
            "Kd490_KD2",
        }

        assert instance.flags == {}


    def test_with_sza(self,raster):
        instance = Transparency(raster)
        instance.process()

        expected_variables = {
            "Kd_par",
            "Kd490_KD2",
            "Zsd_Lee15",
            "Kd490_Lee13",
            "Kd_par_Mo07",
            "Zeu_Lee05",
            "Kd_par_Lee05",
        }

        assert set(instance.output.data_vars) == expected_variables | {
            "zsd_a_below_aw"
        } - set(instance.flags)

        assert isinstance(instance.output, xr.Dataset)

        assert instance.flags.keys() == {"zsd_a_below_aw"}


    def test_output_names(self,raster):
        instance = Transparency(raster)
        instance.process()

        assert instance.Kd_par.name == "Kd_par"
        assert instance.Kd490_KD2.name == "Kd490_KD2"
        assert instance.Zsd.name == "Zsd_Lee15"
        assert instance.Kd490.name == "Kd490_Lee13"
        assert instance.Kd_par_Mo07.name == "Kd_par_Mo07"
        assert instance.Zeu.name == "Zeu_Lee05"
        assert instance.Kd_par_Lee05.name == "Kd_par_Lee05"


    def test_metadata(self,raster):
        instance = Transparency(raster)
        instance.process()

        assert instance.Kd_par.attrs["units"] == "m-1"
        assert instance.Kd490_KD2.attrs["units"] == "m-1"
        assert instance.Zsd.attrs["units"] == "m"
        assert instance.Zeu.attrs["units"] == "m"

        assert instance.Kd_par.attrs["reference"].startswith(
            "Roy and Das"
        )

        assert instance.Kd490_KD2.attrs["range"] == [0, 50]
        assert instance.Zsd.attrs["range"] == [0, 100]


# ============================================================
# Dask compatibility
# ============================================================

def test_process_dask_lazy(raster):
    raster = raster.chunk({
        "y": 1,
        "x": 1,
        "wl": 2,
    })

    instance = Transparency(raster)
    instance.process()

    assert isinstance(instance.output["Kd_par"].data, da.Array)
    assert isinstance(instance.output["Kd490_KD2"].data, da.Array)

    computed = instance.output.compute()

    assert computed["Kd_par"].shape == (2, 2)
    assert computed["Kd490_KD2"].shape == (2, 2)


def test_dask(raster):
    Rrs = raster["Rrs"].chunk({
        "y": 1,
        "x": 1,
        "wl": 2,
    })

    instance = Transparency(
        xr.Dataset({"Rrs": Rrs})
    )

    result = instance.kd490_KD2()

    assert isinstance(result.data, da.Array)

    computed = result.compute()

    assert computed.shape == (2, 2)
