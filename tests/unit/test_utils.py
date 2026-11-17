import numpy as np
import pandas as pd
import pytest
import xarray as xr

from unittest.mock import patch

import GRSl2bgen.utils as utils


# ============================================================
# Fixtures and helpers
# ============================================================

@pytest.fixture
def water_iops():
    """Create an iopw instance with synthetic optical properties."""

    obj = utils.iopw.__new__(utils.iopw)

    wl = np.array([400, 443, 490, 560, 665, 700])

    obj.iop_w = xr.Dataset(
        {
            "a": ("wl", [0.01, 0.02, 0.03, 0.05, 0.4, 0.5]),
            "b": ("wl", [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        },
        coords={"wl": wl},
    )

    obj.water_abs_corr = xr.Dataset(
        {
            "PsiT": ("wl", [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]),
            "PsiS_ASW": ("wl", [0.0001, 0.0002, 0.0003,
                                0.0004, 0.0005, 0.0006]),
        },
        coords={"wl": wl},
    )

    return obj


@pytest.fixture
def wavelengths():
    return np.array([443, 490, 560, 665])


@pytest.fixture
def synthetic_rrs(wavelengths):
    """Create synthetic above-water remote sensing reflectance."""

    return xr.DataArray(
        [0.01, 0.012, 0.015, 0.001],
        dims=["wl"],
        coords={"wl": wavelengths},
        name="Rrs",
    )


# ============================================================
# Tests for iopw
# ============================================================

class TestIOPW:

    def test_get_iopw_returns_dataarrays(self, water_iops):
        """Check that absorption and backscattering are DataArrays."""

        aw, bbw = water_iops.get_iopw([443, 490, 560, 665])

        assert isinstance(aw, xr.DataArray)
        assert isinstance(bbw, xr.DataArray)

    def test_get_iopw_output_names(self, water_iops):
        """Check output variable names."""

        aw, bbw = water_iops.get_iopw([443, 490, 560, 665])

        assert aw.name == "aw"
        assert bbw.name == "bbw"

    def test_get_iopw_wavelengths(self, water_iops):
        """Check that requested wavelengths are returned."""

        wl = np.array([443, 490, 560, 665])

        aw, bbw = water_iops.get_iopw(wl)

        np.testing.assert_array_equal(aw.wl.values, wl)
        np.testing.assert_array_equal(bbw.wl.values, wl)

    def test_get_iopw_backscattering_is_half_scattering(
        self, water_iops
    ):
        """Backscattering should equal half the scattering coefficient."""

        wl = [443, 490, 560, 665]

        _, bbw = water_iops.get_iopw(wl)

        expected = water_iops.iop_w.b.interp(wl=wl) / 2

        np.testing.assert_allclose(
            bbw.values,
            expected.values,
        )

    def test_get_iopw_reference_temperature(self, water_iops):
        """At 22 degrees C, temperature correction should be zero."""

        wl = [443, 490, 560, 665]

        aw, _ = water_iops.get_iopw(
            wl,
            Twater=22,
            Salinity=0,
        )

        expected = water_iops.iop_w.a.interp(wl=wl)

        np.testing.assert_allclose(
            aw.values,
            expected.values,
        )

    def test_get_iopw_temperature_correction(self, water_iops):
        """Check temperature correction against the analytical formula."""

        wl = [443, 490, 560, 665]
        temperature = 25

        aw, _ = water_iops.get_iopw(
            wl,
            Twater=temperature,
            Salinity=0,
        )

        expected = (
            water_iops.iop_w.a.interp(wl=wl)
            + (temperature - 22)
            * water_iops.water_abs_corr.PsiT.interp(wl=wl).fillna(0)
        )

        np.testing.assert_allclose(
            aw.values,
            expected.values,
        )

    def test_get_iopw_salinity_correction(self, water_iops):
        """Check salinity correction against the analytical formula."""

        wl = [443, 490, 560, 665]
        salinity = 35

        aw, _ = water_iops.get_iopw(
            wl,
            Twater=22,
            Salinity=salinity,
        )

        expected = (
            water_iops.iop_w.a.interp(wl=wl)
            + salinity
            * water_iops.water_abs_corr.PsiS_ASW.interp(wl=wl)
        )

        np.testing.assert_allclose(
            aw.values,
            expected.values,
        )

    def test_get_iopw_combined_corrections(self, water_iops):
        """Check simultaneous temperature and salinity corrections."""

        wl = [443, 490, 560, 665]
        temperature = 25
        salinity = 35

        aw, _ = water_iops.get_iopw(
            wl,
            Twater=temperature,
            Salinity=salinity,
        )

        expected = (
            water_iops.iop_w.a.interp(wl=wl)
            + (temperature - 22)
            * water_iops.water_abs_corr.PsiT.interp(wl=wl).fillna(0)
            + salinity
            * water_iops.water_abs_corr.PsiS_ASW.interp(wl=wl).fillna(0)
        )

        np.testing.assert_allclose(
            aw.values,
            expected.values,
        )

    def test_get_iopw_scalar_wavelength(self, water_iops):
        """Check that scalar wavelengths are supported."""

        aw, bbw = water_iops.get_iopw(560)

        assert aw.size == 1
        assert bbw.size == 1

    def test_get_iopw_custom_interpolation_method(self, water_iops):
        """Check that interpolation method is forwarded."""

        wl = [443, 490, 560]

        aw, bbw = water_iops.get_iopw(
            wl,
            method="nearest",
        )

        assert aw.size == 3
        assert bbw.size == 3

    def test_get_iopw_mute(self, water_iops):
        """With mute=True, values should be stored but not returned."""

        result = water_iops.get_iopw(
            [443, 490, 560],
            mute=True,
        )

        assert result is None
        assert hasattr(water_iops, "aw")
        assert hasattr(water_iops, "bbw")

    def test_get_iopw_stores_requested_wavelengths(self, water_iops):
        """Check that requested wavelengths are stored in the instance."""

        wl = [443, 490, 560]

        water_iops.get_iopw(wl)

        np.testing.assert_array_equal(water_iops.wl, wl)

    def test_get_iopw_outside_wavelength_range(self, water_iops):
        """Check interpolation behavior outside the data range."""

        aw, bbw = water_iops.get_iopw(
            [350, 750],
        )

        assert np.isnan(aw.values).all()

        # Backscattering uses fill_value=0 outside the data range.
        np.testing.assert_allclose(bbw.values, [0, 0])


# ============================================================
# Tests for aphi_bricaud1998
# ============================================================

class TestAphiBricaud1998:

    @pytest.fixture
    def bricaud_table(self):
        """Synthetic Bricaud coefficient table."""

        return pd.DataFrame(
            {
                "wl": [400, 500, 600, 700],
                "Ap": [1, 2, 3, 4],
                "Ep": [0.1, 0.2, 0.3, 0.4],
                "Aphi": [0.01, 0.02, 0.03, 0.04],
                "Ephi": [0.6, 0.7, 0.8, 0.9],
            }
        )

    def test_returns_two_dataarrays(self, bricaud_table):
        """Check output types."""

        with patch(
            "GRSl2bgen.utils.pd.read_csv",
            return_value=bricaud_table.copy(),
        ):
            Aphi, Ephi = utils.aphi_bricaud1998([450, 550])

        assert isinstance(Aphi, xr.DataArray)
        assert isinstance(Ephi, xr.DataArray)

    def test_output_names(self, bricaud_table):
        """Check output variable names."""

        with patch(
            "GRSl2bgen.utils.pd.read_csv",
            return_value=bricaud_table.copy(),
        ):
            Aphi, Ephi = utils.aphi_bricaud1998([450, 550])

        assert Aphi.name == "Aphi"
        assert Ephi.name == "Ephi"

    def test_interpolation(self, bricaud_table):
        """Check linear interpolation of Aphi and Ephi."""

        with patch(
            "GRSl2bgen.utils.pd.read_csv",
            return_value=bricaud_table.copy(),
        ):
            Aphi, Ephi = utils.aphi_bricaud1998([450, 550])

        np.testing.assert_allclose(
            Aphi.values,
            [0.015, 0.025],
        )

        np.testing.assert_allclose(
            Ephi.values,
            [0.65, 0.75],
        )

    def test_exact_wavelengths(self, bricaud_table):
        """Check that original tabulated values are preserved."""

        with patch(
            "GRSl2bgen.utils.pd.read_csv",
            return_value=bricaud_table.copy(),
        ):
            Aphi, Ephi = utils.aphi_bricaud1998([400, 600])

        np.testing.assert_allclose(
            Aphi.values,
            [0.01, 0.03],
        )

        np.testing.assert_allclose(
            Ephi.values,
            [0.6, 0.8],
        )

    def test_scalar_wavelength(self, bricaud_table):
        """Check support for scalar wavelength input."""

        with patch(
            "GRSl2bgen.utils.pd.read_csv",
            return_value=bricaud_table.copy(),
        ):
            Aphi, Ephi = utils.aphi_bricaud1998(500)

        assert Aphi.size == 1
        assert Ephi.size == 1

    def test_custom_interpolation_method(self, bricaud_table):
        """Check nearest-neighbor interpolation."""

        with patch(
            "GRSl2bgen.utils.pd.read_csv",
            return_value=bricaud_table.copy(),
        ):
            Aphi, Ephi = utils.aphi_bricaud1998(
                [450, 550],
                method="nearest",
            )

        np.testing.assert_allclose(
            Aphi.values,
            [0.01, 0.02],
        )

        np.testing.assert_allclose(
            Ephi.values,
            [0.6, 0.7],
        )

    def test_outside_wavelength_range(self, bricaud_table):
        """Check behavior outside the tabulated wavelength range."""

        with patch(
            "GRSl2bgen.utils.pd.read_csv",
            return_value=bricaud_table.copy(),
        ):
            Aphi, Ephi = utils.aphi_bricaud1998([350, 750])

        assert np.isnan(Aphi.values).all()
        assert np.isnan(Ephi.values).all()


# ============================================================
# Tests for qaa_v6_iops
# ============================================================

class TestQAAV6:

    @pytest.fixture
    def mock_water_iops(self, monkeypatch):
        """
        Replace iopw with a deterministic implementation.

        This isolates the QAA calculations from the water
        optical property data files.
        """

        class MockIOPW:

            def get_iopw(
                self,
                wl,
                Twater=20,
                Salinity=0,
            ):
                wl = np.asarray(wl)

                aw = xr.DataArray(
                    np.array([0.006, 0.01, 0.06, 0.43]),
                    dims=["wl"],
                    coords={"wl": wl},
                    name="aw",
                )

                bbw = xr.DataArray(
                    np.array([0.002, 0.002, 0.0015, 0.001]),
                    dims=["wl"],
                    coords={"wl": wl},
                    name="bbw",
                )

                return aw, bbw

        monkeypatch.setattr(utils, "iopw", MockIOPW)

    def test_returns_xarray_dataset(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check output type."""

        result = utils.qaa_v6_iops(synthetic_rrs)

        assert isinstance(result, xr.Dataset)

    def test_output_variables(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check that all expected variables are returned."""

        result = utils.qaa_v6_iops(synthetic_rrs)

        expected = {
            "Rrs",
            "rrs",
            "u",
            "a",
            "bbp",
            "aw",
            "bbw",
        }

        assert set(result.data_vars) == expected

    def test_output_wavelengths(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check that the requested wavelength bands are returned."""

        result = utils.qaa_v6_iops(synthetic_rrs)

        np.testing.assert_array_equal(
            result.wl.values,
            [443, 490, 560, 665],
        )

    def test_below_water_reflectance(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check conversion from above-water to below-water Rrs."""

        result = utils.qaa_v6_iops(synthetic_rrs)

        expected = (
            synthetic_rrs / (0.52 + 1.7 * synthetic_rrs)
        )

        np.testing.assert_allclose(
            result.rrs.values,
            expected.values,
        )

    def test_u_calculation(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check the analytical calculation of u."""

        g0 = 0.089
        g1 = 0.1245

        result = utils.qaa_v6_iops(
            synthetic_rrs,
            g0=g0,
            g1=g1,
        )

        rrs = result.rrs.values

        expected = (
            -g0 + np.sqrt(g0**2 + 4 * g1 * rrs)
        ) / (2 * g1)

        np.testing.assert_allclose(
            result.u.values,
            expected,
        )

    def test_custom_g_coefficients(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check that custom g0 and g1 values are accepted."""

        result = utils.qaa_v6_iops(
            synthetic_rrs,
            g0=0.1,
            g1=0.15,
        )

        assert np.isfinite(result.u.values).all()

    def test_green_reference_wavelength(
        self,
        mock_water_iops,
    ):
        """
        Check that 560 nm is selected when Rrs665 < 0.0015.
        """

        Rrs = xr.DataArray(
            [0.01, 0.012, 0.015, 0.001],
            dims=["wl"],
            coords={"wl": [443, 490, 560, 665]},
        )

        result = utils.qaa_v6_iops(Rrs)

        # Green reference: bbp is referenced to 560 nm.
        assert np.isfinite(result.bbp.values).all()

        # The bbp spectral extrapolation should use 560 nm.
        # Verify the analytical relationship.
        wl = result.wl.values
        eta = 2 * (
            1 - 1.2 * np.exp(
                -0.9 * result.rrs.sel(wl=443).values
                / result.rrs.sel(wl=560).values
            )
        )

        bbp560 = result.bbp.sel(wl=560).values

        expected = bbp560 * (560 / wl) ** eta

        np.testing.assert_allclose(
            result.bbp.values,
            expected,
        )

    def test_red_reference_wavelength(
        self,
        mock_water_iops,
    ):
        """
        Check that 665 nm is selected when Rrs665 >= 0.0015.
        """

        Rrs = xr.DataArray(
            [0.01, 0.012, 0.015, 0.002],
            dims=["wl"],
            coords={"wl": [443, 490, 560, 665]},
        )

        result = utils.qaa_v6_iops(Rrs)

        wl = result.wl.values

        eta = 2 * (
            1 - 1.2 * np.exp(
                -0.9 * result.rrs.sel(wl=443).values
                / result.rrs.sel(wl=560).values
            )
        )

        bbp665 = result.bbp.sel(wl=665).values

        expected = bbp665 * (665 / wl) ** eta

        np.testing.assert_allclose(
            result.bbp.values,
            expected,
        )

    def test_zero_reflectance_is_masked(
        self,
        mock_water_iops,
    ):
        """Non-positive reflectance should be masked."""

        Rrs = xr.DataArray(
            [0.01, 0.012, 0.015, 0.0],
            dims=["wl"],
            coords={"wl": [443, 490, 560, 665]},
        )

        result = utils.qaa_v6_iops(Rrs)

        assert np.isnan(
            result.Rrs.sel(wl=665).values
        )

    def test_negative_reflectance_is_masked(
        self,
        mock_water_iops,
    ):
        """Negative reflectance should be masked."""

        Rrs = xr.DataArray(
            [0.01, 0.012, 0.015, -0.001],
            dims=["wl"],
            coords={"wl": [443, 490, 560, 665]},
        )

        result = utils.qaa_v6_iops(Rrs)

        assert np.isnan(
            result.Rrs.sel(wl=665).values
        )

    def test_total_absorption_is_finite(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check that absorption is finite for valid reflectances."""

        result = utils.qaa_v6_iops(synthetic_rrs)

        assert np.isfinite(result.a.values).all()

    def test_bbp_is_finite(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check that particulate backscattering is finite."""

        result = utils.qaa_v6_iops(synthetic_rrs)

        assert np.isfinite(result.bbp.values).all()

    def test_water_properties_are_preserved(
        self,
        synthetic_rrs,
        mock_water_iops,
    ):
        """Check that pure-water properties are included in the output."""

        result = utils.qaa_v6_iops(synthetic_rrs)

        np.testing.assert_allclose(
            result.aw.values,
            [0.006, 0.01, 0.06, 0.43],
        )

        np.testing.assert_allclose(
            result.bbw.values,
            [0.002, 0.002, 0.0015, 0.001],
        )

    def test_temperature_and_salinity_arguments(
        self,
        synthetic_rrs,
        monkeypatch,
    ):
        """Check that temperature and salinity are forwarded to iopw."""

        captured = {}

        class MockIOPW:

            def get_iopw(
                self,
                wl,
                Twater=20,
                Salinity=0,
            ):
                captured["Twater"] = Twater
                captured["Salinity"] = Salinity

                wl = np.asarray(wl)

                aw = xr.DataArray(
                    [0.006, 0.01, 0.06, 0.43],
                    dims=["wl"],
                    coords={"wl": wl},
                )

                bbw = xr.DataArray(
                    [0.002, 0.002, 0.0015, 0.001],
                    dims=["wl"],
                    coords={"wl": wl},
                )

                return aw, bbw

        monkeypatch.setattr(utils, "iopw", MockIOPW)

        utils.qaa_v6_iops(
            synthetic_rrs,
            Twater=25,
            Salinity=35,
        )

        assert captured["Twater"] == 25
        assert captured["Salinity"] == 35

    def test_custom_wavelengths(
        self,
        mock_water_iops,
    ):
        """Check selection using custom requested wavelengths."""

        wl = [443, 490, 560, 665]

        Rrs = xr.DataArray(
            [0.01, 0.012, 0.015, 0.001],
            dims=["wl"],
            coords={"wl": wl},
        )

        result = utils.qaa_v6_iops(
            Rrs,
            wls=wl,
        )

        np.testing.assert_array_equal(
            result.wl.values,
            wl,
        )

    def test_nearest_wavelength_selection(
        self,
        mock_water_iops,
    ):
        """Check that the selection method is applied."""

        wl = [442, 489, 559, 664]

        Rrs = xr.DataArray(
            [0.01, 0.012, 0.015, 0.001],
            dims=["wl"],
            coords={"wl": wl},
        )

        result = utils.qaa_v6_iops(
            Rrs,
            method="nearest",
        )

        assert len(result.wl) == 4
