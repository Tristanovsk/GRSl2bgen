from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import xarray as xr
from GRSl2bgen.cdom import Cdom
import GRSl2bgen.cdom as cdom_module

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def cdom_test():
    raster = xr.Dataset(
        data_vars={
            "Rrs": (
                ("wl", "y", "x"),
                np.array([
                    [[0.1, 0.5],[1.0,0.2]],
                    [[0.0, 1.0], [-1.0,0.3]],
                ])
            ),
        },
        coords={
            "wl": [550, 680],
            "y": [0,1],
            "x": [0,1]
        }
    )
    chl = xr.DataArray(
        np.array([
            [0.0, np.nan, 50.0],
            [10.0,-1.0,100.0]
        ]),
        dims=("y", "x"),
        coords={
            "y": [0.0, 1.0],
            "x": [0.0, 1.0, 2.0],
        },
        name="Chl-a",
    )
    
    cdom= Cdom.__new__(Cdom)
    cdom.raster = raster
    cdom.Rrs = raster["Rrs"]
    cdom.chl = chl
    return cdom

@pytest.fixture
def mock_qaa_iops(monkeypatch):

    wl = [443, 490, 560, 665]

    coords = {
        "wl": wl,
        "y": [0, 1],
        "x": [0, 1],
    }

    # Total absorption
    a = xr.DataArray(
        np.array([
            [[0.4, 0.4], [0.4, 0.4]],
            [[0.399, 0.399], [0.399, 0.399]],
            [[0.3, 0.3], [0.3, 0.3]],
            [[0.2, 0.2], [0.2, 0.2]],
        ]),
        dims=("wl", "y", "x"),
        coords=coords,
    )

    # Pure water absorption
    aw = xr.DataArray(
        [0.1, 0.1, 0.05, 0.4],
        dims="wl",
        coords={"wl": wl},
    )

    # Backscattering
    bbp = xr.DataArray(
        np.ones((4, 2, 2)) * 0.01,
        dims=("wl", "y", "x"),
        coords=coords,
    )

    # Below-water reflectance
    rrs = xr.DataArray(
        np.ones((4, 2, 2)) * 0.01,
        dims=("wl", "y", "x"),
        coords=coords,
    )

    iops = xr.Dataset({
        "a": a,
        "aw": aw,
        "bbp": bbp,
        "rrs": rrs,
    })

    def fake_qaa_v6_iops(*args, **kwargs):
        return iops

    monkeypatch.setattr(
        cdom_module,
        "qaa_v6_iops",
        fake_qaa_v6_iops,
    )

    return iops

@pytest.fixture
def mock_bricaud(monkeypatch):

    def fake_aphi_bricaud1998(wl, method="linear"):

        wl = np.asarray(wl)

        Aphi = xr.DataArray(
            np.ones(wl.shape),
            dims=("wl",) if wl.ndim else (),
            coords={"wl": wl} if wl.ndim else None,
        )

        Ephi = xr.DataArray(
            np.ones(wl.shape),
            dims=("wl",) if wl.ndim else (),
            coords={"wl": wl} if wl.ndim else None,
        )

        return Aphi, Ephi

    monkeypatch.setattr(
        cdom_module,
        "aphi_bricaud1998",
        fake_aphi_bricaud1998,
    )

@pytest.fixture
def mock_qaa_iops_chl(monkeypatch):

    wl = [443, 490, 560, 665]

    coords = {
        "wl": wl,
        "y": [0, 1],
        "x": [0, 1],
    }

    # Total absorption
    a = xr.DataArray(
        np.array([
            [[100.0, 100.0], [100.0, 100.0]],  # 443
            [[0.399, 0.399], [0.399, 0.399]],  # 490
            [[0.3, 0.3], [0.3, 0.3]],          # 560
            [[0.2, 0.2], [0.2, 0.2]],          # 665
        ]),
        dims=("wl", "y", "x"),
        coords=coords,
    )

    # Pure water absorption
    aw = xr.DataArray(
        [0.1, 0.1, 0.05, 0.4],
        dims="wl",
        coords={"wl": wl},
    )

    # Backscattering
    bbp = xr.DataArray(
        np.ones((4, 2, 2)) * 0.01,
        dims=("wl", "y", "x"),
        coords=coords,
    )

    # Below-water reflectance
    rrs = xr.DataArray(
        np.ones((4, 2, 2)) * 0.01,
        dims=("wl", "y", "x"),
        coords=coords,
    )

    iops = xr.Dataset({
        "a": a,
        "aw": aw,
        "bbp": bbp,
        "rrs": rrs,
    })

    def fake_qaa_v6_iops(*args, **kwargs):
        return iops

    monkeypatch.setattr(
        cdom_module,
        "qaa_v6_iops",
        fake_qaa_v6_iops,
    )

    return iops

@pytest.fixture
def mock_cdom_products(monkeypatch):

    def fake_brezonik15(self, acoef):
        assert acoef == [1.872, -0.83]

        return xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=("y", "x"),
            coords={"y": [0, 1], "x": [0, 1]},
        )

    def fake_a_dg443_QAA_v6(self, Rrs, chl):
        return xr.Dataset({
            "adg443": xr.DataArray(
                [[0.5, 1.0], [1.5, 2.0]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]},
            )
        })

    def fake_a_dg443_QAA_v6_chl(self, Rrs, chl):
        return xr.Dataset({
            "adg443": xr.DataArray(
                [[0.2, 0.4], [0.6, 0.8]],
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]},
            )
        })

    monkeypatch.setattr(Cdom, "brezonik15", fake_brezonik15)
    monkeypatch.setattr(Cdom, "a_dg443_QAA_v6", fake_a_dg443_QAA_v6)
    monkeypatch.setattr(Cdom, "a_dg443_QAA_v6_chl", fake_a_dg443_QAA_v6_chl)
    
# ============================================================
# set_range
# ============================================================

def test_set_range():
    cdom = Cdom.__new__(Cdom)
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
    
    result = cdom.set_range(param=param, minval=minval, maxval=maxval)
    expected = np.array([
        [np.nan, np.nan, np.nan, 50.0],
        [0.01,99.9,np.nan, np.nan]
    ])
    
    np.testing.assert_array_equal(result, expected)

# ============================================================
# brezonik15
# ============================================================

def test_brezonik15(cdom_test):
    cdom = cdom_test
    
    expected = np.array(
        np.array([
            [np.nan, 11.557],
            [np.nan,9.102]
        ]),
    )
    
    result = cdom.brezonik15()
    
    assert result.shape == (2,2)
    np.testing.assert_allclose(result, expected, atol=1e-3)

# ============================================================
# a_dg443_QAA_v6
# ============================================================

class Test_a_dg443_QAA_v6:
    
    def test_output_and_decomposition(
        self,
        cdom_test,
        mock_qaa_iops,
        mock_bricaud,
    ):

        chl = 1.0

        result = cdom_test.a_dg443_QAA_v6(
            Rrs=cdom_test.Rrs,
            chl=chl,
        )

        iops = mock_qaa_iops

        a443 = iops.a.sel(wl=443)
        a490 = iops.a.sel(wl=490)

        aw443 = iops.aw.sel(wl=443)
        aw490 = iops.aw.sel(wl=490)

        rrs443 = iops.rrs.sel(wl=443)
        rrs560 = iops.rrs.sel(wl=560)

        # Bricaud coefficients
        zeta = 1.0

        S = 0.015 + 0.002 / (0.6 + rrs443 / rrs560)
        xi = np.exp(S * (490 - 443))

        expected_adg = (
            (a443 - aw443) - zeta * (a490 - aw490)
        ) / (1 - zeta / xi)

        expected_aph = (a443 - aw443) - expected_adg

        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {
            "adg443",
            "aph443",
            "bbp",
        }
        assert result.adg443.dims == ("y", "x")
        assert result.aph443.dims == ("y", "x")
        assert result.bbp.dims == ("wl", "y", "x")
        assert np.all(result.bbp.values == mock_qaa_iops.bbp.values)
        np.testing.assert_allclose(
            result.adg443.values,
            expected_adg.values,
        )

        np.testing.assert_allclose(
            result.aph443.values,
            expected_aph.values,
        )


    @pytest.mark.parametrize(
        "chl",
        [
            0.0,
            -1.0,
            np.nan,
        ],
    )
    def test_invalid_chl(
        self,
        cdom_test,
        mock_qaa_iops,
        mock_bricaud,
        chl,
    ):
        chl = xr.full_like(
            mock_qaa_iops.a.isel(wl=0, drop=True),
            chl,
        )
        result = cdom_test.a_dg443_QAA_v6(
            Rrs=cdom_test.Rrs,
            chl=chl,
        )
        assert result.adg443.isnull().all()
        assert result.aph443.isnull().all()

# ============================================================
# a_dg443_QAA_v6_chl
# ============================================================

class Test_a_dg433_QAA_v6_chl:

    def test_output_and_decomposition(
        self,
        cdom_test,
        mock_qaa_iops_chl,
        mock_bricaud,
    ):

        chl = 1.0

        result = cdom_test.a_dg443_QAA_v6_chl(
            Rrs=cdom_test.Rrs,
            chl=chl,
        )

        iops = mock_qaa_iops_chl

        a443 = iops.a.sel(wl=443)
        aw443 = iops.aw.sel(wl=443)

        expected_aph = 1.0 * chl ** 1.0
        expected_adg = a443 - aw443 - expected_aph

        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {
            "adg443",
            "aph443",
        }
        assert result.adg443.dims == ("y", "x")
        assert result.aph443.dims == ("y", "x")
        np.testing.assert_allclose(
            result.aph443.values,
            expected_aph,
        )

        np.testing.assert_allclose(
            result.adg443.values,
            expected_adg.values,
        )

    @pytest.mark.parametrize(
        "chl",
        [0.1, 1.0, 10.0, 50.0],
    )
    def test_concentrations(
        self,
        cdom_test,
        mock_qaa_iops_chl,
        mock_bricaud,
        chl,
    ):

        result = cdom_test.a_dg443_QAA_v6_chl(
            Rrs=cdom_test.Rrs,
            chl=chl,
        )

        expected_aph = chl  # Aphi=1 and Ephi=1

        np.testing.assert_allclose(
            result.aph443.values,
            expected_aph,
        )

    def test_spatial_mask(
        self,
        cdom_test,
        mock_qaa_iops_chl,
        mock_bricaud,
    ):

        chl = xr.DataArray(
            [
                [1.0, 0.0],
                [-1.0, np.nan],
            ],
            dims=("y", "x"),
            coords={
                "y": [0, 1],
                "x": [0, 1],
            },
        )

        result = cdom_test.a_dg443_QAA_v6_chl(
            Rrs=cdom_test.Rrs,
            chl=chl,
        )

        invalid = chl.isnull() | (chl <= 0)

        for var in ["adg443", "aph443"]:
            assert result[var].where(invalid).isnull().all()

        # Valid chlorophyll pixels should have finite absorption.
        assert np.isfinite(result.aph443.sel(y=0, x=0))
        assert np.isfinite(result.adg443.sel(y=0, x=0))

    def test_scalar_input(
        self,
        cdom_test,
        mock_qaa_iops_chl,
        mock_bricaud,
    ):

        result = cdom_test.a_dg443_QAA_v6_chl(
            Rrs=cdom_test.Rrs,
            chl=1.0,
        )

        assert result.adg443.dims == ("y", "x")
        assert result.aph443.dims == ("y", "x")


    def test_array_input(
        self,
        cdom_test,
        mock_qaa_iops_chl,
        mock_bricaud,
    ):

        chl = xr.ones_like(
            mock_qaa_iops_chl.a.isel(wl=0, drop=True)
        )

        result = cdom_test.a_dg443_QAA_v6_chl(
            Rrs=cdom_test.Rrs,
            chl=chl,
        )

        assert result.adg443.dims == ("y", "x")
        assert result.aph443.dims == ("y", "x")

# ============================================================
# process
# ============================================================

class TestProcess:

    def test_without_chl(self,cdom_test,mock_cdom_products):

        cdom_test.chl = None

        cdom_test.process()

        assert isinstance(cdom_test.output, xr.Dataset)

        assert list(cdom_test.output.data_vars) == [
            "acdom_B15",
        ]

        assert cdom_test.output.acdom_B15.name == "acdom_B15"

        np.testing.assert_allclose(
            cdom_test.output.acdom_B15.values,
            [[1.0, 2.0], [3.0, 4.0]],
        )

        assert cdom_test.output.acdom_B15.attrs["units"] == "m-1"
        assert cdom_test.output.acdom_B15.attrs["range"] == [0, 30]
        assert cdom_test.output.acdom_B15.attrs["coef"] == str([1.872, -0.83])

    def test_with_chl(self,cdom_test,mock_cdom_products):

        cdom_test.process()

        assert isinstance(cdom_test.output, xr.Dataset)

        assert set(cdom_test.output.data_vars) == {
            "acdom_B15",
            "adg443_QAAv6",
            "adg443_QAAv6Chl",
        }

        np.testing.assert_allclose(
            cdom_test.output.acdom_B15.values,
            [[1.0, 2.0], [3.0, 4.0]],
        )

        np.testing.assert_allclose(
            cdom_test.output.adg443_QAAv6.values,
            [[0.5, 1.0], [1.5, 2.0]],
        )

        np.testing.assert_allclose(
            cdom_test.output.adg443_QAAv6Chl.values,
            [[0.2, 0.4], [0.6, 0.8]],
        )

    def test_metadata(self,cdom_test,mock_cdom_products):

        cdom_test.process()

        output = cdom_test.output

        for var in [
            "acdom_B15",
            "adg443_QAAv6",
            "adg443_QAAv6Chl",
        ]:
            assert output[var].attrs["units"] == "m-1"
            assert output[var].attrs["range"] == [0, 30]
            assert output[var].attrs["reference"]
            assert output[var].attrs["description"]

        assert output.acdom_B15.attrs["appicability"] == "general"

        assert output.adg443_QAAv6.attrs["applicability"] == "general"
        assert output.adg443_QAAv6Chl.attrs["applicability"] == "general"

        assert "Chl-a" in output.adg443_QAAv6.attrs["description"]
        assert "Chl-a" in output.adg443_QAAv6Chl.attrs["description"]
