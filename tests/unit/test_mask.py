from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import xarray as xr
import dask.array as da
import pandas as pd

from GRSl2bgen.mask import l2b_flags, QC_L2A, Masking

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_l2b_flags(monkeypatch):
    flags = {
        "invalid_reflectance": (0, "Invalid reflectance"),
        "cloud": (1, "Cloud detected"),
        "high_turbidity": (2, "High turbidity"),
    }

    monkeypatch.setitem(
        l2b_flags.__globals__,
        "L2B_FLAGS",
        flags,
    )

    monkeypatch.setitem(
        l2b_flags.__globals__,
        "FLAG_NAME_L2B",
        "flags_l2b",
    )

    return flags


@pytest.fixture
def flag_template():
    return xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        coords={
            "y": [10, 20],
            "x": [30, 40],
        },
    )

FLAG_NAME = 'flags'
@pytest.fixture
def masking_test():
    # Three flags:
    # bit 0: cloud
    # bit 1: land
    # bit 2: invalid

    flags = xr.DataArray(
        np.array([
            [1, 3, 0],
            [4, 5, 7],
        ], dtype=np.uint32),
        dims=("y", "x"),
        coords={
            "y": [0, 1],
            "x": [0, 1, 2],
        },
        name=FLAG_NAME,
        attrs={
            "flag_names": [
                "cloud",
                "land",
                "invalid",
                "None",
            ],
            "flag_descriptions": [
                "Cloud pixels",
                "Land pixels",
                "Invalid pixels",
                "None",
            ],
        },
    )

    prod = xr.Dataset({
        FLAG_NAME: flags,
    })

    return Masking(prod)

@pytest.fixture
def masking_process_test():

    prod = xr.Dataset(
        data_vars={
            "Rrs": (
                ("y", "x"),
                np.ones((2, 3)),
            ),
        },
        coords={
            "y": [0, 1],
            "x": [0, 1, 2],
        },
    )

    masking = Masking(prod)

    masking.flags = np.array([
        [1, 3, 0],
        [4, 5, 7],
    ], dtype=np.uint32)

    masking.flag_names[:4] = [
        "cloud",
        "land",
        "invalid",
        "None",
    ]

    masking.flag_descriptions[:4] = [
        "Cloud pixels",
        "Land pixels",
        "Invalid pixels",
        "None",
    ]

    return masking

@pytest.fixture
def flags_test():

    flags = xr.DataArray(
        np.array([
            [0, 1, 2],
            [3, 4, 7],
        ], dtype=np.uint32),
        dims=("y", "x"),
        coords={
            "y": [0, 1],
            "x": [0, 1, 2],
        },
        name="flags_l2b",
    )

    return flags

# ============================================================
# Tests for l2b_flags
# ============================================================

class TestL2BFlags:
    
    def test_output(self,flag_template, mock_l2b_flags):
        conditions = {
            "invalid_reflectance": xr.DataArray(
                [[True, False],
                [False, False]],
                dims=("y", "x"),
            ),
            "cloud": xr.DataArray(
                [[False, True],
                [False, True]],
                dims=("y", "x"),
            ),
            "high_turbidity": xr.DataArray(
                [[False, False],
                [True, True]],
                dims=("y", "x"),
            ),
        }

        result = l2b_flags(
            flag_template,
            conditions,
        )

        expected = np.array([
            [1, 2],
            [4, 6],
        ], dtype=np.uint32)

        np.testing.assert_array_equal(result.values, expected)

        assert isinstance(result, xr.DataArray)
        assert result.dtype == np.uint32
        assert result.dims == ("y", "x")
        assert result.name == "flags_l2b"
        
    def test_no_flags(self,flag_template, mock_l2b_flags):
        conditions = {
            "invalid_reflectance": xr.zeros_like(
                flag_template, dtype=bool
            ),
            "cloud": xr.zeros_like(
                flag_template, dtype=bool
            ),
        }

        result = l2b_flags(
            flag_template,
            conditions,
        )

        np.testing.assert_array_equal(
            result.values,
            np.zeros((2, 2), dtype=np.uint32),
        )


    @pytest.mark.parametrize(
        "flag_name, expected_value",
        [
            ("invalid_reflectance", 1),
            ("cloud", 2),
            ("high_turbidity", 4),
        ],
    )
    def test_single_flag(self,flag_template,mock_l2b_flags,flag_name,expected_value):
        conditions = {
            flag_name: xr.ones_like(flag_template, dtype=bool)
        }

        result = l2b_flags(
            flag_template,
            conditions,
        )

        np.testing.assert_array_equal(
            result.values,
            np.full((2, 2), expected_value, dtype=np.uint32),
        )

    def test_metadata(self,flag_template, mock_l2b_flags):
        conditions = {}

        result = l2b_flags(
            flag_template,
            conditions,
        )

        assert result.attrs["long_name"] == (
            "flags computed by the L2B processing"
        )

        names = result.attrs["flag_names"]
        descriptions = result.attrs["flag_descriptions"]

        assert names[0] == "invalid_reflectance"
        assert names[1] == "cloud"
        assert names[2] == "high_turbidity"

        assert descriptions[0] == "Invalid reflectance"
        assert descriptions[1] == "Cloud detected"
        assert descriptions[2] == "High turbidity"

    def test_custom_descriptions(self,flag_template,mock_l2b_flags):
        conditions = {
            "cloud": xr.DataArray(
                [[True, False],
                [False, False]],
                dims=("y", "x"),
            ),
        }

        descriptions = {
            "cloud": "Cloud detected with threshold = 0.2",
        }

        result = l2b_flags(
            flag_template,
            conditions,
            descriptions=descriptions,
        )

        assert result.attrs["flag_names"][1] == "cloud"

        assert result.attrs["flag_descriptions"][1] == (
            "Cloud detected with threshold = 0.2"
        )

        # The bit position must remain unchanged.
        assert result.values[0, 0] == 2

    def test_number_of_flags(self,flag_template,mock_l2b_flags):
        result = l2b_flags(
            flag_template,
            conditions={},
            number_of_flags=8,
        )

        assert len(result.attrs["flag_names"]) == 8
        assert len(result.attrs["flag_descriptions"]) == 8

    def test_unknown_flag(self,flag_template,mock_l2b_flags):
        conditions = {
            "unknown_flag": xr.ones_like(
                flag_template,
                dtype=bool,
            )
        }

        with pytest.raises(KeyError):
            l2b_flags(
                flag_template,
                conditions,
            )


# ============================================================
# Tests for QC_L2A process
# ============================================================

def test_process():
    qc = QC_L2A.__new__(QC_L2A)
    
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
            "wl": [1600, 2000],
            "y": [0,1],
            "x": [0,1]
        }
    )
    
    qc.__init__(raster=raster)
    qc.swir_module_thresholds = (0.05, 0.1, 1.40)
    qc.process()
    
    expected_swir_module = xr.DataArray(
        np.sqrt(
            np.array([
                [0.1**2, 0.5**2 + 1.0**2],
                [1.0**2 + (-1.0)**2, 0.2**2 + 0.3**2],
            ])
        ),
        dims=("y", "x"),
        coords={
            "y": [0.0, 1.0],
            "x": [0.0, 1.0],
        },
        name='SWIR_module'
    )
    
    assert qc.swir_module.name == expected_swir_module.name
    np.testing.assert_allclose(qc.swir_module, expected_swir_module, atol=1e-3)
    
    expected_flags = xr.Dataset(
        data_vars={
            "swir_module_low": (
                ("y", "x"),
                np.array([
                    [1, 1],
                    [1, 1], 
                ])
            ),
            "swir_module_medium": (
                ("y", "x"),
                np.array([
                    [0, 1],
                    [1, 1], 
                ])
            ),
            "swir_module_high": (
                ("y", "x"),
                np.array([
                    [0, 0],
                    [1, 0], 
                ])
            ),
        },
        coords={
            "y": [0,1],
            "x": [0,1]
        }
    )
    
    for name in qc.swir_flag_names:
        assert qc.flags[name].dtype == bool
        assert qc.flags[name].dims == ("y", "x")

    np.testing.assert_array_equal(qc.flags["swir_module_low"], expected_flags["swir_module_low"])
    np.testing.assert_array_equal(qc.flags["swir_module_medium"], expected_flags["swir_module_medium"])
    np.testing.assert_array_equal(qc.flags["swir_module_high"], expected_flags["swir_module_high"])
    assert qc.flag_descriptions["swir_module_low"] == f'SWIR module sqrt(Rrs1**2 + Rrs2**2) of the L2A Rrs, bands 1600 and 2000 nm, above 0.05 sr-1'
    assert qc.flag_descriptions["swir_module_medium"] == f'SWIR module sqrt(Rrs1**2 + Rrs2**2) of the L2A Rrs, bands 1600 and 2000 nm, above 0.1 sr-1'
    assert qc.flag_descriptions["swir_module_high"] == f'SWIR module sqrt(Rrs1**2 + Rrs2**2) of the L2A Rrs, bands 1600 and 2000 nm, above 1.4 sr-1'

# ============================================================
# Tests for get_stats
# ============================================================

class TestGetStats:
    
    def test_nominal(self,masking_test):

        masking_test.get_stats()

        expected = {
            "flag_cloud": 4 / 6,
            "flag_land": 2 / 6,
            "flag_invalid": 3 / 6,
        }

        for name, value in expected.items():
            assert masking_test.flag_stats[name] == pytest.approx(value)

        # Check that statistics are also stored in the Dataset attributes
        for name, value in expected.items():
            assert masking_test.prod.attrs[name] == pytest.approx(value)

    def test_ignores_none_flags(self,masking_test):

        masking_test.get_stats()

        assert "flag_None" not in masking_test.flag_stats

    def test_no_flags(self):

        flags = xr.DataArray(
            np.zeros((2, 2), dtype=np.uint32),
            dims=("y", "x"),
            attrs={
                "flag_names": ["None"],
                "flag_descriptions": ["None"],
            },
        )

        prod = xr.Dataset({FLAG_NAME: flags})
        masking = Masking(prod)

        masking.get_stats()

        assert masking.flag_stats == {}
        assert masking.prod.attrs == {}
    
# ============================================================
# Tests for print_stats
# ============================================================

class TestPrintStats:
    
    def test_nominal(self,masking_test):

        masking_test.get_stats()

        result = masking_test.print_stats()

        assert isinstance(result, pd.DataFrame)

        assert list(result.columns) == [
            "name",
            "description",
            "bit",
            "statistics",
        ]

        assert len(result) == 3

        assert result["name"].tolist() == [
            "cloud",
            "land",
            "invalid",
        ]

        assert result["description"].tolist() == [
            "Cloud pixels",
            "Land pixels",
            "Invalid pixels",
        ]

        assert result["bit"].tolist() == [0, 1, 2]

        np.testing.assert_allclose(
            result["statistics"].values,
            [4 / 6, 2 / 6, 3 / 6],
        )

    def test_excludes_none_flags(self,masking_test):

        masking_test.get_stats()

        result = masking_test.print_stats()

        assert "None" not in result["name"].values

    def test_no_flags(self):

        flags = xr.DataArray(
            np.zeros((2, 2), dtype=np.uint32),
            dims=("y", "x"),
            attrs={
                "flag_names": ["None", "None"],
                "flag_descriptions": ["None", "None"],
            },
        )

        prod = xr.Dataset({FLAG_NAME: flags})
        masking = Masking(prod)

        result = masking.print_stats()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

        assert list(result.columns) == [
            "name",
            "description",
            "bit",
            "statistics",
        ]


# ============================================================
# Tests for process
# ============================================================

class TestProcess:
    
    def test_output_prod(self,masking_process_test):

        masking = masking_process_test

        result = masking.process(output="prod")

        assert isinstance(result, xr.Dataset)

        assert FLAG_NAME in result

        np.testing.assert_array_equal(
            result[FLAG_NAME].values,
            masking.flags,
        )

        # Check flag metadata
        assert result[FLAG_NAME].attrs["flag_names"][:4].tolist() == [
            "cloud",
            "land",
            "invalid",
            "None",
        ]

        assert result[FLAG_NAME].attrs["flag_descriptions"][:4].tolist() == [
            "Cloud pixels",
            "Land pixels",
            "Invalid pixels",
            "None",
        ]

        # Check statistics
        expected_stats = {
            "flag_cloud": 4 / 6,
            "flag_land": 2 / 6,
            "flag_invalid": 3 / 6,
        }

        for name, value in expected_stats.items():
            assert result.attrs[name] == pytest.approx(value)

    def test_output_flags(self,masking_process_test):

        masking = masking_process_test

        result = masking.process(output="flags")

        assert isinstance(result, xr.DataArray)

        assert result.name == FLAG_NAME

        assert result.dims == ("y", "x")

        np.testing.assert_array_equal(
            result.values,
            masking.flags,
        )

        # Check statistics
        expected_stats = {
            "flag_cloud": 4 / 6,
            "flag_land": 2 / 6,
            "flag_invalid": 3 / 6,
        }

        for name, value in expected_stats.items():
            assert result.attrs[name] == pytest.approx(value)

    def test_output_none(self,masking_process_test):

        masking = masking_process_test

        result = masking.process(output=None)

        assert result is None

        assert FLAG_NAME in masking.prod

        np.testing.assert_array_equal(
            masking.prod[FLAG_NAME].values,
            masking.flags,
        )

        assert "flag_cloud" in masking.prod.attrs

    def test_flags_dataarray(self,masking_process_test):

        masking = masking_process_test

        masking.flags = xr.DataArray(
            np.array([
                [1, 3, 0],
                [4, 5, 7],
            ], dtype=np.uint32),
            dims=("y", "x"),
            coords={
                "y": [0, 1],
                "x": [0, 1, 2],
            },
        )

        result = masking.process()

        assert isinstance(result, xr.Dataset)

        np.testing.assert_array_equal(
            result[FLAG_NAME].values,
            masking.flags.values,
        )

    def test_flag_statistics(self,masking_process_test):

        masking = masking_process_test

        masking.process()

        expected = {
            "flag_cloud": 4 / 6,
            "flag_land": 2 / 6,
            "flag_invalid": 3 / 6,
        }

        for name, value in expected.items():

            assert masking.flag_stats[name] == pytest.approx(value)

            assert masking.prod.attrs[name] == pytest.approx(value)

# ============================================================
# Tests for create_mask
# ============================================================

class TestCreateMask:
    
    def test_tomask_and_tokeep(self,flags_test):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[0, 2],
            tokeep=[1],
        )

        expected = np.array([
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=np.uint8)

        np.testing.assert_array_equal(result.values, expected)

        assert result.dtype == np.uint8

    def test_only_tomask(self,flags_test):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[0, 2],
            tokeep=[],
        )

        expected = np.array([
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=bool)

        np.testing.assert_array_equal(result.values, expected)

    def test_only_tokeep(self,flags_test):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[],
            tokeep=[0,2],
        )

        expected = np.array([
            [1, 0, 1],
            [0, 0, 0],
        ], dtype=bool)

        np.testing.assert_array_equal(result.values, expected)

    def test_empty_flags(self,flags_test):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[],
            tokeep=[],
        )

        expected = np.zeros((2, 3), dtype=np.uint8)

        np.testing.assert_array_equal(result.values, expected)

        assert result.dtype == np.uint8

    def test_empty_flags(self,flags_test):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[],
            tokeep=[],
        )

        expected = np.zeros((2, 3), dtype=np.uint8)

        np.testing.assert_array_equal(result.values, expected)

        assert result.dtype == np.uint8

    @pytest.mark.parametrize(
        "dtype",
        [np.uint8, np.uint16, np.int32],
    )
    def test_custom_dtype(self,flags_test, dtype):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[0],
            tokeep=[],
            _type=dtype,
        )

        assert result.dtype == dtype

    def test_coordinates(self,flags_test):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[0],
        )

        assert result.dims == flags_test.dims

        xr.testing.assert_equal(
            result.coords.to_dataset(),
            flags_test.coords.to_dataset(),
        )

    def test_attributes(self,flags_test):

        result = Masking.create_mask(
            flags=flags_test,
            tomask=[0],
            mask_name="test_mask",
        )

        assert result.name == "test_mask"

        assert result.attrs["long_name"] == "binary mask from flags"

        assert result.attrs["description"] == (
            "good pixels for mask == 0, bad pixels when mask == 1"
        )


def test_qc_l2a_negative_rrs():
    from GRSl2bgen.mask import QC_L2A, L2B_FLAGS, l2b_flags
    wl = [443, 490, 560, 1610, 2200]
    rrs = np.zeros((len(wl), 1, 5)) + 0.005
    rrs[3:, 0, :] = 0.                 # black SWIR
    rrs[2, 0, 1] = -0.0002             # low
    rrs[0, 0, 2] = -0.001              # low + medium, black SWIR
    rrs[0, 0, 3] = -0.003              # low + medium + high, non-black SWIR -> neg_rrs_swir
    rrs[3, 0, 3] = 0.004
    rrs[3, 0, 4] = -0.01               # negative in the SWIR only: outside the tested range
    raster = xr.Dataset({'Rrs': (('wl', 'y', 'x'), rrs)}, coords={'wl': wl, 'y': [0], 'x': list(range(5))})
    qc = QC_L2A(raster)
    qc.process()
    assert qc.flags['neg_rrs_low'].values.tolist() == [[False, True, True, True, False]]
    assert qc.flags['neg_rrs_medium'].values.tolist() == [[False, False, True, True, False]]
    assert qc.flags['neg_rrs_high'].values.tolist() == [[False, False, False, True, False]]
    assert qc.flags['neg_rrs_swir'].values.tolist() == [[False, False, False, True, False]]
    assert np.allclose(qc.negativity.values, [[0, 0.0002, 0.001, 0.003, 0]])
    assert '443' in qc.flag_descriptions['neg_rrs_low']
    flags = l2b_flags(raster.Rrs.isel(wl=0, drop=True), qc.flags, qc.flag_descriptions)
    for name in ('neg_rrs_low', 'neg_rrs_medium', 'neg_rrs_high', 'neg_rrs_swir'):
        assert flags.attrs['flag_names'][L2B_FLAGS[name][0]] == name
    assert ((flags.values >> L2B_FLAGS['neg_rrs_swir'][0]) & 1).tolist() == [[0, 0, 0, 1, 0]]

    qc = QC_L2A(raster, negative_thresholds=None)
    qc.process()
    assert not any(k.startswith('neg_rrs') for k in qc.flags)
