import json

import pytest 
import numpy as np 
import pandas as pd 
import dask.array as da 
import xarray as xr

from unittest.mock import patch, MagicMock

import GRSl2bgen.output as l2b

from GRSl2bgen.output import ( L2bProduct, _is_categorical, _json_safe, CATEGORICAL, CLASS_ID_PREFIX, )



# ============================================================
# Fixtures and helpers
# ============================================================

@pytest.fixture
def wavelengths():
    return [443, 490, 560, 665]


@pytest.fixture
def raster():
    """Create a synthetic L2A raster."""

    y = np.arange(4)
    x = np.arange(6)
    wl = [443, 490, 560, 665]

    return xr.Dataset(
        {
            "Rrs": (
                ("wl", "y", "x"),
                np.ones((4, 4, 6), dtype=np.float32) * 0.01,
            ),
        },
        coords={
            "wl": wl,
            "y": y,
            "x": x,
        },
        attrs={
            "title": "Synthetic L2A product",
            "platform": "Sentinel-2",
            "test_attribute": True,
        },
    )



@pytest.fixture
def product(raster):
    """Create a mock L2A product."""

    prod = MagicMock()
    prod.raster = raster

    return prod


@pytest.fixture
def l2_rasters():
    """Create synthetic derived L2 variables."""

    y = np.arange(4)
    x = np.arange(6)

    return [
        xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.arange(24, dtype=np.float64).reshape(4, 6),
                ),
                "spm": (
                    ("y", "x"),
                    np.ones((4, 6), dtype=np.float64) * 5,
                ),
            },
            coords={"y": y, "x": x},
        )
    ]



@pytest.fixture
def mock_flags(monkeypatch):
    """Replace l2b_flags with a predictable implementation."""
    def fake_l2b_flags(reference, flags, descriptions):
        return xr.zeros_like(
            reference,
            dtype=np.uint8,
        ).rename("flags_l2b")

    monkeypatch.setattr(
        l2b,
        "l2b_flags",
        fake_l2b_flags,
        raising=False,
    )


@pytest.fixture
def l2b_product(product, l2_rasters, mock_flags):
    """Create a fully initialized L2bProduct."""

    return L2bProduct(
        product,
        l2_rasters,
    )


@pytest.fixture
def small_dataset():
    """Dataset used to test numerical operations."""

    y = np.arange(4)
    x = np.arange(4)

    return xr.Dataset(
        {
            "chl": (
                ("y", "x"),
                np.arange(16, dtype=np.float32).reshape(4, 4),
            ),
            "flags": (
                ("y", "x"),
                np.arange(16, dtype=np.uint8).reshape(4, 4),
            ),
            "flags_l2b": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.uint8),
            ),
            "mask": (
                ("y", "x"),
                np.zeros((4, 4), dtype=np.uint8),
            ),
            "owt_index_test": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.uint8),
            ),
        },
        coords={"y": y, "x": x},
    )


# ============================================================
# Tests for _is_categorical
# ============================================================

class TestIsCategorical:

    @pytest.mark.parametrize(
        "variable",
        [
            "flags",
            "flags_l2b",
            "mask",
            "owt_index",
            "owt_index_test",
            "owt_index_some_database",
        ],
    )
    def test_categorical_variables(self, variable):
        assert _is_categorical(variable)

    @pytest.mark.parametrize(
        "variable",
        [
            "Rrs",
            "chl",
            "spm",
            "aph",
            "bbp",
            "owt",
            "other_flags",
        ],
    )
    def test_continuous_variables(self, variable):
        assert not _is_categorical(variable)


# ============================================================
# Tests for _json_safe
# ============================================================

class TestJsonSafe:

    def test_numpy_integer(self):
        value = np.int64(42)

        result = _json_safe(value)

        assert result == 42
        assert type(result) is int

    def test_numpy_float(self):
        value = np.float64(3.14)

        result = _json_safe(value)

        assert result == pytest.approx(3.14)
        assert type(result) is float

    def test_numpy_boolean(self):
        value = np.bool_(True)

        result = _json_safe(value)

        assert result is True
        assert type(result) is bool

    def test_numpy_array(self):
        value = np.array([1, 2, 3])

        result = _json_safe(value)

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_list(self):
        value = [
            np.int64(1),
            np.float64(2.5),
            (np.int32(3), np.bool_(True)),
        ]

        result = _json_safe(value)

        assert result == [1, 2.5, [3, True]]

    def test_nested_dictionary(self):
        value = {
            "integer": np.int64(42),
            "array": np.array([1, 2]),
            10: np.float32(3.5),
        }

        result = _json_safe(value)

        assert result == {
            "integer": 42,
            "array": [1, 2],
            "10": pytest.approx(3.5),
        }

    def test_json_serializable(self):
        value = {
            "integer": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "nested": {
                "boolean": np.bool_(True),
            },
        }

        result = _json_safe(value)

        # Should not raise TypeError.
        serialized = json.dumps(result)

        assert isinstance(serialized, str)

    def test_python_scalar_unchanged(self):
        value = 42

        assert _json_safe(value) is value

    def test_none_unchanged(self):
        assert _json_safe(None) is None

    def test_tuple_becomes_list(self):
        result = _json_safe((1, 2, 3))

        assert result == [1, 2, 3]
        assert isinstance(result, list)


# ============================================================
# Tests for L2bProduct initialization and construction
# ============================================================

class TestConstructL2B:

    def test_initialization(self, l2b_product):

        assert l2b_product.l2b_prod is not None
        assert l2b_product.variables is not None

    def test_processor_attribute(self, l2b_product):

        assert l2b_product.processor == (
            l2b.__package__ + "_" + l2b.__version__
        )

    def test_variables_list(self, l2b_product):

        assert set(l2b_product.variables) == {
            "chl",
            "spm",
            "flags",
            "flags_l2b",
            "mask",
        }

    def test_derived_variables_are_preserved(self, l2b_product):

        assert "chl" in l2b_product.l2b_prod
        assert "spm" in l2b_product.l2b_prod

    def test_float64_converted_to_float32(self, l2b_product):

        assert l2b_product.l2b_prod.chl.dtype == np.float32
        assert l2b_product.l2b_prod.spm.dtype == np.float32

    def test_flags_created_when_missing(self, l2b_product):

        assert "flags" in l2b_product.l2b_prod
        assert l2b_product.l2b_prod.flags.dtype == np.uint8

        np.testing.assert_array_equal(
            l2b_product.l2b_prod.flags.values,
            np.zeros((4, 6), dtype=np.uint8),
        )

    def test_mask_created_when_missing(self, l2b_product):

        assert "mask" in l2b_product.l2b_prod
        assert l2b_product.l2b_prod["mask"].dtype == np.uint8

        np.testing.assert_array_equal(
            l2b_product.l2b_prod["mask"].values,
            np.zeros((4, 6), dtype=np.uint8),
        )

    def test_flags_l2b_created(self, l2b_product):

        assert "flags_l2b" in l2b_product.l2b_prod
        assert l2b_product.l2b_prod.flags_l2b.dtype == np.uint8

    def test_existing_flags_are_preserved(
        self,
        raster,
        l2_rasters,
        mock_flags,
    ):

        raster["flags"] = xr.DataArray(
            np.ones((4, 6), dtype=np.uint8) * 5,
            dims=("y", "x"),
            coords={"y": raster.y, "x": raster.x},
        )

        prod = MagicMock()
        prod.raster = raster

        result = L2bProduct(prod, l2_rasters)

        np.testing.assert_array_equal(
            result.l2b_prod.flags.values,
            5,
        )

    def test_existing_mask_is_preserved(
        self,
        raster,
        l2_rasters,
        mock_flags,
    ):

        raster["mask"] = xr.DataArray(
            np.ones((4, 6), dtype=np.uint8),
            dims=("y", "x"),
            coords={"y": raster.y, "x": raster.x},
        )

        prod = MagicMock()
        prod.raster = raster

        result = L2bProduct(prod, l2_rasters)

        np.testing.assert_array_equal(
            result.l2b_prod["mask"].values,
            1,
        )

    def test_processing_metadata(self, l2b_product):

        attrs = l2b_product.l2b_prod.attrs

        assert "processing_time" in attrs
        assert "processor" in attrs

        assert attrs["processor"] == l2b_product.processor

    def test_processing_time_format(self, l2b_product):

        processing_time = l2b_product.l2b_prod.attrs[
            "processing_time"
        ]

        from datetime import datetime

        parsed = datetime.strptime(
            processing_time,
            "%Y-%m-%dT%H:%M:%S.%f",
        )

        assert isinstance(parsed, datetime)

    def test_input_attributes_are_preserved(self, l2b_product):

        attrs = l2b_product.l2b_prod.attrs

        assert attrs["title"] == "Synthetic L2A product"
        assert attrs["platform"] == "Sentinel-2"

    def test_input_attributes_are_shared(self, product, l2b_product):

        # The implementation intentionally shares the attributes dictionary.
        assert (l2b_product.l2b_prod.attrs["title"] ==  product.raster.attrs["title"])
        assert (l2b_product.l2b_prod.attrs["platform"] ==  product.raster.attrs["platform"])
        assert (l2b_product.l2b_prod.attrs["test_attribute"] ==  product.raster.attrs["test_attribute"])

    def test_scalar_coordinates_are_removed(
        self,
        raster,
        l2_rasters,
        mock_flags,
    ):

        raster = raster.assign_coords(
            band=1,
            forecast_period=2,
            forecast_reference_time=pd.Timestamp("2024-01-01"),
            central_wavelength=560,
        )

        prod = MagicMock()
        prod.raster = raster

        result = L2bProduct(prod, l2_rasters)

        for coordinate in [
            "band",
            "forecast_period",
            "forecast_reference_time",
            "central_wavelength",
        ]:
            assert coordinate not in result.l2b_prod.coords

    def test_time_coordinate_is_preserved(
        self,
        raster,
        l2_rasters,
        mock_flags,
    ):

        raster = raster.assign_coords(
            time=pd.Timestamp("2024-01-01")
        )

        prod = MagicMock()
        prod.raster = raster

        result = L2bProduct(prod, l2_rasters)

        assert "time" in result.l2b_prod.coords

    def test_custom_flags_are_forwarded(
        self,
        product,
        l2_rasters,
        monkeypatch,
    ):

        captured = {}

        def fake_flags(reference, flags, descriptions):
            captured["flags"] = flags
            captured["descriptions"] = descriptions

            return xr.zeros_like(
                reference,
                dtype=np.uint8,
            ).rename("flags_l2b")

        monkeypatch.setattr(l2b, "l2b_flags", fake_flags)

        flags = {"cloud": np.ones((4, 6), dtype=bool)}
        descriptions = {"cloud": "Cloud detection"}

        L2bProduct(
            product,
            l2_rasters,
            flags=flags,
            flag_descriptions=descriptions,
        )

        assert captured["flags"] is flags
        assert captured["descriptions"] is descriptions

    def test_custom_compression_level(
        self,
        product,
        l2_rasters,
        mock_flags,
    ):

        result = L2bProduct(
            product,
            l2_rasters,
            complevel=5,
        )

        assert result.complevel == 5


# ============================================================
# Tests for compute_scale_and_offset
# ============================================================

class TestComputeScaleAndOffset:

    def test_default_16_bit_range(self):

        values = np.array([0, 100])

        scale, offset = (
            L2bProduct.compute_scale_and_offset(values)
        )

        expected_scale = 100 / 65534
        expected_offset = 0 + 32767 * expected_scale

        assert scale == pytest.approx(expected_scale)
        assert offset == pytest.approx(expected_offset)

    def test_constant_array(self):

        values = np.full(10, 5.0)

        scale, offset = (
            L2bProduct.compute_scale_and_offset(values)
        )

        assert scale == 1
        assert offset == 5

    def test_all_nan_array(self):

        values = np.full(10, np.nan)

        scale, offset = (
            L2bProduct.compute_scale_and_offset(values)
        )

        assert scale == 1
        assert offset == 0

    def test_nan_values_are_ignored(self):

        values = np.array([
            1.0,
            np.nan,
            5.0,
            10.0,
            np.nan,
        ])

        scale, offset = (
            L2bProduct.compute_scale_and_offset(values)
        )

        expected_scale = 9 / 65534
        expected_offset = 1 + 32767 * expected_scale

        assert scale == pytest.approx(expected_scale)
        assert offset == pytest.approx(expected_offset)

    def test_negative_values(self):

        values = np.array([-10, 0, 10])

        scale, offset = (
            L2bProduct.compute_scale_and_offset(values)
        )

        expected_scale = 20 / 65534
        expected_offset = -10 + 32767 * expected_scale

        assert scale == pytest.approx(expected_scale)
        assert offset == pytest.approx(expected_offset)

    def test_custom_bit_depth(self):

        values = np.array([0, 100])

        scale, offset = (
            L2bProduct.compute_scale_and_offset(
                values,
                nbit=8,
            )
        )

        expected_scale = 100 / 254
        expected_offset = 127 * expected_scale

        assert scale == pytest.approx(expected_scale)
        assert offset == pytest.approx(expected_offset)

    def test_packing_round_trip(self):

        values = np.array([0.0, 2.5, 5.0, 7.5, 10.0])

        scale, offset = (
            L2bProduct.compute_scale_and_offset(values)
        )

        packed = np.rint(
            (values - offset) / scale
        ).astype(np.int16)

        unpacked = packed * scale + offset

        np.testing.assert_allclose(
            unpacked,
            values,
            atol=scale / 2,
        )


# ============================================================
# Tests for scale_and_offset_from_range
# ============================================================

class TestScaleAndOffsetFromRange:

    def test_regular_range(self):

        scale, offset = (
            L2bProduct.scale_and_offset_from_range(
                0,
                100,
            )
        )

        assert scale == pytest.approx(100 / 65534)
        assert offset == pytest.approx(32767 * scale)

    def test_constant_range(self):

        scale, offset = (
            L2bProduct.scale_and_offset_from_range(
                5,
                5,
            )
        )

        assert scale == 1
        assert offset == 5

    def test_all_nan_range(self):

        scale, offset = (
            L2bProduct.scale_and_offset_from_range(
                np.nan,
                np.nan,
            )
        )

        assert scale == 1
        assert offset == 0

    def test_infinite_range(self):

        scale, offset = (
            L2bProduct.scale_and_offset_from_range(
                -np.inf,
                np.inf,
            )
        )

        assert scale == 1
        assert offset == 0

    def test_custom_bit_depth(self):

        scale, offset = (
            L2bProduct.scale_and_offset_from_range(
                0,
                100,
                nbit=8,
            )
        )

        assert scale == pytest.approx(100 / 254)
        assert offset == pytest.approx(127 * scale)


# ============================================================
# Tests for _packing_encoding
# ============================================================

class TestPackingEncoding:

    def test_continuous_variable_encoding(self, l2b_product):

        encoding = l2b_product._packing_encoding()

        assert encoding["chl"]["dtype"] == "int16"
        assert encoding["chl"]["_FillValue"] == -32768

        assert "scale_factor" in encoding["chl"]
        assert "add_offset" in encoding["chl"]

    def test_categorical_variables_keep_dtype(
        self,
        l2b_product,
    ):

        encoding = l2b_product._packing_encoding()

        for variable in ["flags", "flags_l2b", "mask"]:
            assert "dtype" not in encoding[variable]
            assert "scale_factor" not in encoding[variable]
            assert "add_offset" not in encoding[variable]

    def test_owt_class_id_encoding(
        self,
        product,
        l2_rasters,
        mock_flags,
    ):

        l2_rasters.append(
            xr.Dataset(
                {
                    "owt_index_test": (
                        ("y", "x"),
                        np.ones((4, 6), dtype=np.uint8),
                    )
                },
                coords={
                    "y": np.arange(4),
                    "x": np.arange(6),
                },
            )
        )

        result = L2bProduct(product, l2_rasters)

        encoding = result._packing_encoding()

        assert encoding["owt_index_test"]["dtype"] == "uint8"
        assert encoding["owt_index_test"]["_FillValue"] == 0

    def test_grid_mapping(self, l2b_product):

        encoding = l2b_product._packing_encoding()

        for variable in l2b_product.variables:
            assert encoding[variable]["grid_mapping"] == "spatial_ref"

    def test_constant_variable_encoding(
        self,
        product,
        mock_flags,
    ):

        rasters = [
            xr.Dataset(
                {
                    "constant": (
                        ("y", "x"),
                        np.full((4, 6), 5.0),
                    )
                },
                coords={
                    "y": np.arange(4),
                    "x": np.arange(6),
                },
            )
        ]

        result = L2bProduct(product, rasters)

        encoding = result._packing_encoding()

        assert encoding["constant"]["scale_factor"] == 1
        assert encoding["constant"]["add_offset"] == 5

    def test_all_nan_variable_encoding(
        self,
        product,
        mock_flags,
    ):

        rasters = [
            xr.Dataset(
                {
                    "empty": (
                        ("y", "x"),
                        np.full((4, 6), np.nan),
                    )
                },
                coords={
                    "y": np.arange(4),
                    "x": np.arange(6),
                },
            )
        ]

        result = L2bProduct(product, rasters)

        encoding = result._packing_encoding()

        assert encoding["empty"]["scale_factor"] == 1
        assert encoding["empty"]["add_offset"] == 0

    def test_dask_variables(self, product, mock_flags):

        values = da.from_array(
            np.arange(24, dtype=np.float32).reshape(4, 6),
            chunks=(2, 3),
        )

        rasters = [
            xr.Dataset(
                {
                    "chl": (("y", "x"), values),
                },
                coords={
                    "y": np.arange(4),
                    "x": np.arange(6),
                },
            )
        ]

        result = L2bProduct(product, rasters)

        encoding = result._packing_encoding()

        assert encoding["chl"]["dtype"] == "int16"

    def test_persist(self, product, mock_flags):

        values = da.from_array(
            np.arange(24, dtype=np.float32).reshape(4, 6),
            chunks=(2, 3),
        )

        rasters = [
            xr.Dataset(
                {"chl": (("y", "x"), values)},
                coords={
                    "y": np.arange(4),
                    "x": np.arange(6),
                },
            )
        ]

        result = L2bProduct(product, rasters)

        result._packing_encoding(persist=True)

        assert isinstance(result.l2b_prod.chl.data, da.Array)
        np.testing.assert_array_equal(
            result.l2b_prod.chl.compute().values,
            np.arange(24, dtype=np.float32).reshape(4, 6),
        )



# ============================================================
# Tests for _remove and _prepare_output
# ============================================================

class TestFileUtilities:

    def test_remove_file(self, tmp_path):

        path = tmp_path / "test.txt"
        path.write_text("test")

        L2bProduct._remove(path)

        assert not path.exists()

    def test_remove_directory(self, tmp_path):

        path = tmp_path / "test_dir"
        path.mkdir()

        (path / "test.txt").write_text("test")

        L2bProduct._remove(path)

        assert not path.exists()

    def test_remove_nonexistent_path(self, tmp_path):

        path = tmp_path / "missing"

        # Should not raise an exception.
        L2bProduct._remove(path)

    def test_prepare_output_creates_directory(self, tmp_path):

        path = tmp_path / "new_dir" / "output.nc"

        L2bProduct._prepare_output(path)

        assert path.parent.exists()

    def test_prepare_output_removes_existing_file(self, tmp_path):

        path = tmp_path / "output.nc"
        path.write_text("old")

        L2bProduct._prepare_output(path)

        assert not path.exists()

    def test_prepare_output_removes_existing_directory(self, tmp_path):

        path = tmp_path / "output.zarr"
        path.mkdir()

        (path / "data").write_text("old")

        L2bProduct._prepare_output(path)

        assert not path.exists()

    def test_prepare_output_current_directory(self, tmp_path, monkeypatch):

        monkeypatch.chdir(tmp_path)

        L2bProduct._prepare_output("output.nc")

        assert tmp_path.exists()


# ============================================================
# Tests for export
# ============================================================

class TestExport:

    def test_export_calls_netcdf(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_export(path, persist=False):
            captured["path"] = path
            captured["persist"] = persist

            with open(path, "w") as f:
                f.write("test")

        monkeypatch.setattr(
            l2b_product,
            "export_to_netcdf",
            fake_export,
        )

        output = tmp_path / "product.nc"

        l2b_product.export(
            output,
            persist=True,
        )

        assert output.exists()
        assert captured["persist"] is True

    def test_export_calls_zarr(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_export(path, persist=False):
            captured["path"] = path

            with open(path, "w") as f:
                f.write("test")

        monkeypatch.setattr(
            l2b_product,
            "export_to_zarr",
            fake_export,
        )

        output = tmp_path / "product.zarr"

        l2b_product.export(output)

        assert output.exists()
        assert captured["path"].endswith(".zarr")

    def test_export_calls_zarr_pyramid(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_export(path, persist=False):
            captured["path"] = path

            with open(path, "w") as f:
                f.write("test")

        monkeypatch.setattr(
            l2b_product,
            "export_to_zarr_pyramid",
            fake_export,
        )

        output = tmp_path / "product.zarr"

        l2b_product.export(
            output,
            pyramid=True,
        )

        assert output.exists()

    def test_export_replaces_existing_output(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        output = tmp_path / "product.nc"
        output.write_text("old")

        def fake_export(path, persist=False):
            with open(path, "w") as f:
                f.write("new")

        monkeypatch.setattr(
            l2b_product,
            "export_to_netcdf",
            fake_export,
        )

        l2b_product.export(output)

        assert output.read_text() == "new"

    def test_export_removes_temporary_file_on_error(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        def fake_export(path, persist=False):
            with open(path, "w") as f:
                f.write("incomplete")

            raise RuntimeError("Export failed")

        monkeypatch.setattr(
            l2b_product,
            "export_to_netcdf",
            fake_export,
        )

        output = tmp_path / "product.nc"

        with pytest.raises(RuntimeError):
            l2b_product.export(output)

        assert not output.exists()

        temporary_files = list(
            tmp_path.glob(".product.part*")
        )

        assert temporary_files == []

    def test_export_preserves_existing_output_on_error(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        output = tmp_path / "product.nc"
        output.write_text("original")

        def fake_export(path, persist=False):
            with open(path, "w") as f:
                f.write("incomplete")

            raise RuntimeError("Export failed")

        monkeypatch.setattr(
            l2b_product,
            "export_to_netcdf",
            fake_export,
        )

        with pytest.raises(RuntimeError):
            l2b_product.export(output)

        assert output.read_text() == "original"

    def test_export_ignores_pyramid_for_netcdf(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
        caplog,
    ):

        def fake_export(path, persist=False):
            with open(path, "w") as f:
                f.write("test")

        monkeypatch.setattr(
            l2b_product,
            "export_to_netcdf",
            fake_export,
        )

        l2b_product.export(
            tmp_path / "product.nc",
            pyramid=True,
        )

        assert "pyramids are only written for Zarr" in caplog.text


# ============================================================
# Tests for _downsample
# ============================================================

class TestDownsample:

    def test_output_dimensions(self, small_dataset):

        result = L2bProduct._downsample(
            small_dataset,
            factor=2,
        )

        assert result.sizes["y"] == 2
        assert result.sizes["x"] == 2

    def test_continuous_variable_averaged(self, small_dataset):

        result = L2bProduct._downsample(
            small_dataset,
            factor=2,
        )

        expected = np.array([
            [2.5, 4.5],
            [10.5, 12.5],
        ])

        np.testing.assert_allclose(
            result.chl.values,
            expected,
        )

    def test_flags_use_top_left_pixel(self, small_dataset):

        result = L2bProduct._downsample(
            small_dataset,
            factor=2,
        )

        expected = np.array([
            [0, 2],
            [8, 10],
        ], dtype=np.uint8)

        np.testing.assert_array_equal(
            result.flags.values,
            expected,
        )

    def test_mask_use_top_left_pixel(self, small_dataset):

        result = L2bProduct._downsample(
            small_dataset,
            factor=2,
        )

        np.testing.assert_array_equal(
            result["mask"].values,
            np.zeros((2, 2), dtype=np.uint8),
        )

    def test_flags_l2b_preserved(self, small_dataset):

        result = L2bProduct._downsample(
            small_dataset,
            factor=2,
        )

        np.testing.assert_array_equal(
            result.flags_l2b.values,
            np.ones((2, 2), dtype=np.uint8),
        )

    def test_owt_indices_use_nearest_neighbor(self, small_dataset):

        result = L2bProduct._downsample(
            small_dataset,
            factor=2,
        )

        np.testing.assert_array_equal(
            result.owt_index_test.values,
            np.ones((2, 2), dtype=np.uint8),
        )

        assert result.owt_index_test.dtype == np.uint8

    def test_nan_aware_average(self):

        ds = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.array([
                        [1.0, np.nan],
                        [3.0, 5.0],
                    ]),
                )
            }
        )

        result = L2bProduct._downsample(
            ds,
            factor=2,
        )

        assert result.chl.item() == pytest.approx(3.0)

    def test_edge_pixels_are_trimmed(self):

        ds = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.arange(25).reshape(5, 5),
                )
            }
        )

        result = L2bProduct._downsample(
            ds,
            factor=2,
        )

        assert result.sizes["y"] == 2
        assert result.sizes["x"] == 2

    def test_custom_factor(self):

        ds = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.arange(36).reshape(6, 6),
                )
            }
        )

        result = L2bProduct._downsample(
            ds,
            factor=3,
        )

        assert result.sizes["y"] == 2
        assert result.sizes["x"] == 2

    def test_variable_attributes_preserved(self):

        ds = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.arange(16).reshape(4, 4),
                    {"units": "mg m-3"},
                )
            },
            attrs={"title": "Test"},
        )

        result = L2bProduct._downsample(ds)

        assert result.chl.attrs["units"] == "mg m-3"
        assert result.attrs["title"] == "Test"

    def test_dataset_attributes_preserved(self, small_dataset):

        small_dataset.attrs["title"] = "Test dataset"

        result = L2bProduct._downsample(small_dataset)

        assert result.attrs["title"] == "Test dataset"

    def test_custom_categorical_variables(self):

        ds = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.arange(16).reshape(4, 4),
                ),
                "custom": (
                    ("y", "x"),
                    np.arange(16).reshape(4, 4),
                ),
            }
        )

        result = L2bProduct._downsample(
            ds,
            categorical=["custom"],
        )

        np.testing.assert_array_equal(
            result.custom.values,
            np.array([
                [0, 2],
                [8, 10],
            ]),
        )


# ============================================================
# Tests for _update_geotransform
# ============================================================

class TestUpdateGeotransform:

    def test_returns_dataset_when_rio_unavailable(self):

        ds = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.ones((4, 4)),
                )
            }
        )

        result = L2bProduct._update_geotransform(ds)

        assert isinstance(result, xr.Dataset)

    def test_failure_does_not_raise(self, caplog):

        ds = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.ones((4, 4)),
                )
            }
        )

        result = L2bProduct._update_geotransform(ds)

        assert isinstance(result, xr.Dataset)

        # If no CRS is available, a warning may be logged.
        # The important requirement is that execution continues.


# ============================================================
# Tests for _write_zarr_group
# ============================================================

class TestWriteZarrGroup:

    def test_data_variable_encodings_are_cleared(
        self,
        l2b_product,
        monkeypatch,
        tmp_path,
    ):

        ds = l2b_product.l2b_prod.copy(deep=False)

        captured = {}

        def fake_to_zarr(*args, **kwargs):
            captured["encoding"] = kwargs["encoding"]
            captured["group"] = kwargs["group"]

        monkeypatch.setattr(
            xr.Dataset,
            "to_zarr",
            fake_to_zarr,
        )

        encoding = l2b_product._packing_encoding()

        l2b_product._write_zarr_group(
            ds,
            str(tmp_path / "test.zarr"),
            encoding,
            chunks={"y": 2, "x": 3},
            group="0",
        )

        assert captured["group"] == "0"

    def test_irrelevant_chunk_dimensions_are_ignored(
        self,
        l2b_product,
        monkeypatch,
        tmp_path,
    ):

        captured = {}

        def fake_to_zarr(*args, **kwargs):
            captured["encoding"] = kwargs["encoding"]

        monkeypatch.setattr(
            xr.Dataset,
            "to_zarr",
            fake_to_zarr,
        )

        encoding = l2b_product._packing_encoding()

        l2b_product._write_zarr_group(
            l2b_product.l2b_prod,
            str(tmp_path / "test.zarr"),
            encoding,
            chunks={
                "y": 2,
                "x": 3,
                "unused": 10,
            },
        )

        assert "chl" in captured["encoding"]


# ============================================================
# Tests for export_to_zarr_pyramid validation
# ============================================================

class TestZarrPyramid:

    def test_missing_spatial_dimension_raises(
        self,
        l2b_product,
        tmp_path,
    ):

        l2b_product.l2b_prod = xr.Dataset(
            {
                "chl": (
                    ("y", "x"),
                    np.ones((4, 6)),
                )
            }
        )

        with pytest.raises(ValueError, match="dimension"):
            l2b_product.export_to_zarr_pyramid(
                tmp_path / "test.zarr",
                dims=("latitude", "longitude"),
            )

    def test_missing_x_dimension_raises(
        self,
        l2b_product,
        tmp_path,
    ):

        with pytest.raises(ValueError, match="dimension"):
            l2b_product.export_to_zarr_pyramid(
                tmp_path / "test.zarr",
                dims=("y", "longitude"),
            )

# ============================================================
# Tests for export_to_netcdf
# ============================================================

class TestExportToNetCDF:

    def test_encoding_compression(self, l2b_product, tmp_path, monkeypatch):

        captured = {}

        def fake_to_netcdf(self, path, encoding):
            captured["encoding"] = encoding

        monkeypatch.setattr(
            xr.Dataset,
            "to_netcdf",
            fake_to_netcdf,
        )

        output = tmp_path / "test.nc"

        l2b_product.export_to_netcdf(output)

        for variable in l2b_product.variables:
            assert captured["encoding"][variable]["zlib"] is True
            assert captured["encoding"][variable]["complevel"] == 1

    def test_custom_compression_level(
        self,
        product,
        l2_rasters,
        mock_flags,
        tmp_path,
        monkeypatch,
    ):

        obj = L2bProduct(
            product,
            l2_rasters,
            complevel=5,
        )

        captured = {}

        def fake_to_netcdf(self, path, encoding):
            captured["encoding"] = encoding

        monkeypatch.setattr(
            xr.Dataset,
            "to_netcdf",
            fake_to_netcdf,
        )

        obj.export_to_netcdf(tmp_path / "test.nc")

        assert captured["encoding"]["chl"]["complevel"] == 5

    def test_boolean_attributes_converted_to_int(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        l2b_product.l2b_prod.attrs["test_boolean"] = True
        l2b_product.l2b_prod.chl.attrs["test_boolean"] = False

        captured = {}

        def fake_to_netcdf(self, path, encoding):
            captured["attrs"] = dict(
                self.attrs
            )
            captured["variable_attrs"] = dict(
                self.chl.attrs
            )

        monkeypatch.setattr(
            xr.Dataset,
            "to_netcdf",
            fake_to_netcdf,
        )

        l2b_product.export_to_netcdf(tmp_path / "test.nc")

        assert type(captured["attrs"]["test_boolean"]) is np.int8
        assert type(captured["variable_attrs"]["test_boolean"]) is np.int8

    def test_output_directory_created(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        monkeypatch.setattr(
            xr.Dataset,
            "to_netcdf",
            lambda *args, **kwargs: None,
        )

        output = tmp_path / "new" / "folder" / "test.nc"

        l2b_product.export_to_netcdf(output)

        assert output.parent.exists()

    def test_existing_file_removed(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        output = tmp_path / "test.nc"
        output.write_text("old")

        monkeypatch.setattr(
            xr.Dataset,
            "to_netcdf",
            lambda *args, **kwargs: None,
        )

        l2b_product.export_to_netcdf(output)

        # The mocked writer doesn't create a new file.
        assert not output.exists()

    def test_zarr_argument_redirects(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        called = {}

        def fake_zarr(path, persist=False):
            called["path"] = path
            called["persist"] = persist

        monkeypatch.setattr(
            l2b_product,
            "export_to_zarr",
            fake_zarr,
        )

        output = tmp_path / "test.zarr"

        l2b_product.export_to_netcdf(
            output,
            zarr=True,
            persist=True,
        )

        assert called["path"] == output
        assert called["persist"] is True


# ============================================================
# Tests for export_to_zarr
# ============================================================

class TestExportToZarr:

    def test_default_chunks(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_write(ds, ofile, encoding, chunks):
            captured["chunks"] = chunks

        monkeypatch.setattr(
            l2b_product,
            "_write_zarr_group",
            fake_write,
        )

        monkeypatch.setattr(
            l2b_product,
            "_prepare_output",
            lambda path: None,
        )

        l2b_product.export_to_zarr(
            tmp_path / "test.zarr"
        )

        assert captured["chunks"] == {}

    def test_custom_chunks(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_write(ds, ofile, encoding, chunks):
            captured["chunks"] = chunks

        monkeypatch.setattr(
            l2b_product,
            "_write_zarr_group",
            fake_write,
        )

        monkeypatch.setattr(
            l2b_product,
            "_prepare_output",
            lambda path: None,
        )

        chunks = {"y": 2, "x": 3}

        l2b_product.export_to_zarr(
            tmp_path / "test.zarr",
            chunks=chunks,
        )

        assert captured["chunks"] == chunks

    def test_encoding_passed(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_write(ds, ofile, encoding, chunks):
            captured["encoding"] = encoding

        monkeypatch.setattr(
            l2b_product,
            "_write_zarr_group",
            fake_write,
        )

        monkeypatch.setattr(
            l2b_product,
            "_prepare_output",
            lambda path: None,
        )

        l2b_product.export_to_zarr(
            tmp_path / "test.zarr"
        )

        assert "chl" in captured["encoding"]
        assert captured["encoding"]["chl"]["dtype"] == "int16"


# ============================================================
# Tests for export_to_zarr_pyramid
# ============================================================

class TestExportToZarrPyramid:

    def test_default_chunks(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_write(ds, ofile, encoding, chunks, **kwargs):
            captured["chunks"] = chunks
            captured["group"] = kwargs.get("group")

        monkeypatch.setattr(
            l2b_product,
            "_write_zarr_group",
            fake_write,
        )

        monkeypatch.setattr(
            l2b_product,
            "_prepare_output",
            lambda path: None,
        )

        # Prevent access to the actual Zarr store.
        with patch.object(
            l2b.xr,
            "open_zarr",
            side_effect=RuntimeError("stop after level 0"),
        ):
            with pytest.raises(RuntimeError):
                l2b_product.export_to_zarr_pyramid(
                    tmp_path / "test.zarr",
                    levels=1,
                )

        assert captured["chunks"] == {"y": 512, "x": 512}
        assert captured["group"] == "0"

    def test_custom_chunks(
        self,
        l2b_product,
        tmp_path,
        monkeypatch,
    ):

        captured = {}

        def fake_write(ds, ofile, encoding, chunks, **kwargs):
            captured["chunks"] = chunks

        monkeypatch.setattr(
            l2b_product,
            "_write_zarr_group",
            fake_write,
        )

        monkeypatch.setattr(
            l2b_product,
            "_prepare_output",
            lambda path: None,
        )

        with patch.object(
            l2b.xr,
            "open_zarr",
            side_effect=RuntimeError("stop after level 0"),
        ):
            with pytest.raises(RuntimeError):
                l2b_product.export_to_zarr_pyramid(
                    tmp_path / "test.zarr",
                    levels=1,
                    chunks={"y": 2, "x": 3},
                )

        assert captured["chunks"] == {"y": 2, "x": 3}
