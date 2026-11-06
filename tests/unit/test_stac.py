import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pystac
import pytest
import xarray as xr
import rioxarray

from rasterio.transform import from_origin
from GRSl2bgen.stac import (
    STAC_EXTENSIONS,
    MEDIA_TYPES,
    item_path,
    _file_info,
    _open,
    _footprint,
    _datetime,
    make_stac_item,
    write_stac_item,
)

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_dataset(tmp_path):
    """Create a synthetic georeferenced L2B NetCDF product."""

    y = np.arange(3)
    x = np.arange(4)

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.random.default_rng(42).random((3, 4)),
            ),
        },
        coords={
            "y": y,
            "x": x,
        },
        attrs={
            "constellation": "Sentinel-2",
            "DATATAKE_1_SPACECRAFT_NAME": "SENTINEL-2A",
            "instrument": "MSI",
            "tile": "31TFJ",
            "version": "1.0",
            "acquisition_date": "2021-02-11T10:41:51",
            "processing_time": "2021-02-11T12:00:00",
            "CLOUD_COVERAGE_ASSESSMENT": 12.5,
        },
    )

    ds = ds.rio.set_spatial_dims(
        x_dim="x",
        y_dim="y",
    )

    ds = ds.rio.write_crs("EPSG:32631")

    ds = ds.rio.write_transform(
        from_origin(500000, 4800000, 10, 10)
    )

    path = tmp_path / "l2b.nc"
    ds.to_netcdf(path)
    ds.close()

    return path

@pytest.fixture
def sample_l2a(tmp_path):
    """Create a dummy L2A input path."""

    path = tmp_path / "l2a_product"
    path.mkdir()

    return str(path)

@pytest.fixture
def sample_log(tmp_path):
    """Create a processing log."""

    path = tmp_path / "l2b.log"
    path.write_text("Processing completed successfully\n")

    return path

@pytest.fixture
def sample_dataset_without_optional_attrs(tmp_path):
    """Create a dataset without optional metadata attributes."""

    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((2, 2)),
            ),
        },
        attrs={
            "acquisition_date": "2022-01-01T12:00:00",
        },
    )

    ds = ds.rio.set_spatial_dims(
        x_dim="x",
        y_dim="y",
    )

    ds = ds.rio.write_crs("EPSG:4326")

    ds = ds.rio.write_transform(
        from_origin(4.0, 44.0, 0.01, 0.01)
    )

    path = tmp_path / "minimal.nc"
    ds.to_netcdf(path)
    ds.close()

    return path

@pytest.fixture
def spatial_dataset():
    """Small spatial dataset with a valid UTM CRS."""
    ds = xr.Dataset(
        {
            "Rrs": (
                ("y", "x"),
                np.ones((4, 4)),
            )
        },
        coords={
            "x": np.arange(500000, 500400, 100),
            "y": np.arange(4800000, 4800400, 100),
        },
    )

    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_crs("EPSG:32631")

    return ds

# ============================================================
# Tests for item_path
# ============================================================

class TestItemPath:
    
    def test_nc(self):
        assert item_path("/data/l2b.nc") == "/data/l2b.json"

    def test_zarr(self):
        assert item_path("/data/l2b.zarr") == "/data/l2b.json"

    def test_trailing_slash(self):
        assert item_path("/data/l2b.zarr/") == "/data/l2b.json"

    def test_trailing_backslash(self):
        assert item_path(r"C:\data\l2b.zarr\\") == r"C:\data\l2b.json"

    def test_pathlib(self,tmp_path):
        path = tmp_path / "product.nc"

        assert item_path(path) == str(tmp_path / "product.json")


# ============================================================
# Tests for _file_info
# ============================================================

class TestFileInfo:
    
    def test_nominal(self,tmp_path):
        path = tmp_path / "test.txt"
        content = b"hello world"

        path.write_bytes(content)

        result = _file_info(path)

        expected_hash = hashlib.sha256(content).hexdigest()

        assert result["file:size"] == len(content)
        assert result["file:checksum"] == "1220" + expected_hash


    def test_empty_file(self,tmp_path):
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")

        result = _file_info(path)

        assert result["file:size"] == 0
        assert result["file:checksum"] == (
            "1220" + hashlib.sha256(b"").hexdigest()
        )


    def test_large_file(self,tmp_path):
        path = tmp_path / "large.bin"
        content = os.urandom((1 << 20) + 100)

        path.write_bytes(content)

        result = _file_info(path)

        assert result["file:size"] == len(content)
        assert result["file:checksum"] == (
            "1220" + hashlib.sha256(content).hexdigest()
        )


    def test_missing_file(self,tmp_path):
        path = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            _file_info(path)


# ============================================================
# Tests for _datetime
# ============================================================

class TestDatetime:
    
    def test_without_timezone(self):
        result = _datetime("2021-02-11T10:41:51")

        assert isinstance(result, datetime.datetime)
        assert result.year == 2021
        assert result.month == 2
        assert result.day == 11


    def test_with_z(self):
        result = _datetime("2021-02-11T10:41:51Z")

        assert result.tzinfo is not None
        assert result.utcoffset() == datetime.timedelta(0)


    def test_with_timezone(self):
        result = _datetime("2021-02-11T12:00:00+02:00")

        assert result.utcoffset() == datetime.timedelta(hours=2)


    def test_datetime_object(self):
        value = datetime.datetime(2021, 2, 11, 10, 41, 51)

        result = _datetime(value)

        assert result == value


    def test_invalid(self):
        with pytest.raises(ValueError):
            _datetime("not-a-date")


# ============================================================
# Tests for _open
# ============================================================

class TestOpen:
    
    def test_netcdf(self,sample_dataset):
        ds, root = _open(sample_dataset)

        try:
            assert isinstance(ds, xr.Dataset)
            assert root is None
            assert "Rrs" in ds
            assert ds.rio.crs.to_string() == "EPSG:32631"
        finally:
            ds.close()


    def test_netcdf_preserves_attrs(self,sample_dataset):
        ds, root = _open(sample_dataset)

        try:
            assert root is None
            assert ds.attrs["constellation"] == "Sentinel-2"
            assert ds.attrs["tile"] == "31TFJ"
        finally:
            ds.close()


    def test_missing_netcdf(self,tmp_path):
        with pytest.raises(FileNotFoundError):
            _open(tmp_path / "missing.nc")


    def test_zarr_with_root_data_vars(self,tmp_path):
        pytest.importorskip("zarr")

        path = tmp_path / "product.zarr"

        ds = xr.Dataset(
            {
                "Rrs": (
                    ("y", "x"),
                    np.ones((2, 2)),
                ),
            },
            attrs={"test_attribute": "root"},
        )

        ds.to_zarr(path)

        opened, root = _open(path)

        try:
            assert isinstance(opened, xr.Dataset)
            assert root is None
            assert "Rrs" in opened
        finally:
            opened.close()


    def test_zarr_with_level_zero(self,tmp_path):
        pytest.importorskip("zarr")

        path = tmp_path / "pyramid.zarr"

        root_ds = xr.Dataset(
            attrs={"test_attribute": "root"},
        )

        level_ds = xr.Dataset(
            {
                "Rrs": (
                    ("y", "x"),
                    np.ones((2, 2)),
                ),
            },
        )

        root_ds.to_zarr(path)
        level_ds.to_zarr(path, group="0")

        opened, root = _open(path)

        try:
            assert isinstance(opened, xr.Dataset)
            assert isinstance(root, xr.Dataset)
            assert "Rrs" in opened
            assert root.attrs["test_attribute"] == "root"
        finally:
            opened.close()
            root.close()


# ============================================================
# Tests for _footprint
# ============================================================

class TestFootprint:
    
    def test_geometry(self,spatial_dataset):
        geometry, bbox = _footprint(spatial_dataset)

        assert geometry["type"] == "Polygon"
        assert len(geometry["coordinates"]) == 1

        ring = geometry["coordinates"][0]

        # Closed polygon
        assert ring[0] == ring[-1]

        # Four corners plus closing point
        assert len(ring) == 5

        assert len(bbox) == 4
        assert bbox[0] < bbox[2]
        assert bbox[1] < bbox[3]

    def test_coordinates_are_rounded(self,spatial_dataset):
        geometry, bbox = _footprint(spatial_dataset)

        for lon, lat in geometry["coordinates"][0]:
            assert round(lon, 7) == lon
            assert round(lat, 7) == lat

        for value in bbox:
            assert round(value, 7) == value

    @pytest.mark.parametrize("npts", [2, 3, 5, 10])
    def test_custom_number_of_points(self,spatial_dataset, npts):
        geometry, bbox = _footprint(spatial_dataset, npts=npts)

        ring = geometry["coordinates"][0]

        assert len(ring) == 4 * (npts - 1) + 1
        assert ring[0] == ring[-1]

        assert len(bbox) == 4

# ============================================================
# Tests for make_stac_item
# ============================================================

class TestMakeStacItem:

    def test_basic(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert isinstance(item, pystac.Item)

        assert item.id == "l2b"
        assert item.datetime == datetime.datetime(2021, 2, 11, 10, 41, 51)

        assert item.properties["processing:level"] == "L2B"
        assert item.properties["proj:code"] == "EPSG:32631"
        assert item.properties["proj:shape"] == [3, 4]

    def test_extensions(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.stac_extensions == STAC_EXTENSIONS

    def test_constellation(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.properties["constellation"] == "sentinel-2"

    def test_platform(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.properties["platform"] == "sentinel-2a"

    def test_instruments(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.properties["instruments"] == ["msi"]

    def test_projection_properties(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.properties["proj:code"] == "EPSG:32631"
        assert item.properties["proj:shape"] == [3, 4]
        assert len(item.properties["proj:transform"]) == 6
        assert len(item.properties["proj:bbox"]) == 4

    def test_processing_properties(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.properties["processing:level"] == "L2B"
        assert "GRSl2bgen" in item.properties["processing:software"]

        assert item.properties["processing:software"]["GRS"] == "1.0"

        assert item.properties["processing:datetime"] is not None

    def test_processing_datetime_utc(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        result = item.properties["processing:datetime"]

        expected = _datetime("2021-02-11T12:00:00").astimezone(
            datetime.timezone.utc
        ).isoformat()

        assert result == expected

    def test_lineage(self,sample_dataset, sample_l2a):
        item = make_stac_item(
            sample_dataset,
            l2a=sample_l2a,
        )

        assert item.properties["processing:lineage"] is not None
        assert "l2a_product" in item.properties["processing:lineage"]

    def test_grid_code(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.properties["grid:code"] == "MGRS-31TFJ"

    def test_cloud_cover(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.properties["eo:cloud_cover"] == 12.5

    def test_data_asset(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        asset = item.assets["data"]

        assert asset.href == os.path.abspath(sample_dataset)
        assert asset.roles == ["data"]
        assert asset.media_type == MEDIA_TYPES[".nc"]
        assert "L2B" in asset.title

        assert asset.extra_fields["file:size"] == os.path.getsize(sample_dataset)
        assert asset.extra_fields["file:checksum"].startswith("1220")

    def test_log_asset(self,sample_dataset, sample_log):
        item = make_stac_item(
            sample_dataset,
            log_file=sample_log,
        )

        assert "log" in item.assets

        asset = item.assets["log"]

        assert asset.href == os.path.abspath(sample_log)
        assert asset.roles == ["metadata"]
        assert asset.media_type == "text/plain"
        assert asset.extra_fields["file:size"] == len(
            "Processing completed successfully\n"
        )

    def test_missing_log(self,sample_dataset, tmp_path):
        item = make_stac_item(
            sample_dataset,
            log_file=tmp_path / "missing.log",
        )

        assert "log" not in item.assets

    def test_derived_from_link(self,sample_dataset, sample_l2a):
        item = make_stac_item(
            sample_dataset,
            l2a=sample_l2a,
        )

        links = item.get_links("derived_from")

        assert len(links) == 1
        assert links[0].target == os.path.abspath(sample_l2a)

    def test_without_l2a(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.get_links("derived_from") == []
        assert "processing:lineage" not in item.properties

    def test_optional_properties(self,sample_dataset_without_optional_attrs):
        item = make_stac_item(sample_dataset_without_optional_attrs)

        assert "constellation" not in item.properties
        assert "platform" not in item.properties
        assert "instruments" not in item.properties
        assert "grid:code" not in item.properties
        assert "eo:cloud_cover" not in item.properties
        assert "processing:datetime" not in item.properties
        assert "processing:lineage" not in item.properties

    def test_geometry(self,sample_dataset):
        item = make_stac_item(sample_dataset)

        assert item.geometry["type"] == "Polygon"
        assert item.geometry["coordinates"][0][0] == (
            item.geometry["coordinates"][0][-1]
        )

        assert len(item.bbox) == 4
        assert item.bbox[0] < item.bbox[2]
        assert item.bbox[1] < item.bbox[3]

    def test_zarr_without_file_checksum(self,tmp_path):
        pytest.importorskip("zarr")

        path = tmp_path / "product.zarr"

        ds = xr.Dataset(
            {
                "Rrs": (
                    ("y", "x"),
                    np.ones((2, 2)),
                ),
            },
            attrs={
                "acquisition_date": "2021-02-11T10:41:51",
            },
        )

        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.write_transform(
            from_origin(4, 44, 0.01, 0.01)
        )

        ds.to_zarr(path)

        item = make_stac_item(path)

        assert item.assets["data"].media_type == MEDIA_TYPES[".zarr"]
        assert "file:size" not in item.assets["data"].extra_fields
        assert "file:checksum" not in item.assets["data"].extra_fields

# ============================================================
# Tests for write_stac_item
# ============================================================

class TestWriteStacItem:
    
    def test_creates_file(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        assert os.path.isfile(result)
        assert result == os.path.abspath(item_path(sample_dataset))

    def test_valid_json(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        with open(result) as f:
            data = json.load(f)

        assert data["type"] == "Feature"
        assert data["id"] == "l2b"

    def test_relative_data_href(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        with open(result) as f:
            data = json.load(f)

        assert data["assets"]["data"]["href"] == "./l2b.nc"

    def test_relative_log_href(self,sample_dataset, sample_log):
        result = write_stac_item(
            sample_dataset,
            log_file=sample_log,
        )

        with open(result) as f:
            data = json.load(f)

        assert data["assets"]["log"]["href"] == "./l2b.log"

    def test_custom_path(self,sample_dataset, tmp_path):
        custom_path = tmp_path / "metadata" / "custom.json"
        custom_path.parent.mkdir()

        result = write_stac_item(
            sample_dataset,
            item_file=custom_path,
        )

        assert result == str(custom_path.resolve())
        assert custom_path.is_file()

    def test_derived_from_absolute(self,sample_dataset, sample_l2a):
        result = write_stac_item(
            sample_dataset,
            l2a=sample_l2a,
        )

        with open(result) as f:
            data = json.load(f)

        link = next(
            link for link in data["links"]
            if link["rel"] == "derived_from"
        )

        assert link["href"] == os.path.abspath(sample_l2a)

    def test_no_self_link(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        with open(result) as f:
            data = json.load(f)

        assert not any(
            link.get("rel") == "self"
            for link in data["links"]
        )

    def test_readable_by_pystac(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        item = pystac.Item.from_file(result)

        assert item.id == "l2b"
        assert "data" in item.assets
        assert item.assets["data"].get_absolute_href() == (
            os.path.abspath(sample_dataset)
        )

    def test_atomic_no_temporary_file(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        directory = os.path.dirname(result)

        assert os.path.isfile(result)

        temporary_files = [
            name for name in os.listdir(directory)
            if name.startswith(".l2b.json.part")
        ]

        assert temporary_files == []

    def test_overwrites_existing_item(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        with open(result, "w") as f:
            f.write("old content")

        write_stac_item(sample_dataset)

        with open(result) as f:
            data = json.load(f)

        assert data["id"] == "l2b"


    def test_returns_absolute_path(self,sample_dataset):
        result = write_stac_item(sample_dataset)

        assert os.path.isabs(result)

    def test_custom_name(self,sample_dataset, tmp_path):
        custom_path = tmp_path / "different_name.json"

        result = write_stac_item(
            sample_dataset,
            item_file=custom_path,
        )

        assert result == str(custom_path.resolve())

        with open(result) as f:
            data = json.load(f)

        # Item ID comes from the product filename, not the JSON filename.
        assert data["id"] == "l2b"