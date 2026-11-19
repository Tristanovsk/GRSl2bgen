import logging
import os

import numpy as np
import pytest
import xarray as xr

from GRSl2bgen.product import Product, SPATIAL_CHUNK

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def raster():
    """Small in-memory L2A dataset."""
    return xr.Dataset(
        {
            "Rrs": (
                ("y", "x", "wl"),
                np.random.rand(4, 4, 3).astype(np.float32),
            ),
            "sza": (
                ("y", "x"),
                np.full((4, 4), 30.0, dtype=np.float32),
            ),
        },
        coords={
            "y": np.arange(4),
            "x": np.arange(4),
            "wl": [443, 560, 665],
        },
    )


@pytest.fixture
def netcdf_file(tmp_path, raster):
    """Write a small NetCDF product."""
    path = tmp_path / "product.nc"
    raster.to_netcdf(path)
    return str(path)

@pytest.fixture
def beam_raster():
    wls = [443, 560, 665]

    data = {
        f"Rrs_{wl}": (
            ("y", "x"),
            np.full((2, 2), wl / 10000, dtype=np.float32),
        )
        for wl in wls
    }

    data.update({
        f"Rrs_g_{wl}": (
            ("y", "x"),
            np.full((2, 2), wl / 20000, dtype=np.float32),
        )
        for wl in wls
    })

    data["sza"] = (
        ("y", "x"),
        np.full((2, 2), 30, dtype=np.float32),
    )

    return xr.Dataset(
        data,
        coords={
            "y": [0, 1],
            "x": [0, 1],
            "wl": wls,
        },
        attrs={"metadata_profile": "beam"},
    )

# ============================================================
# Tests for initialization
# ============================================================

class TestInitialization:
    
    def test_init_from_dataset(self,raster):
        product = Product(raster)

        assert product.raster is raster
        assert product.ancillary is None
        assert product.processor
        assert product._opened == []


    def test_init_from_dataset_does_not_rechunk(self,raster):
        product = Product(raster, spatial_chunk=2)

        assert product.raster is raster


    def test_output_is_not_defined(self,raster):
        product = Product(raster)

        assert not hasattr(product, "output")

# ============================================================
# Tests for netcdf
# ============================================================

class TestNetCDF:
    
    def test_init_from_netcdf(self,netcdf_file):
        product = Product(netcdf_file)

        assert isinstance(product.raster, xr.Dataset)
        assert "Rrs" in product.raster
        assert "sza" in product.raster
        assert product.ancillary is None
        assert len(product._opened) == 1

        product.close()


    def test_wavelengths_are_rechunked(self,netcdf_file):
        product = Product(netcdf_file)

        assert product.raster["Rrs"].chunks is not None
        assert product.raster["Rrs"].chunksizes["wl"] == (3,)

        product.close()


    def test_spatial_rechunking(self,netcdf_file):
        product = Product(netcdf_file, spatial_chunk=2)

        assert product.raster["Rrs"].chunksizes["y"] == (2, 2)
        assert product.raster["Rrs"].chunksizes["x"] == (2, 2)

        product.close()


    def test_netcdf_without_spatial_rechunking(self,netcdf_file):
        product = Product(netcdf_file, spatial_chunk=None)

        assert product.raster["Rrs"].chunksizes["wl"] == (3,)

        product.close()


    def test_custom_chunks(self,netcdf_file):
        product = Product(
            netcdf_file,
            chunks={"y": 2, "x": 2, "wl": 1},
        )

        assert product.raster["Rrs"].chunksizes["wl"] == (1, 1, 1)

        product.close()

# ============================================================
# Directory containing NetCDF and ancillary files
# ============================================================

def test_init_from_directory(tmp_path, raster):
    directory = tmp_path / "product"
    directory.mkdir()

    main_file = directory / "product.nc"
    ancillary_file = directory / "product_anc.nc"

    raster.to_netcdf(main_file)

    xr.Dataset(
        {
            "wind": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            )
        },
        coords={"y": np.arange(4), "x": np.arange(4)},
    ).to_netcdf(ancillary_file)

    product = Product(str(directory))

    assert isinstance(product.raster, xr.Dataset)
    assert isinstance(product.ancillary, xr.Dataset)
    assert "Rrs" in product.raster
    assert "wind" in product.ancillary
    assert len(product._opened) == 2

    product.close()


# ============================================================
# Unrecognized inputs
# ============================================================

def test_unrecognized_input(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="GRSl2bgen.product")
    unknown_path = tmp_path / "unknown.xyz"

    result = Product(str(unknown_path))

    assert not hasattr(result, "raster")
    assert "input file format not recognized" in caplog.text


def test_unrecognized_file(tmp_path):
    path = tmp_path / "unknown.txt"
    path.write_text("not a product")

    product = Product(str(path))

    assert not hasattr(product, "raster")


# ============================================================
# Tests for _rechunk
# ============================================================

def test_rechunk_none(raster):
    result = Product._rechunk(raster, None)

    assert result is raster


def test_rechunk_ignores_missing_dimensions(raster):
    result = Product._rechunk(
        raster,
        {"y": 2, "x": 2, "wl": -1, "time": 1},
    )

    assert result["Rrs"].chunksizes["y"] == (2, 2)
    assert result["Rrs"].chunksizes["x"] == (2, 2)
    assert result["Rrs"].chunksizes["wl"] == (3,)


def test_rechunk_empty_dimensions(raster):
    result = Product._rechunk(raster, {})

    assert result is raster


# ============================================================
# Zarr
# ============================================================

def test_init_from_regular_zarr(tmp_path, raster):
    path = tmp_path / "product.zarr"
    raster.to_zarr(path)

    product = Product(str(path))

    assert isinstance(product.raster, xr.Dataset)
    assert "Rrs" in product.raster
    assert product.ancillary is None
    assert len(product._opened) == 1

    product.close()


def test_zarr_does_not_apply_spatial_rechunking(tmp_path, raster):
    path = tmp_path / "product.zarr"
    raster.to_zarr(path)

    product = Product(str(path), spatial_chunk=2)

    # Zarr inputs are opened without an explicit chunks argument.
    assert isinstance(product.raster, xr.Dataset)

    product.close()


def test_init_from_store_directory(tmp_path, raster):
    product_dir = tmp_path / "product"
    product_dir.mkdir()

    zarr_path = product_dir / "product.zarr"
    raster.to_zarr(str(zarr_path), mode="w")

    result = Product(str(product_dir))

    xr.testing.assert_equal(result.raster, raster)
    assert result.ancillary is None

    result.close()



# ============================================================
# Multiscale Zarr pyramid
# ============================================================

def test_init_from_multiscale_zarr(tmp_path, raster):
    path = tmp_path / "pyramid.zarr"

    raster.to_zarr(path, group="level0")

    coarse = raster.coarsen(y=2, x=2, boundary="trim").mean()
    coarse.to_zarr(path, group="level1")

    # Add the multiscales metadata to the root group.
    import zarr

    root = zarr.open_group(str(path), mode="a")
    root.attrs["multiscales"] = [
        {
            "layout": [
                {"group": "level0"},
                {"group": "level1"},
            ]
        }
    ]

    product = Product(str(path), level=0)

    assert product.raster.sizes["y"] == 4
    assert product.raster.sizes["x"] == 4

    product.close()


def test_multiscale_zarr_selects_requested_level(tmp_path, raster):
    path = tmp_path / "pyramid.zarr"

    raster.to_zarr(path, group="level0")

    coarse = raster.coarsen(y=2, x=2, boundary="trim").mean()
    coarse.to_zarr(path, group="level1")

    import zarr

    root = zarr.open_group(str(path), mode="a")
    root.attrs["multiscales"] = [
        {
            "layout": [
                {"group": "level0"},
                {"group": "level1"},
            ]
        }
    ]

    product = Product(str(path), level=1)

    assert product.raster.sizes["y"] == 2
    assert product.raster.sizes["x"] == 2

    product.close()


def test_multiscale_zarr_invalid_level(tmp_path, raster):
    path = tmp_path / "pyramid.zarr"

    raster.to_zarr(path, group="level0")

    import zarr

    root = zarr.open_group(str(path), mode="a")
    root.attrs["multiscales"] = [
        {
            "layout": [
                {"group": "level0"},
            ]
        }
    ]

    with pytest.raises(ValueError, match="pyramid level 1 not in"):
        Product(str(path), level=1)


def test_multiscale_zarr_negative_level(tmp_path, raster):
    path = tmp_path / "pyramid.zarr"

    raster.to_zarr(path, group="level0")

    import zarr

    root = zarr.open_group(str(path), mode="a")
    root.attrs["multiscales"] = [
        {
            "layout": [
                {"group": "level0"},
            ]
        }
    ]

    with pytest.raises(ValueError, match="pyramid level -1 not in"):
        Product(str(path), level=-1)


def test_multiscale_zarr_with_ancillary(tmp_path, raster):
    path = tmp_path / "pyramid.zarr"

    raster.to_zarr(path, group="level0")

    ancillary = xr.Dataset(
        {
            "wind": (
                ("y", "x"),
                np.ones((4, 4), dtype=np.float32),
            )
        },
        coords={"y": np.arange(4), "x": np.arange(4)},
    )

    ancillary.to_zarr(path, group="ancillary")

    import zarr

    root = zarr.open_group(str(path), mode="a")
    root.attrs["multiscales"] = [
        {
            "layout": [
                {"group": "level0"},
            ]
        }
    ]

    product = Product(str(path))

    assert product.ancillary is not None
    assert "wind" in product.ancillary
    assert len(product._opened) == 2

    product.close()

# ============================================================
# Legacy BEAM metadata profile
# ============================================================

def test_beam_profile_reshapes_rrs(beam_raster, tmp_path):
    path = tmp_path / "beam.nc"
    beam_raster.to_netcdf(path)

    product = Product(str(path))

    assert "Rrs" in product.raster
    assert "Rrs_g" in product.raster

    assert product.raster["Rrs"].dims == ("wl", "y", "x")
    assert product.raster["Rrs_g"].dims == ("wl", "y", "x")

    assert list(product.raster.wl.values) == [443, 560, 665]

    product.close()


def test_beam_profile_preserves_other_variables(beam_raster, tmp_path):
    path = tmp_path / "beam.nc"
    beam_raster.to_netcdf(path)

    product = Product(str(path))

    assert "sza" in product.raster

    product.close()


def test_beam_profile_removes_individual_band_variables(beam_raster, tmp_path):
    path = tmp_path / "beam.nc"
    beam_raster.to_netcdf(path)

    product = Product(str(path))

    for wl in [443, 560, 665]:
        assert f"Rrs_{wl}" not in product.raster
        assert f"Rrs_g_{wl}" not in product.raster

    product.close()


def test_non_beam_profile_is_not_reshaped(raster, tmp_path):
    raster.attrs["metadata_profile"] = "other"

    path = tmp_path / "product.nc"
    raster.to_netcdf(path)

    product = Product(str(path))

    assert "Rrs" in product.raster
    assert product.raster["Rrs"].dims == ("y", "x", "wl")

    product.close()


# ============================================================
# Closing ressources
# ============================================================

def test_close_clears_opened_datasets(netcdf_file):
    product = Product(netcdf_file)

    assert len(product._opened) == 1

    product.close()

    assert product._opened == []


def test_close_can_be_called_twice(netcdf_file):
    product = Product(netcdf_file)

    product.close()
    product.close()

    assert product._opened == []


def test_close_does_not_close_in_memory_dataset(raster):
    product = Product(raster)

    product.close()

    assert product.raster is raster

    # The original dataset remains usable.
    assert "Rrs" in raster
