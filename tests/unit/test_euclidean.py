from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import xarray as xr
import dask.array as da
from GRSl2bgen.euclidean import _euclid_core, _check_classes, euclidean, euclidean_dataarray

# ============================================================
# Tests for _euclid_core
# ============================================================

class TestEuclidCore:
    
    def test_distances(self):
        Rrs = np.array([
            [[1.0, 1.0]],
        ])

        owt = np.array([
            [2.0, 1.0],  # distance = 1
            [1.0, 3.0],  # distance = 2
            [4.0, 5.0],  # distance = 5
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=3)

        np.testing.assert_allclose(
            dist[0, 0],
            [1.0, 2.0, 5.0],
        )

        np.testing.assert_array_equal(
            index[0, 0],
            [1.0, 2.0, 3.0],
        )

        assert dist.dtype == np.float32
        assert index.dtype == np.float32

    def test_nsaved(self):
        Rrs = np.array([
            [[1.0, 1.0]],
        ])

        owt = np.array([
            [2.0, 1.0],
            [1.0, 3.0],
            [4.0, 5.0],
            [1.0, 6.0],
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=2)

        assert dist.shape == (1, 1, 2)
        assert index.shape == (1, 1, 2)

        np.testing.assert_allclose(
            dist[0, 0],
            [1.0, 2.0],
        )

        np.testing.assert_array_equal(
            index[0, 0],
            [1.0, 2.0],
        )

    def test_multiple_pixels(self):
        Rrs = np.array([
            [
                [1.0, 1.0],
                [5.0, 5.0],
            ],
            [
                [2.0, 2.0],
                [3.0, 3.0],
            ],
        ])

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=1)

        expected_dist = np.array([
            [0.0, 0.0],
            [0.0, np.sqrt(2)],
        ])

        expected_index = np.array([
            [1.0, 3.0],
            [2.0, 2.0],
        ])

        np.testing.assert_allclose(
            dist[..., 0],
            expected_dist,
        )

        np.testing.assert_array_equal(
            index[..., 0],
            expected_index,
        )

    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_invalid_pixels(self,invalid_value):

        Rrs = np.array([
            [
                [1.0, 1.0],
                [invalid_value, 1.0],
            ],
            [
                [1.0, invalid_value],
                [2.0, 2.0],
            ],
        ])

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=2)

        # Valid pixels
        assert np.isfinite(dist[0, 0]).all()
        assert np.isfinite(index[0, 0]).all()
        assert np.isfinite(dist[1, 1]).all()
        assert np.isfinite(index[1, 1]).all()

        # Invalid pixels
        assert np.isnan(dist[0, 1]).all()
        assert np.isnan(index[0, 1]).all()
        assert np.isnan(dist[1, 0]).all()
        assert np.isnan(index[1, 0]).all()

    def test_nsaved_greater_than_nclasses(self):

        Rrs = np.array([
            [[1.0, 1.0]],
        ])

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=4)

        assert dist.shape == (1, 1, 4)
        assert index.shape == (1, 1, 4)

        np.testing.assert_allclose(
            dist[0, 0, :2],
            [0.0, np.sqrt(2)],
        )

        np.testing.assert_array_equal(
            index[0, 0, :2],
            [1.0, 2.0],
        )

        assert np.isnan(dist[0, 0, 2:]).all()
        assert np.isnan(index[0, 0, 2:]).all()

    def test_nsaved_one(self):

        Rrs = np.array([
            [[3.0, 4.0]],
        ])

        owt = np.array([
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=1)

        assert dist.shape == (1, 1, 1)
        assert index.shape == (1, 1, 1)

        assert dist[0, 0, 0] == 0.0
        assert index[0, 0, 0] == 2.0

    def test_one_based_indices(self):

        Rrs = np.array([
            [[10.0, 10.0]],
        ])

        owt = np.array([
            [0.0, 0.0],
            [10.0, 10.0],
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=2)

        np.testing.assert_array_equal(
            index[0, 0],
            [2.0, 1.0],
        )

    def test_equal_distances(self):

        Rrs = np.array([
            [[0.0, 0.0]],
        ])

        owt = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ])

        dist, index = _euclid_core(Rrs, owt, Nsaved=3)

        np.testing.assert_allclose(
            dist[0, 0],
            [1.0, 1.0, 1.0],
        )

        # Ties are resolved using the original class ordering.
        np.testing.assert_array_equal(
            index[0, 0],
            [1.0, 2.0, 3.0],
        )

# ============================================================
# Tests for _check_classes
# ============================================================

class TestCheckClasses:
    
    def test_valid(self):
        Rrs = np.ones((4,2,2), dtype=np.float64)
        
        result = _check_classes(Rrs_owt=Rrs)
        np.testing.assert_array_equal(result, Rrs)

    def test_invalid(self):
        Rrs = np.full((4,2,2), np.nan)
        
        with pytest.raises(ValueError,  match="OWT spectra must be finite"):
            result = _check_classes(Rrs_owt=Rrs)

# ============================================================
# Tests for euclidean
# ============================================================

class TestEuclidean:

    @pytest.mark.parametrize("parallel", [True, False])
    def test_output(self,parallel):

        Rrs = np.array([
            [
                [1.0, 1.0],
                [5.0, 5.0],
            ],
            [
                [2.0, 2.0],
                [3.0, 3.0],
            ],
        ])

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
        ])

        dist, index = euclidean(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            parallel=parallel,
        )

        assert dist.shape == (2, 2, 2)
        assert index.shape == (2, 2, 2)

        assert dist.dtype == np.float32
        assert index.dtype == np.float32

        expected_dist = np.array([
            [
                [0.0, 0.0],
                [0.0, np.sqrt(2)],
            ],
            [
                [np.sqrt(2), np.sqrt(18)],
                [np.sqrt(2), np.sqrt(8)],
            ],
        ])

        expected_index = np.array([
            [
                [1.0, 3.0],
                [2.0, 2.0],
            ],
            [
                [2.0, 2.0],
                [1.0, 1.0],
            ],
        ])


        np.testing.assert_allclose(dist, expected_dist)
        np.testing.assert_array_equal(index, expected_index)

    def test_nclasses_to_be_saved(self):

        Rrs = np.array([
            [[1.0, 1.0]],
        ])

        owt = np.array([
            [2.0, 1.0],
            [1.0, 3.0],
            [4.0, 5.0],
            [1.0, 6.0],
        ])

        dist, index = euclidean(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            parallel=False,
        )

        assert dist.shape == (2, 1, 1)
        assert index.shape == (2, 1, 1)

        np.testing.assert_allclose(
            dist[:, 0, 0],
            [1.0, 2.0],
        )

        np.testing.assert_array_equal(
            index[:, 0, 0],
            [1.0, 2.0],
        )

    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_invalid_pixels(self,invalid_value):

        Rrs = np.array([
            [
                [1.0, 1.0],
                [invalid_value, 1.0],
            ],
            [
                [1.0, invalid_value],
                [2.0, 2.0],
            ],
        ])

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = euclidean(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            parallel=False,
        )

        # Valid pixels
        assert np.isfinite(dist[:, 0, 0]).all()
        assert np.isfinite(index[:, 0, 0]).all()
        assert np.isfinite(dist[:, 1, 1]).all()
        assert np.isfinite(index[:, 1, 1]).all()

        # Invalid pixels
        assert np.isnan(dist[:, 0, 1]).all()
        assert np.isnan(index[:, 0, 1]).all()
        assert np.isnan(dist[:, 1, 0]).all()
        assert np.isnan(index[:, 1, 0]).all()

    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_invalid_owt(self,invalid_value):

        Rrs = np.ones((2, 2, 3))

        owt = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])

        owt[0, 1] = invalid_value

        with pytest.raises(ValueError, match="OWT spectra must be finite"):
            euclidean(Rrs, owt)

    def test_invalid_rrs_dimensions(self):

        Rrs = np.ones((2, 3))

        owt = np.ones((2, 3))

        with pytest.raises(ValueError, match="Rrs must have shape"):
            euclidean(Rrs, owt)


    def test_mismatched_wavelengths(self):

        Rrs = np.ones((2, 2, 4))

        owt = np.ones((3, 3))

        with pytest.raises(ValueError, match="Rrs must have shape"):
            euclidean(Rrs, owt)

    def test_nsaved_greater_than_nclasses(self):

        Rrs = np.array([
            [[1.0, 1.0]],
        ])

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = euclidean(
            Rrs,
            owt,
            Nclasses_to_be_saved=4,
            parallel=False,
        )

        assert dist.shape == (4, 1, 1)
        assert index.shape == (4, 1, 1)

        np.testing.assert_allclose(
            dist[:2, 0, 0],
            [0.0, np.sqrt(2)],
        )

        np.testing.assert_array_equal(
            index[:2, 0, 0],
            [1.0, 2.0],
        )

        assert np.isnan(dist[2:, 0, 0]).all()
        assert np.isnan(index[2:, 0, 0]).all()

    def test_non_contiguous_input(self):

        Rrs = np.ones((2, 4, 4))[:, ::2, :]

        assert not Rrs.flags["C_CONTIGUOUS"]

        owt = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ])

        dist, index = euclidean(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            parallel=False,
        )

        assert dist.shape == (2, 2, 2)
        assert index.shape == (2, 2, 2)

        np.testing.assert_allclose(
            dist[0],
            0.0,
        )

        np.testing.assert_array_equal(
            index[0],
            1.0,
        )

    def test_parallel_vs_serial(self):

        rng = np.random.default_rng(42)

        Rrs = rng.random((10, 20, 4))
        owt = rng.random((8, 4))

        dist_parallel, index_parallel = euclidean(
            Rrs,
            owt,
            Nclasses_to_be_saved=3,
            parallel=True,
        )

        dist_serial, index_serial = euclidean(
            Rrs,
            owt,
            Nclasses_to_be_saved=3,
            parallel=False,
        )

        np.testing.assert_allclose(
            dist_parallel,
            dist_serial,
        )

        np.testing.assert_array_equal(
            index_parallel,
            index_serial,
        )


# ============================================================
# Tests for euclidean_dataarray
# ============================================================

class TestEuclideanDataArray:

    def test_output(self):

        Rrs = xr.DataArray(
            np.array([
                [
                    [1.0, 1.0],
                    [5.0, 5.0],
                ],
                [
                    [2.0, 2.0],
                    [3.0, 3.0],
                ],
            ]),
            dims=("y", "x", "wl"),
            coords={
                "y": [0, 1],
                "x": [0, 1],
                "wl": [443, 490],
            },
            name="Rrs",
        )

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
        ])

        dist, index = euclidean_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
        )

        assert isinstance(dist, xr.DataArray)
        assert isinstance(index, xr.DataArray)

        assert dist.name == "owt_dist"
        assert index.name == "owt_index"

        assert dist.dims == ("Nowt", "y", "x")
        assert index.dims == ("Nowt", "y", "x")

        assert dist.shape == (2, 2, 2)
        assert index.shape == (2, 2, 2)

        assert dist.dtype == np.float32
        assert index.dtype == np.float32

        expected_dist = np.array([
            [
                [0.0, 0.0],
                [0.0, np.sqrt(2)],
            ],
            [
                [np.sqrt(2), np.sqrt(18)],
                [np.sqrt(2), np.sqrt(8)],
            ],
        ])

        expected_index = np.array([
            [
                [1.0, 3.0],
                [2.0, 2.0],
            ],
            [
                [2.0, 2.0],
                [1.0, 1.0],
            ],
        ])

        np.testing.assert_allclose(dist.values, expected_dist)
        np.testing.assert_array_equal(index.values, expected_index)

    @pytest.mark.parametrize(
        "dims",
        [
            ("y", "x", "wl"),
            ("wl", "y", "x"),
            ("x", "wl", "y"),
        ],
    )
    def test_dimension_order(self,dims):

        data = np.array([
            [
                [1.0, 1.0],
                [5.0, 5.0],
            ],
            [
                [2.0, 2.0],
                [3.0, 3.0],
            ],
        ])

        Rrs = xr.DataArray(
            data,
            dims=("y", "x", "wl"),
            coords={
                "y": [10, 20],
                "x": [30, 40],
                "wl": [443, 490],
            },
        ).transpose(*dims)

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
        ])

        dist, index = euclidean_dataarray(Rrs, owt)

        expected_dims = ("Nowt",) + tuple(d for d in dims if d != "wl")

        assert dist.dims == expected_dims
        assert index.dims == expected_dims


        assert dist.shape == (3, 2, 2)
        assert index.shape == (3, 2, 2)

        # Spatial coordinates must be preserved.
        np.testing.assert_array_equal(dist.y, [10, 20])
        np.testing.assert_array_equal(dist.x, [30, 40])

        np.testing.assert_array_equal(index.y, [10, 20])
        np.testing.assert_array_equal(index.x, [30, 40])

    def test_custom_dimensions(self):

        Rrs = xr.DataArray(
            np.ones((2, 3, 2)),
            dims=("latitude", "longitude", "wavelength"),
            coords={
                "latitude": [10, 20],
                "longitude": [30, 40, 50],
                "wavelength": [443, 490],
            },
        )

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = euclidean_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            wl_dim="wavelength",
            name_dim="class",
        )

        assert dist.dims == ("class", "latitude", "longitude")
        assert index.dims == ("class", "latitude", "longitude")

        assert dist.shape == (2, 2, 3)
        assert index.shape == (2, 2, 3)

        assert dist.name == "owt_dist"
        assert index.name == "owt_index"

    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_invalid_pixels(self,invalid_value):

        Rrs = xr.DataArray(
            [
                [
                    [1.0, 1.0],
                    [invalid_value, 1.0],
                ],
                [
                    [1.0, invalid_value],
                    [2.0, 2.0],
                ],
            ],
            dims=("y", "x", "wl"),
            coords={
                "y": [0, 1],
                "x": [0, 1],
                "wl": [443, 490],
            },
        )

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = euclidean_dataarray(Rrs, owt, Nclasses_to_be_saved=2,)

        assert np.isnan(dist.sel(y=0, x=1)).all()
        assert np.isnan(index.sel(y=0, x=1)).all()

        assert np.isnan(dist.sel(y=1, x=0)).all()
        assert np.isnan(index.sel(y=1, x=0)).all()

        assert np.isfinite(dist.sel(y=0, x=0)).all()
        assert np.isfinite(index.sel(y=0, x=0)).all()

    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_invalid_owt(self,invalid_value):

        Rrs = xr.DataArray(
            np.ones((2, 2, 3)),
            dims=("y", "x", "wl"),
        )

        owt = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])

        owt[0, 1] = invalid_value

        with pytest.raises(ValueError, match="OWT spectra must be finite"):
            euclidean_dataarray(Rrs, owt)

    def test_invalid_dimensions(self):

        Rrs = xr.DataArray(
            np.ones((2, 3)),
            dims=("y", "x"),
        )

        owt = np.ones((2, 3))

        with pytest.raises(
            ValueError,
            match="Rrs must have exactly three dimensions",
        ):
            euclidean_dataarray(Rrs, owt)


    def test_mismatched_wavelengths(self):

        Rrs = xr.DataArray(
            np.ones((2, 2, 4)),
            dims=("y", "x", "wl"),
        )

        owt = np.ones((3, 3))

        with pytest.raises(
            ValueError,
            match="different number of wavelengths",
        ):
            euclidean_dataarray(Rrs, owt)

    def test_nsaved_greater_than_nclasses(self):

        Rrs = xr.DataArray(
            [[[1.0, 1.0]]],
            dims=("y", "x", "wl"),
            coords={"wl": [443, 490]},
        )

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = euclidean_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=4,
        )

        assert dist.shape == (4, 1, 1)
        assert index.shape == (4, 1, 1)

        np.testing.assert_allclose(
            dist.values[:2, 0, 0],
            [0.0, np.sqrt(2)],
        )

        np.testing.assert_array_equal(
            index.values[:2, 0, 0],
            [1.0, 2.0],
        )

        assert np.isnan(dist.values[2:]).all()
        assert np.isnan(index.values[2:]).all()

    def test_dask(self):

        data = np.array([
            [
                [1.0, 1.0, 2.0, 2.0],
                [5.0, 5.0, 4.0, 4.0],
            ],
            [
                [2.0, 2.0, 3.0, 3.0],
                [3.0, 3.0, 1.0, 1.0],
            ],
        ])

        Rrs = xr.DataArray(
            data,
            dims=("y", "x", "wl"),
            coords={"wl": [443, 490, 560, 665]},
        )

        Rrs_dask = Rrs.chunk({
            "y": 1,
            "x": 1,
            "wl": 2,
        })

        owt = np.array([
            [1.0, 1.0, 2.0, 2.0],
            [2.0, 2.0, 3.0, 3.0],
            [5.0, 5.0, 4.0, 4.0],
        ])

        dist_dask, index_dask = euclidean_dataarray(
            Rrs_dask,
            owt,
            Nclasses_to_be_saved=2,
        )

        # Results must remain lazy.
        assert isinstance(dist_dask.data, da.Array)
        assert isinstance(index_dask.data, da.Array)

        # Wavelength must have been rechunked into a single chunk.
        assert Rrs_dask.chunks[-1] == (2, 2)
        assert dist_dask.chunks is not None

        # Compute results.
        dist_computed = dist_dask.compute()
        index_computed = index_dask.compute()

        # Compare with NumPy-backed implementation.
        dist_numpy, index_numpy = euclidean_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
        )

        np.testing.assert_allclose(
            dist_computed.values,
            dist_numpy.values,
        )

        np.testing.assert_array_equal(
            index_computed.values,
            index_numpy.values,
        )

    @pytest.mark.parametrize(
        "nsaved",
        [1, 2, 3],
    )
    def test_class_dimension(self,nsaved):

        Rrs = xr.DataArray(
            np.ones((2, 2, 3)),
            dims=("y", "x", "wl"),
        )

        owt = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ])

        dist, index = euclidean_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=nsaved,
            name_dim="rank",
        )

        assert dist.sizes["rank"] == nsaved
        assert index.sizes["rank"] == nsaved
