import numpy as np
import pytest
import xarray as xr
import dask.array as da

from GRSl2bgen.sam import sam, sam_dataarray, _sam_core

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def owt():
    """Three synthetic OWT spectra with distinct spectral directions."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def Rrs():
    """Synthetic reflectance cube (Ny, Nx, Nwl)."""
    return np.array([
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
    ])


@pytest.fixture
def Rrs_dataarray(Rrs):
    """Synthetic xarray reflectance cube."""
    return xr.DataArray(
        Rrs,
        dims=("y", "x", "wl"),
        coords={
            "y": [10, 20],
            "x": [30, 40],
            "wl": [443, 490, 560],
        },
    )

@pytest.fixture
def owt_unit():
    """Three orthogonal unit OWT spectra."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# ============================================================
# Tests for _unit_classes
# ============================================================

class TestUnitClasses:
    
    def test_normalisation(self,owt):
        from GRSl2bgen.sam import _unit_classes

        result = _unit_classes(owt)

        assert result.shape == owt.shape
        assert result.dtype == np.float64

        norms = np.sqrt((result ** 2).sum(axis=1))

        np.testing.assert_allclose(norms, 1.0)


    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_invalid_values(self,invalid_value, owt):
        from GRSl2bgen.sam import _unit_classes

        invalid_owt = owt.copy()
        invalid_owt[0, 0] = invalid_value

        with pytest.raises(ValueError, match="finite and non-zero"):
            _unit_classes(invalid_owt)


    def test_zero_spectrum(self):
        from GRSl2bgen.sam import _unit_classes

        owt = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
        ])

        with pytest.raises(ValueError, match="finite and non-zero"):
            _unit_classes(owt)


# ============================================================
# Tests for sam
# ============================================================

class TestSam:
    
    @pytest.mark.parametrize("parallel", [True, False])
    def test_output(self,Rrs, owt, parallel):

        dist, index = sam(
            Rrs,
            owt,
            Nclasses_to_be_saved=3,
            parallel=parallel,
        )

        assert dist.shape == (3, 2, 2)
        assert index.shape == (3, 2, 2)

        assert dist.dtype == np.float32
        assert index.dtype == np.float32

        expected_dist = np.array([
            [
                [0.0, 0.0],
                [0.0, np.pi / 4],
            ],
            [
                [np.pi / 2, np.pi / 2],
                [np.pi / 2, np.pi / 4],
            ],
            [
                [np.pi / 2, np.pi / 2],
                [np.pi / 2, np.pi / 2],
            ],
        ])

        expected_index = np.array([
            [
                [1.0, 2.0],
                [3.0, 1.0],
            ],
            [
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            [
                [3.0, 3.0],
                [2.0, 3.0],
            ],
        ])
        
        np.testing.assert_allclose(
            dist,
            expected_dist,
            atol=1e-6,
        )

        np.testing.assert_array_equal(
            index,
            expected_index,
        )

    @pytest.mark.parametrize("parallel", [True, False])
    def test_single_class(self,Rrs, owt, parallel):

        dist, index = sam(
            Rrs,
            owt,
            Nclasses_to_be_saved=1,
            parallel=parallel,
        )

        assert dist.shape == (1, 2, 2)
        assert index.shape == (1, 2, 2)

        expected_index = np.array([
            [[1.0, 2.0],
            [3.0, 1.0]],
        ], dtype=np.float32)

        np.testing.assert_array_equal(index, expected_index)

        expected_dist = np.array([
            [[0.0, 0.0],
            [0.0, np.pi / 4]],
        ], dtype=np.float32)

        np.testing.assert_allclose(
            dist,
            expected_dist,
            atol=1e-6,
        )

    def test_more_classes_than_available(self,Rrs, owt):

        dist, index = sam(
            Rrs,
            owt,
            Nclasses_to_be_saved=5,
            parallel=False,
        )

        assert dist.shape == (5, 2, 2)
        assert index.shape == (5, 2, 2)

        assert np.isfinite(dist[:3]).all()
        assert np.isfinite(index[:3]).all()

        assert np.isnan(dist[3:]).all()
        assert np.isnan(index[3:]).all()


    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    @pytest.mark.parametrize("parallel", [True, False])
    def test_invalid_pixels(self,invalid_value, parallel):

        Rrs = np.array([
            [
                [1.0, 0.0],
                [invalid_value, 1.0],
            ],
            [
                [0.0, invalid_value],
                [0.0, 1.0],
            ],
        ])

        owt = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        dist, index = sam(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            parallel=parallel,
        )

        assert np.isnan(dist[:, 0, 1]).all()
        assert np.isnan(index[:, 0, 1]).all()

        assert np.isnan(dist[:, 1, 0]).all()
        assert np.isnan(index[:, 1, 0]).all()

        assert np.isfinite(dist[:, 0, 0]).all()
        assert np.isfinite(index[:, 0, 0]).all()

        assert np.isfinite(dist[:, 1, 1]).all()
        assert np.isfinite(index[:, 1, 1]).all()


    @pytest.mark.parametrize("parallel", [True, False])
    def test_zero_pixel_spectrum(self,parallel):

        Rrs = np.array([
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
        ])

        owt = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        dist, index = sam(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            parallel=parallel,
        )

        assert np.isnan(dist[:, 0, 0]).all()
        assert np.isnan(index[:, 0, 0]).all()

        assert np.isfinite(dist[:, 0, 1]).all()
        assert np.isfinite(index[:, 0, 1]).all()


    @pytest.mark.parametrize(
        "Rrs, owt",
        [
            (
                np.ones((2, 2)),
                np.ones((2, 2)),
            ),
            (
                np.ones((2, 2, 3)),
                np.ones((2, 2)),
            ),
        ],
    )
    def test_invalid_dimensions(self,Rrs, owt):

        with pytest.raises(ValueError, match="Rrs must have shape"):
            sam(Rrs, owt)


    def test_normalisation(self):

        Rrs = np.array([
            [[10.0, 0.0]],
        ])

        owt = np.array([
            [100.0, 0.0],
            [0.0, 100.0],
        ])

        dist, index = sam(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            parallel=False,
        )

        np.testing.assert_allclose(
            dist[:, 0, 0],
            [0.0, np.pi / 2],
            atol=1e-6,
        )

        np.testing.assert_array_equal(
            index[:, 0, 0],
            [1.0, 2.0],
        )


# ============================================================
# Tests for sam_dataarray
# ============================================================

class TestSamDataArray:
    
    def test_output(self,Rrs_dataarray, owt):

        dist, index = sam_dataarray(
            Rrs_dataarray,
            owt,
            Nclasses_to_be_saved=3,
        )

        assert dist.dims == ("Nowt", "y", "x")
        assert index.dims == ("Nowt", "y", "x")

        assert dist.shape == (3, 2, 2)
        assert index.shape == (3, 2, 2)

        assert dist.dtype == np.float32
        assert index.dtype == np.float32

        assert dist.name == "owt_dist"
        assert index.name == "owt_index"

        expected_first_dist = np.array([
            [0.0, 0.0],
            [0.0, np.pi / 4],
        ])

        np.testing.assert_allclose(
            dist.values[0],
            expected_first_dist,
            atol=1e-6,
        )

        expected_first_index = np.array([
            [1.0, 2.0],
            [3.0, 1.0],
        ])

        np.testing.assert_array_equal(
            index.values[0],
            expected_first_index,
        )

    @pytest.mark.parametrize(
        "dims",
        [
            ("y", "x", "wl"),
            ("wl", "y", "x"),
            ("x", "wl", "y"),
            ("x", "y", "wl"),
            ("wl", "x", "y"),
            ("y", "wl", "x"),
        ],
    )
    def test_dimension_order(self,Rrs_dataarray, owt, dims):

        Rrs = Rrs_dataarray.transpose(*dims)

        dist, index = sam_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
        )

        expected_spatial_dims = tuple(d for d in dims if d != "wl")

        assert dist.dims == ("Nowt", *expected_spatial_dims)
        assert index.dims == ("Nowt", *expected_spatial_dims)


        assert dist.shape == (2, 2, 2)
        assert index.shape == (2, 2, 2)

        expected_dist = np.array([
            [0.0, 0.0],
            [0.0, np.pi / 4],
        ])

        np.testing.assert_allclose(
            dist.isel(Nowt=0),
            expected_dist,
            atol=1e-6,
        )

    def test_custom_dimensions(self,Rrs_dataarray, owt):

        dist, index = sam_dataarray(
            Rrs_dataarray,
            owt,
            Nclasses_to_be_saved=2,
            wl_dim="wl",
            name_dim="class_rank",
        )

        assert dist.dims == ("class_rank", "y", "x")
        assert index.dims == ("class_rank", "y", "x")

    def test_custom_wavelength_dimension(self):

        Rrs = xr.DataArray(
            [
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                ],
            ],
            dims=("y", "x", "band"),
            coords={
                "y": [0],
                "x": [0, 1],
                "band": [443, 490],
            },
        )

        owt = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ])

        dist, index = sam_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
            wl_dim="band",
            name_dim="class_rank",
        )

        assert dist.dims == ("class_rank", "y", "x")
        assert index.dims == ("class_rank", "y", "x")

        assert dist.shape == (2, 1, 2)
        assert index.shape == (2, 1, 2)

        assert dist.name == "owt_dist"
        assert index.name == "owt_index"

        np.testing.assert_allclose(
            dist.isel(class_rank=0),
            0.0,
            atol=1e-6,
        )

        np.testing.assert_array_equal(
            index.isel(class_rank=0),
            [[1.0, 1.0]],
        )

    def test_invalid_dimensions(self,owt):

        Rrs = xr.DataArray(
            np.ones((2, 2)),
            dims=("y", "x"),
        )

        with pytest.raises(ValueError, match="exactly three dimensions"):
            sam_dataarray(Rrs, owt)

    def test_wavelength_mismatch(self):

        Rrs = xr.DataArray(
            np.ones((2, 2, 3)),
            dims=("y", "x", "wl"),
        )

        owt = np.ones((2, 2))

        with pytest.raises(ValueError, match="different number of wavelengths"):
            sam_dataarray(Rrs, owt)


    @pytest.mark.parametrize(
        "invalid_value",
        [np.nan, np.inf, -np.inf],
    )
    def test_invalid_owt(self,invalid_value, Rrs_dataarray, owt):

        invalid_owt = owt.copy()
        invalid_owt[0, 0] = invalid_value

        with pytest.raises(ValueError, match="finite and non-zero"):
            sam_dataarray(Rrs_dataarray, invalid_owt)

    def test_zero_owt(self,Rrs_dataarray):

        owt = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        with pytest.raises(ValueError, match="finite and non-zero"):
            sam_dataarray(Rrs_dataarray, owt)

    def test_invalid_pixels(self):

        Rrs = xr.DataArray(
            [
                [
                    [1.0, 0.0],
                    [np.nan, 1.0],
                ],
                [
                    [np.inf, 1.0],
                    [0.0, 1.0],
                ],
            ],
            dims=("y", "x", "wl"),
        )

        owt = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        dist, index = sam_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
        )

        assert np.isfinite(dist.sel(y=0, x=0)).all()

        assert np.isnan(dist.sel(y=0, x=1)).all()
        assert np.isnan(index.sel(y=0, x=1)).all()

        assert np.isnan(dist.sel(y=1, x=0)).all()
        assert np.isnan(index.sel(y=1, x=0)).all()

    def test_zero_pixel_spectrum(self):

        Rrs = xr.DataArray(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                ],
            ],
            dims=("y", "x", "wl"),
        )

        owt = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        dist, index = sam_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
        )

        assert np.isnan(dist.sel(y=0, x=0)).all()
        assert np.isnan(index.sel(y=0, x=0)).all()

        assert np.isfinite(dist.sel(y=0, x=1)).all()

    def test_dask_lazy(self,Rrs_dataarray, owt):

        Rrs = Rrs_dataarray.chunk({
            "y": 1,
            "x": 1,
            "wl": 2,
        })

        dist, index = sam_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
        )

        # Results should remain lazy until explicitly computed.
        assert isinstance(dist.data, da.Array)
        assert isinstance(index.data, da.Array)

        dist_computed = dist.compute()
        index_computed = index.compute()

        assert dist_computed.shape == (2, 2, 2)
        assert index_computed.shape == (2, 2, 2)

        expected_dist = np.array([
            [
                [0.0, 0.0],
                [0.0, np.pi / 4],
            ],
            [
                [np.pi / 2, np.pi / 2],
                [np.pi / 2, np.pi / 4],
            ],
        ])

        np.testing.assert_allclose(
            dist_computed,
            expected_dist,
            atol=1e-6,
        )

    def test_dask_wavelength_rechunking(self,Rrs_dataarray, owt):

        Rrs = Rrs_dataarray.chunk({
            "y": 1,
            "x": 1,
            "wl": 1,
        })

        dist, index = sam_dataarray(
            Rrs,
            owt,
            Nclasses_to_be_saved=2,
        )

        assert isinstance(dist.data, da.Array)
        assert isinstance(index.data, da.Array)

        expected = np.array([
            [0.0, 0.0],
            [0.0, np.pi / 4],
        ])

        np.testing.assert_allclose(
            dist.compute().isel(Nowt=0),
            expected,
            atol=1e-6,
        )

# ============================================================
# Tests for _sam_core
# ============================================================

class TestSamCore:
    
    def test_output_shape_and_dtype(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 2)

        assert dist.shape == (2, 2, 2)
        assert index.shape == (2, 2, 2)

        assert dist.dtype == np.float32
        assert index.dtype == np.float32

    def test_closest_class(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            [
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 1)

        expected_dist = np.array([
            [[0.0], [0.0]],
            [[0.0], [np.pi / 4]],
        ])

        expected_index = np.array([
            [[1.0], [2.0]],
            [[3.0], [1.0]],
        ])

        np.testing.assert_allclose(dist, expected_dist, atol=1e-6)
        np.testing.assert_array_equal(index, expected_index)

    def test_multiple_classes(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 1.0, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 3)

        expected_dist = np.array([
            [
                [np.pi / 4, np.pi / 4, np.pi / 2],
            ],
        ])

        expected_index = np.array([
            [
                [1.0, 2.0, 3.0],
            ],
        ])

        np.testing.assert_allclose(dist, expected_dist, atol=1e-6)
        np.testing.assert_array_equal(index, expected_index)

    def test_invalid_nan_pixel(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 0.0, 0.0],
                [np.nan, 1.0, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 2)

        # First pixel is valid.
        assert np.all(np.isfinite(dist[0, 0]))
        assert np.all(np.isfinite(index[0, 0]))

        # Second pixel contains NaN.
        assert np.all(np.isnan(dist[0, 1]))
        assert np.all(np.isnan(index[0, 1]))

    @pytest.mark.parametrize("invalid_value", [
        np.nan,
        np.inf,
        -np.inf,
    ])
    def test_invalid_nonfinite_pixel(self,owt_unit, invalid_value):

        Rrs = np.array([
            [
                [1.0, invalid_value, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 2)

        assert np.all(np.isnan(dist))
        assert np.all(np.isnan(index))

    def test_zero_spectrum(self,owt_unit):

        Rrs = np.zeros((1, 1, 3), dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 2)

        assert np.all(np.isnan(dist))
        assert np.all(np.isnan(index))

    def test_more_saved_classes_than_available(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 0.0, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 5)

        assert dist.shape == (1, 1, 5)
        assert index.shape == (1, 1, 5)

        expected_dist = np.array([
            [[0.0, np.nan, np.nan, np.nan, np.nan]]
        ]).reshape(1, 1, 5)

        # First three classes have valid results.
        np.testing.assert_allclose(
            dist[0, 0, :3],
            [0.0, np.pi / 2, np.pi / 2],
            atol=1e-6,
        )

        np.testing.assert_array_equal(
            index[0, 0, :3],
            [1.0, 2.0, 3.0],
        )

        # Remaining positions must stay NaN.
        assert np.all(np.isnan(dist[0, 0, 3:]))
        assert np.all(np.isnan(index[0, 0, 3:]))

    def test_distances_sorted(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 3)

        # Distances must be sorted in ascending order.
        assert np.all(np.diff(dist, axis=-1) >= -1e-6)

        # Every class must appear exactly once per valid pixel.
        for iy in range(2):
            for ix in range(2):
                np.testing.assert_array_equal(
                    np.sort(index[iy, ix]),
                    [1.0, 2.0, 3.0],
                )

    def test_independent_pixels(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 1)

        expected_index = np.array([
            [[1.0], [2.0], [3.0]]
        ])

        expected_dist = np.zeros((1, 3, 1))

        np.testing.assert_array_equal(index, expected_index)
        np.testing.assert_allclose(dist, expected_dist, atol=1e-6)

    def test_negative_reflectance(self,owt_unit):

        Rrs = np.array([
            [
                [-1.0, 0.0, 0.0],
            ],
        ], dtype=np.float64)

        dist, index = _sam_core(Rrs, owt_unit, 3)

        expected_dist = np.array([
            [[[np.pi]]],
            [[[np.pi / 2]]],
            [[[np.pi / 2]]],
        ]).reshape(1, 1, 3)

        # The negative first component points opposite to OWT 1.
        np.testing.assert_allclose(
            dist[0, 0],
            [np.pi / 2, np.pi / 2, np.pi],
            atol=1e-6,
        )

        np.testing.assert_array_equal(
            index[0, 0],
            [2.0, 3.0, 1.0],
        )

    def test_parallel_and_serial(self,owt_unit):

        Rrs = np.array([
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ], dtype=np.float64)

        from GRSl2bgen.sam import _sam_parallel, _sam_serial

        dist_parallel, index_parallel = _sam_parallel(Rrs, owt_unit, 3)
        dist_serial, index_serial = _sam_serial(Rrs, owt_unit, 3)

        np.testing.assert_allclose(
            dist_parallel,
            dist_serial,
            atol=1e-6,
            equal_nan=True,
        )

        np.testing.assert_array_equal(
            index_parallel,
            index_serial,
        )
