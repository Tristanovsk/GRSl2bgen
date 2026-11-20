import numpy as np
import pandas as pd
import pytest
import xarray as xr

from unittest.mock import MagicMock

from GRSl2bgen.owt import OWT, OWT_process


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def raster():
    """Small synthetic reflectance raster."""

    return xr.Dataset(
        data_vars={
            "Rrs": (
                ("wl", "y", "x"),
                np.array([
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                ]),
            )
        },
        coords={
            "wl": [400, 500, 600],
            "y": [0, 1],
            "x": [0, 1],
        },
    )


@pytest.fixture
def xowt():
    """Synthetic OWT database with three classes."""

    return xr.DataArray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dims=("owt", "wl"),
        coords={
            "owt": [1, 2, 3],
            "wl": [400, 500, 600],
        },
        name="m_Rrs",
    )


@pytest.fixture
def owt(raster, xowt):
    """OWT instance using externally supplied spectra."""

    return OWT(
        raster=raster,
        xowt=xowt,
        Nclasses_to_be_saved=2,
    )

# ============================================================
# Tests for init
# ============================================================

class TestInitialization:
    
    def test_external_owt(self,raster, xowt):

        model = OWT(
            raster=raster,
            xowt=xowt,
            Nclasses_to_be_saved=2,
        )

        assert model.raster is raster
        assert model.owt is xowt

        assert model.Nwl == 3
        assert model.height == 2
        assert model.width == 2

        assert model.Nowt == 3
        assert model.Nclasses_to_be_saved == 2

        assert model.owt_index_name == "owt_index"
        assert model.owt_dist_name == "owt_dist"

        assert model.dist_units == "radians"

        assert model.Rrs_owt.shape == (3, 3)
        assert model.Rrs_owt.dims == ("owt", "wl")

        np.testing.assert_array_equal(
            model.Rrs_owt.values,
            xowt.values,
        )


    def test_custom_database_name(self,raster, xowt):

        model = OWT(
            raster=raster,
            xowt=xowt,
            owt_database_name="test",
        )

        assert model.owt_index_name == "owt_index_test"
        assert model.owt_dist_name == "owt_dist_test"


    def test_spectral_distance_euclidean(self,raster, xowt):

        model = OWT(
            raster=raster,
            xowt=xowt,
            spectral_distance="euclidean",
        )

        assert model.dist_units == "sr-1"


    def test_invalid_spectral_distance(self,raster, xowt):

        with pytest.raises(ValueError, match="unknown sepctral distance"):
            OWT(
                raster=raster,
                xowt=xowt,
                spectral_distance="invalid",
            )


    def test_invalid_database(self,raster):

        with pytest.raises(ValueError, match="unknown OWT database"):
            OWT(
                raster=raster,
                owt_database="invalid",
            )


    def test_wavelength_range(self,raster, xowt):
        owt = OWT(
            raster,
            xowt=xowt,
            wl_range=slice(450, 600),
        )

        assert owt.Nwl == 2
        np.testing.assert_array_equal(
            owt.Rrs.wl.values,
            [500, 600],
        )
        assert owt.Rrs_owt.dims == ("owt", "wl")
        assert owt.Rrs_owt.shape == (3, 2)



    def test_custom_number_of_classes(self,raster, xowt):

        model = OWT(
            raster=raster,
            xowt=xowt,
            Nclasses_to_be_saved=3,
        )

        assert model.Nclasses_to_be_saved == 3


# ============================================================
# Tests for xSAM
# ============================================================

class TestXSAM:
    
    def test_nominal(self):

        R1 = xr.DataArray(
            [1.0, 0.0, 0.0],
            dims="wl",
            coords={"wl": [400, 500, 600]},
        )

        R2 = xr.DataArray(
            [0.0, 1.0, 0.0],
            dims="wl",
            coords={"wl": [400, 500, 600]},
        )

        result = OWT.xSAM(R1, R2)

        np.testing.assert_allclose(result, np.pi / 2)


    def test_identical_spectra(self):

        R1 = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims="wl",
        )

        result = OWT.xSAM(R1, R1)

        np.testing.assert_allclose(result, 0.0, atol=1e-7)


    def test_opposite_spectra(self):

        R1 = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims="wl",
        )

        R2 = xr.DataArray(
            [-1.0, -2.0, -3.0],
            dims="wl",
        )

        result = OWT.xSAM(R1, R2)

        np.testing.assert_allclose(result, np.pi)


# ============================================================
# Tests for SCS
# ============================================================

class TestSCS:
    
    def test_identical_spectra(self):

        R1 = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0],
            dims="wl",
        )

        result = OWT.SCS(R1, R1)

        np.testing.assert_allclose(result, 1.0)


    def test_opposite_spectra(self):

        R1 = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0],
            dims="wl",
        )

        R2 = xr.DataArray(
            [-1.0, -2.0, -3.0, -4.0],
            dims="wl",
        )

        result = OWT.SCS(R1, R2)

        np.testing.assert_allclose(result, -1.0)


    def test_uncorrelated_spectra(self):

        R1 = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0],
            dims="wl",
        )

        R2 = xr.DataArray(
            [1.0, -1.0, -1.0, 1.0],
            dims="wl",
        )

        result = OWT.SCS(R1, R2)

        np.testing.assert_allclose(result, 0.0, atol=1e-7)


# ============================================================
# Tests for SAM
# ============================================================

def test_SAM():

    Rrs = np.array([
        [
            [1.0, 1.0],
        ],
        [
            [1.0, 2.0],
        ],
    ])

    Rrs_owt = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
    ])

    dist, index = OWT.SAM(
        Rrs=Rrs,
        Rrs_owt=Rrs_owt,
        Nclasses_to_be_saved=1,
    )

    assert dist.shape == (1, 1, 2)
    assert index.shape == (1, 1, 2)

    np.testing.assert_allclose(dist, 0.0, atol=1e-7)
    np.testing.assert_array_equal(index, [[[1.0, 2.0]]])




# ============================================================
# Tests for multi_process
# ============================================================

class TestMultiProcess:
    
    @pytest.mark.parametrize(
        "spectral_distance, expected_units",
        [
            ("sam", "radians"),
            ("euclidean", "sr-1"),
        ],
    )
    def test_nominal(self,spectral_distance, expected_units, raster, xowt):

        model = OWT(
            raster=raster,
            xowt=xowt,
            spectral_distance=spectral_distance,
            Nclasses_to_be_saved=2,
        )

        result = model.multi_process()

        assert isinstance(result, xr.Dataset)

        assert "owt_dist" in result
        assert "owt_index" in result

        assert result.owt_dist.dims == ("Nclasses", "y", "x")
        assert result.owt_index.dims == ("Nclasses", "y", "x")

        assert result.owt_dist.shape == (2, 2, 2)
        assert result.owt_index.shape == (2, 2, 2)

        assert result.owt_dist.dtype == np.float32
        assert result.owt_index.dtype == np.float32

        assert result.owt_dist.attrs["units"] == expected_units

        assert "definition" in result.owt_index.attrs

        assert result.Nclasses.values.tolist() == [0, 1]


    def test_one_class(self,raster, xowt):

        model = OWT(
            raster=raster,
            xowt=xowt,
            Nclasses_to_be_saved=1,
        )

        result = model.multi_process()

        assert result.owt_dist.dims == ("y", "x")
        assert result.owt_index.dims == ("y", "x")

        assert result.owt_dist.shape == (2, 2)
        assert result.owt_index.shape == (2, 2)


    def test_custom_names(self,raster, xowt):

        model = OWT(
            raster=raster,
            xowt=xowt,
            owt_database_name="test",
            Nclasses_to_be_saved=2,
        )

        result = model.multi_process()

        assert "owt_dist_test" in result
        assert "owt_index_test" in result


    def test_classification(self,raster):

        # Each pixel has a spectrum matching one OWT class.
        xowt = xr.DataArray(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ],
            dims=("owt", "wl"),
            coords={
                "owt": [1, 2, 3, 4],
                "wl": [400, 500, 600],
            },
        )

        model = OWT(
            raster=raster,
            xowt=xowt,
            spectral_distance="euclidean",
            Nclasses_to_be_saved=1,
        )

        result = model.multi_process()

        expected_index = np.array([
            [1, 2],
            [3, 4],
        ])

        np.testing.assert_array_equal(
            result.owt_index.values,
            expected_index,
        )

        np.testing.assert_allclose(
            result.owt_dist.values,
            0.0,
            atol=1e-7,
        )


# ============================================================
# Tests for set_range
# ============================================================

def test_set_range():
    values = xr.DataArray(
        [0, 1, 5, 10, 20, 30],
        dims="x",
    )

    owt = OWT.__new__(OWT)

    result = owt.set_range(
        values,
        minval=1,
        maxval=20,
    )

    expected = xr.DataArray(
        [np.nan, np.nan, 5, 10, np.nan, np.nan],
        dims="x",
    )

    xr.testing.assert_equal(result, expected)



def test_set_range_custom_thresholds():
    values = xr.DataArray([1, 5, 10, 15, 20], dims="x")

    owt = OWT.__new__(OWT)

    result = owt.set_range(values, minval=5, maxval=15)

    expected = xr.DataArray(
        [np.nan, np.nan, 10, np.nan, np.nan],
        dims="x",
    )

    xr.testing.assert_equal(result, expected)



# ============================================================
# Tests for plot
# ============================================================

def test_plot_external_owt(raster, xowt):

    model = OWT(
        raster=raster,
        xowt=xowt,
    )

    with pytest.raises(
        NotImplementedError,
        match="plot needs owt_info",
    ):
        model.plot()


# ============================================================
# Tests for OWT_process
# ============================================================

class TestOwtProcess:
    
    def test_init(self,raster):

        process = OWT_process(raster)

        assert process.raster is raster
        assert process.spectral_distance == "sam"
        assert process.chunk == 1024
        assert process.Nproc == 8

        assert process.cmaps == {}


    def test_custom_parameters(self,raster):

        process = OWT_process(
            raster,
            spectral_distance="euclidean",
            chunk=256,
            Nproc=4,
        )

        assert process.spectral_distance == "euclidean"
        assert process.chunk == 256
        assert process.Nproc == 4


    def test_execute(self,monkeypatch, raster):

        """Test orchestration without loading the real OWT databases."""

        calls = []

        class FakeOWT:

            def __init__(
                self,
                raster,
                owt_database,
                owt_database_name,
                param="m_nRrs",
                spectral_distance="sam",
                chunk=1024,
                Nproc=8,
                Nclasses_to_be_saved=1,
                exclude_classes=None,
            ):

                calls.append({
                    "exclude": exclude_classes,
                    "database": owt_database,
                    "database_name": owt_database_name,
                    "param": param,
                    "spectral_distance": spectral_distance,
                    "chunk": chunk,
                    "Nproc": Nproc,
                    "Nsaved": Nclasses_to_be_saved,
                })

                self.cmap_owt = MagicMock()

                self.owt_database_name = owt_database_name

            def multi_process(self):

                name = self.owt_database_name

                return xr.Dataset({
                    f"owt_dist_{name}": (
                        ("y", "x"),
                        np.zeros((2, 2)),
                    ),
                    f"owt_index_{name}": (
                        ("y", "x"),
                        np.ones((2, 2)),
                    ),
                })

        # Replace OWT with our fake implementation.
        monkeypatch.setattr(
            "GRSl2bgen.owt.OWT",
            FakeOWT,
        )

        process = OWT_process(
            raster,
            spectral_distance="sam",
            chunk=512,
            Nproc=4,
        )

        process.execute()

        # Four databases must be processed.
        assert len(calls) == 4

        expected_databases = [
            "Spyrakos2018",
            "Bi2024",
            "Tarasenko2025",
            "invRrsWT",
        ]

        assert [c["database"] for c in calls] == expected_databases

        # Spyrakos2018 uses the requested spectral distance.
        assert calls[0]["spectral_distance"] == "sam"

        # Bi2024 uses the default SAM distance.
        assert calls[1]["spectral_distance"] == "sam"

        # Tarasenko2025 uses the default SAM distance.
        assert calls[2]["spectral_distance"] == "sam"

        # the invRrs water types always use the spectral angle.
        assert calls[3]["spectral_distance"] == "sam"

        # Check the number of saved classes.
        assert calls[0]["Nsaved"] == 3
        assert calls[3]["Nsaved"] == 2

        # Check common parameters.
        for call in calls:
            assert call["chunk"] == 512
            assert call["Nproc"] == 4

        # Check that all four colormaps are stored.
        assert len(process.cmaps) == 4

        assert all(
            name in process.cmaps
            for name in expected_databases
        )

        # Check that the individual results exist.
        assert hasattr(process, "xowt_spyrakos2018")
        assert hasattr(process, "xowt_bi2024")
        assert hasattr(process, "xowt_Ta2025")
        assert hasattr(process, "xowt_invRrsWT")

        # Check the merged output.
        assert isinstance(process.output, xr.Dataset)

        expected_variables = [
            "owt_dist_Spyrakos2018",
            "owt_index_Spyrakos2018",
            "owt_dist_Bi2024",
            "owt_index_Bi2024",
            "owt_dist_Ta2025",
            "owt_index_Ta2025",
            "owt_dist_invRrsWT",
            "owt_index_invRrsWT",
        ]

        for variable in expected_variables:
            assert variable in process.output


    def test_execute_euclidean(self,monkeypatch, raster):

        """Check that the requested distance is passed to Spyrakos."""

        calls = []

        class FakeOWT:

            def __init__(self, raster, owt_database, **kwargs):

                calls.append({
                    "database": owt_database,
                    **kwargs,
                })

                self.cmap_owt = None

                self.database_name = kwargs["owt_database_name"]

            def multi_process(self):

                name = self.database_name

                return xr.Dataset({
                    f"owt_dist_{name}": (
                        ("y", "x"),
                        np.zeros((2, 2)),
                    ),
                    f"owt_index_{name}": (
                        ("y", "x"),
                        np.ones((2, 2)),
                    ),
                })

        monkeypatch.setattr(
            "GRSl2bgen.owt.OWT",
            FakeOWT,
        )

        process = OWT_process(
            raster,
            spectral_distance="euclidean",
        )

        process.execute()

        assert calls[0]["spectral_distance"] == "euclidean"

        # the invRrs water types keep the spectral angle.
        assert calls[3]["spectral_distance"] == "sam"


# ============================================================
# Negative reflectance
# ============================================================

def _rrs_with_negatives():
    # pixel (0, 0): small negative; (0, 1): large negative; row 1: positive
    data = np.array([
        [[0.010, 0.010], [0.010, 0.010]],
        [[0.008, -0.002], [0.008, 0.008]],
        [[-0.0001, 0.004], [0.004, 0.004]],
    ])
    return xr.DataArray(data, dims=('wl', 'y', 'x'), coords={'wl': [400, 500, 600], 'y': [0, 1], 'x': [0, 1]})


class TestNegativeRrs:

    def test_offset(self):
        from GRSl2bgen.owt import handle_negative_rrs
        rrs = _rrs_with_negatives()
        out = handle_negative_rrs(rrs, 'offset')
        assert np.allclose(out.min('wl').sel(y=0), 0.)                         # minimum brought to 0
        assert np.allclose(out.sel(y=0, x=1), rrs.sel(y=0, x=1) + 0.002)      # flat offset
        assert np.allclose(out.sel(y=1), rrs.sel(y=1))                         # positive pixels unchanged
        assert np.isfinite(out).all()

    def test_clip(self):
        from GRSl2bgen.owt import handle_negative_rrs
        out = handle_negative_rrs(_rrs_with_negatives(), 'clip')
        assert out.sel(wl=500, y=0, x=1) == 0. and out.sel(wl=600, y=0, x=0) == 0.

    def test_mask(self):
        from GRSl2bgen.owt import handle_negative_rrs
        out = handle_negative_rrs(_rrs_with_negatives(), 'mask', 5e-4)
        assert out.sel(y=0, x=1).isnull().all()
        assert out.sel(wl=600, y=0, x=0) == 0.

    def test_none_and_unknown(self):
        from GRSl2bgen.owt import handle_negative_rrs
        rrs = _rrs_with_negatives()
        xr.testing.assert_identical(handle_negative_rrs(rrs, None), rrs)
        with pytest.raises(ValueError):
            handle_negative_rrs(rrs, 'foo')

    def test_owt_default_offset_classifies_all(self, xowt):
        raster = xr.Dataset({'Rrs': _rrs_with_negatives()})
        res = OWT(raster, xowt=xowt, owt_database_name='test').multi_process()
        assert np.isfinite(res['owt_index_test']).all()

    def test_owt_mask(self, xowt):
        raster = xr.Dataset({'Rrs': _rrs_with_negatives()})
        res = OWT(raster, xowt=xowt, owt_database_name='test', negative_handling='mask').multi_process()
        assert np.isnan(res['owt_index_test'].sel(y=0, x=1))
        assert np.isfinite(res['owt_index_test'].sel(y=1)).all()


def test_invrrs_water_types_database():
    """Built-in invRrs water types: 14 classes in sr-1, labels and colours, classified by spectral angle."""
    wl = [443, 490, 560, 665, 705, 740, 783]
    raster = xr.Dataset({'Rrs': (('wl', 'y', 'x'), np.full((len(wl), 1, 2), 0.01))},
                        coords={'wl': wl, 'y': [0], 'x': [0, 1]})
    o = OWT(raster, owt_database='invRrsWT', owt_database_name='invRrsWT', Nclasses_to_be_saved=2)
    assert o.Nowt == 14
    assert list(o.owt.owt.values[:2]) == ['WT01', 'WT02']
    assert len(o.owt_info) == 14 and o.owt_info[14]['label'].startswith('WT14')
    res = o.multi_process()
    assert res['owt_dist_invRrsWT'].attrs['units'] == 'radians'
    assert res.sizes['Nclasses'] == 2
    assert np.isfinite(res['owt_index_invRrsWT']).all()


# ============================================================
# Tests for exclude_classes
# ============================================================

class TestExcludeClasses:

    @pytest.fixture
    def xowt4(self):
        return xr.DataArray(
            [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 1.0]],
            dims=("owt", "wl"), coords={"owt": ["a", "b", "c", "d"], "wl": [400, 500, 600]}, name="m_Rrs")

    @pytest.fixture
    def raster4(self):
        # pixels equal to the classes a, b, c and d
        sp = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 1.0]]).T.reshape(3, 2, 2)
        return xr.Dataset({"Rrs": (("wl", "y", "x"), sp)}, coords={"wl": [400, 500, 600], "y": [0, 1], "x": [0, 1]})

    def test_none(self, raster4, xowt4):
        r = OWT(raster4, xowt=xowt4).multi_process()
        np.testing.assert_array_equal(r.owt_index.values, [[1, 2], [3, 4]])
        assert "excluded_classes" not in r.owt_index.attrs

    @pytest.mark.parametrize("exclude", [["b"], [2], "b"])
    def test_numbering_kept(self, raster4, xowt4, exclude):
        r = OWT(raster4, xowt=xowt4, exclude_classes=exclude, Nclasses_to_be_saved=2).multi_process()
        idx = r.owt_index.values
        # class b never assigned; the other pixels keep their class and number
        assert not np.any(idx == 2)
        np.testing.assert_array_equal(idx[0, :, :].ravel()[[0, 2, 3]], [1, 3, 4])
        assert r.owt_index.attrs["excluded_classes"] == "b"
        # the pixel equal to b goes to its closest kept class, at a positive angle
        assert idx[0, 0, 1] in (1, 3, 4) and r.owt_dist.values[0, 0, 1] > 0

    def test_dask(self, raster4, xowt4):
        lazy = OWT(raster4.chunk({"y": 1}), xowt=xowt4, exclude_classes=["b"], chunk=None).multi_process()
        eager = OWT(raster4, xowt=xowt4, exclude_classes=["b"]).multi_process()
        np.testing.assert_array_equal(lazy.owt_index.values, eager.owt_index.values)

    def test_euclidean(self, raster4, xowt4):
        r = OWT(raster4, xowt=xowt4, exclude_classes=["d"], spectral_distance="euclidean").multi_process()
        assert not np.any(r.owt_index.values == 4)

    def test_errors(self, raster4, xowt4):
        with pytest.raises(ValueError):
            OWT(raster4, xowt=xowt4, exclude_classes=["z"])
        with pytest.raises(ValueError):
            OWT(raster4, xowt=xowt4, exclude_classes=[5])
        with pytest.raises(ValueError):
            OWT(raster4, xowt=xowt4, exclude_classes=["a", "b", "c", "d"])

    def test_tarasenko2025_without_5a(self, raster4):
        k = OWT(raster4, owt_database="Tarasenko2025", owt_database_name="Ta2025", exclude_classes=["5a"])
        assert k.excluded_classes == ["5a"] and len(k.kept_positions) == 15 and 6 not in k.kept_positions

    def test_owt_process_passes_ta2025_exclude(self, monkeypatch, raster):
        seen = {}

        class FakeOWT:
            def __init__(self, raster, owt_database, owt_database_name, exclude_classes=None, **kw):
                seen[owt_database] = exclude_classes
                self.cmap_owt = MagicMock()
                self.owt_database_name = owt_database_name

            def multi_process(self):
                n = self.owt_database_name
                return xr.Dataset({f"owt_index_{n}": (("y", "x"), np.ones((2, 2)))})

        monkeypatch.setattr("GRSl2bgen.owt.OWT", FakeOWT)
        OWT_process(raster, ta2025_exclude=["5a"]).execute()
        assert seen["Tarasenko2025"] == ["5a"]
        assert seen["Spyrakos2018"] is None and seen["Bi2024"] is None
        OWT_process(raster).execute()
        assert seen["Tarasenko2025"] is None
