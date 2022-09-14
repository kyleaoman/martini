import pytest
import os
from astropy.units import isclose, allclose
from martini import DataCube


class TestDataCube:
    def test_datacube_dimensions(self):
        """
        Check that dimensions are as requested.
        """
        datacube = DataCube(n_px_x=10, n_px_y=11, n_channels=12)
        assert datacube._array.shape == (10, 11, 12, 1)

    def test_channel_mids(self, dc):
        """
        Check that first and last channel mids are spaced as expected.
        """
        bandwidth = dc.channel_mids[-1] - dc.channel_mids[0]
        assert bandwidth == (dc.n_channels - 1) * dc.channel_width

    def test_channel_edges(self, dc):
        """
        Check that first and last channel edges are spaced as expected.
        """
        bandwidth = dc.channel_edges[-1] - dc.channel_edges[0]
        assert bandwidth == dc.n_channels * dc.channel_width

    def test_iterators(self, dc):
        """
        Check that iterators over slices give us expected lengths.
        """
        assert len(list(dc.spatial_slices())) == dc.n_channels
        assert len(list(dc.spectra())) == dc.n_px_x * dc.n_px_y

    def test_freq_channels(self, dc):
        raise NotImplementedError

    def test_velocity_channels(self, dc):
        raise NotImplementedError

    def test_channel_mode_switching(self, dc):
        initial_mids = dc.channel_mids
        initial_edges = dc.channel_edges
        dc.freq_channels()
        dc.velocity_channels()
        assert allclose(dc.channel_edges, initial_edges)
        assert allclose(dc.channel_mids, initial_mids)

    def test_add_pad(self, dc):
        raise NotImplementedError

    def test_drop_pad(self, dc):
        raise NotImplementedError

    @pytest.mark.parametrize("with_fchannels", (False, True))
    @pytest.mark.parametrize("with_pad", (False, True))
    def test_copy(self, dc, with_fchannels, with_pad):
        if with_fchannels:
            dc.freq_channels()
        else:
            dc.velocity_channels()
        if with_pad:
            dc.add_pad((3, 3))
        copy = dc.copy()
        for attr in (
            "n_px_x",
            "n_px_y",
            "n_channels",
            "padx",
            "pady",
        ):
            assert getattr(dc, attr) == getattr(copy, attr)
        for attr in (
            "px_size",
            "channel_width",
            "velocity_centre",
            "ra",
            "dec",
        ):
            assert isclose(getattr(dc, attr), getattr(copy, attr))
        for attr in (
            "channel_edges",
            "channel_mids",
            "_array",
        ):
            assert allclose(getattr(dc, attr), getattr(copy, attr))
        assert str(dc.wcs) == str(copy.wcs)

    @pytest.mark.parametrize("with_fchannels", (False, True))
    @pytest.mark.parametrize("with_pad", (False, True))
    def test_save_and_load_state(self, dc, with_fchannels, with_pad):
        try:
            if with_fchannels:
                dc.freq_channels()
            else:
                dc.velocity_channels()
            if with_pad:
                dc.add_pad((3, 3))
            dc.save_state("test_savestate.hdf5", overwrite=True)
            loaded = DataCube.load_state("test_savestate.hdf5")
            for attr in (
                "n_px_x",
                "n_px_y",
                "n_channels",
                "padx",
                "pady",
            ):
                assert getattr(dc, attr) == getattr(loaded, attr)
            for attr in (
                "px_size",
                "channel_width",
                "velocity_centre",
                "ra",
                "dec",
            ):
                assert isclose(getattr(dc, attr), getattr(loaded, attr))
            for attr in (
                "channel_edges",
                "channel_mids",
                "_array",
            ):
                assert allclose(getattr(dc, attr), getattr(loaded, attr))
            assert str(dc.wcs) == str(loaded.wcs)
        except Exception:
            raise
        finally:
            os.remove("test_savestate.hdf5")
