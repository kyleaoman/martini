import pytest
import os
import numpy as np
from astropy import units as U, constants as C
from astropy.units import isclose, allclose
from martini import DataCube
from martini.datacube import HIfreq


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
        v_mid0 = dc.channel_mids[0]
        v_edge0 = dc.channel_edges[0]
        v_mid1 = dc.channel_mids[-1]
        v_edge1 = dc.channel_edges[-1]
        f_mid0 = HIfreq * (1 - v_mid0 / C.c)
        f_edge0 = HIfreq * (1 - v_edge0 / C.c)
        f_mid1 = HIfreq * (1 - v_mid1 / C.c)
        f_edge1 = HIfreq * (1 - v_edge1 / C.c)
        dc.freq_channels()
        assert allclose(
            dc.channel_mids, np.linspace(f_mid0, f_mid1, dc.n_channels), atol=1 * U.Hz
        )
        assert allclose(
            dc.channel_edges,
            np.linspace(f_edge0, f_edge1, dc.n_channels + 1),
            atol=1 * U.Hz,
        )

    def test_velocity_channels(self, dc):
        dc.freq_channels()
        f_mid0 = dc.channel_mids[0]
        f_edge0 = dc.channel_edges[0]
        f_mid1 = dc.channel_mids[-1]
        f_edge1 = dc.channel_edges[-1]
        v_mid0 = C.c * (1 - f_mid0 / HIfreq)
        v_edge0 = C.c * (1 - f_edge0 / HIfreq)
        v_mid1 = C.c * (1 - f_mid1 / HIfreq)
        v_edge1 = C.c * (1 - f_edge1 / HIfreq)
        dc.velocity_channels()
        assert allclose(
            dc.channel_mids,
            np.linspace(v_mid0, v_mid1, dc.n_channels),
            atol=1e-3 * U.m / U.s,
        )
        assert allclose(
            dc.channel_edges,
            np.linspace(v_edge0, v_edge1, dc.n_channels + 1),
            atol=1e-3 * U.m / U.s,
        )

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
