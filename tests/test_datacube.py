import pytest
import os
import numpy as np
from astropy import units as U
from martini import DataCube
from martini.datacube import HIfreq


class TestDataCube:
    def test_datacube_dimensions(self):
        """
        Check that dimensions are as requested.
        """
        datacube = DataCube(n_px_x=10, n_px_y=11, n_channels=12)
        expected_shape = (10, 11, 12, 1) if datacube.stokes_axis else (10, 11, 12)
        assert datacube._array.shape == expected_shape

    def test_channel_mids(self, dc_zeros):
        """
        Check that first and last channel mids are spaced as expected.
        """
        bandwidth = dc_zeros.channel_mids[-1] - dc_zeros.channel_mids[0]
        assert bandwidth == (dc_zeros.n_channels - 1) * dc_zeros.channel_width

    def test_channel_edges(self, dc_zeros):
        """
        Check that first and last channel edges are spaced as expected.
        """
        bandwidth = dc_zeros.channel_edges[-1] - dc_zeros.channel_edges[0]
        assert bandwidth == dc_zeros.n_channels * dc_zeros.channel_width

    def test_iterators(self, dc_zeros):
        """
        Check that iterators over slices give us expected lengths.
        """
        assert len(list(dc_zeros.spatial_slices())) == dc_zeros.n_channels
        assert len(list(dc_zeros.spectra())) == dc_zeros.n_px_x * dc_zeros.n_px_y

    def test_freq_channels(self, dc_zeros):
        """
        Check that we can convert to frequency channels.
        """
        v_channel_mids = dc_zeros.channel_mids
        v_channel_edges = dc_zeros.channel_edges
        dc_zeros.freq_channels()
        assert U.allclose(
            dc_zeros.channel_mids.to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq)),
            v_channel_mids,
        )
        assert U.allclose(
            dc_zeros.channel_edges.to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq)),
            v_channel_edges,
        )

    def test_velocity_channels(self, dc_zeros):
        """
        Check that we can convert to velocity channels.
        """
        dc_zeros.freq_channels()
        f_channel_mids = dc_zeros.channel_mids
        f_channel_edges = dc_zeros.channel_edges
        dc_zeros.velocity_channels()
        assert U.allclose(
            dc_zeros.channel_mids.to(U.Hz, equivalencies=U.doppler_radio(HIfreq)),
            f_channel_mids,
        )
        assert U.allclose(
            dc_zeros.channel_edges.to(U.Hz, equivalencies=U.doppler_radio(HIfreq)),
            f_channel_edges,
        )

    def test_channel_mode_switching(self, dc_zeros):
        """
        Check that switching twice returns to starting point.
        """
        initial_mids = dc_zeros.channel_mids
        initial_edges = dc_zeros.channel_edges
        dc_zeros.freq_channels()
        dc_zeros.velocity_channels()
        assert U.allclose(dc_zeros.channel_edges, initial_edges)
        assert U.allclose(dc_zeros.channel_mids, initial_mids)

    def test_add_pad(self, dc_zeros):
        """
        Check that adding pad gives desired shape.
        """
        old_shape = dc_zeros._array.shape
        pad = (2, 3)
        dc_zeros.add_pad(pad)
        expected_shape = (
            old_shape[0] + 2 * pad[0],
            old_shape[1] + 2 * pad[1],
            old_shape[2],
        )
        if dc_zeros.stokes_axis:
            expected_shape = expected_shape + (old_shape[3],)
        assert dc_zeros._array.shape == expected_shape
        assert dc_zeros.padx == pad[0]
        assert dc_zeros.pady == pad[1]

    def test_add_pad_already_padded(self, dc_zeros):
        """
        Check that we can't double-pad.
        """
        pad = (2, 3)
        dc_zeros.add_pad(pad)
        with pytest.raises(RuntimeError, match="Tried to add padding"):
            dc_zeros.add_pad(pad)

    def test_drop_pad(self, dc_zeros):
        """
        Check that we get expected shape when dropping padding.
        """
        initial_shape = dc_zeros._array.shape
        pad = (2, 3)
        dc_zeros.add_pad(pad)
        old_shape = dc_zeros._array.shape
        dc_zeros.drop_pad()
        expected_shape = (
            old_shape[0] - 2 * pad[0],
            old_shape[1] - 2 * pad[1],
            old_shape[2],
        )
        if dc_zeros.stokes_axis:
            expected_shape = expected_shape + (old_shape[3],)
        assert dc_zeros._array.shape == initial_shape
        assert dc_zeros._array.shape == expected_shape
        assert dc_zeros.padx == 0
        assert dc_zeros.pady == 0

    def test_drop_pad_already_dropped(self, dc_zeros):
        """
        Check that dropping already dropped pad gives no change.
        """
        assert dc_zeros.padx == 0
        assert dc_zeros.pady == 0
        dc_zeros.drop_pad()
        assert dc_zeros.padx == 0
        assert dc_zeros.pady == 0

    @pytest.mark.parametrize("with_fchannels", (False, True))
    @pytest.mark.parametrize("with_pad", (False, True))
    def test_copy(self, dc_random, with_fchannels, with_pad):
        """
        Check that copying a datacube copies all required information.
        """
        if with_fchannels:
            dc_random.freq_channels()
        else:
            dc_random.velocity_channels()
        if with_pad:
            dc_random.add_pad((3, 3))
        copy = dc_random.copy()
        for attr in (
            "n_px_x",
            "n_px_y",
            "n_channels",
            "padx",
            "pady",
        ):
            assert getattr(dc_random, attr) == getattr(copy, attr)
        for attr in (
            "px_size",
            "channel_width",
            "velocity_centre",
            "ra",
            "dec",
        ):
            assert U.isclose(getattr(dc_random, attr), getattr(copy, attr))
        for attr in (
            "channel_edges",
            "channel_mids",
            "_array",
        ):
            assert U.allclose(getattr(dc_random, attr), getattr(copy, attr))
        assert str(dc_random.wcs) == str(copy.wcs)

    @pytest.mark.parametrize("with_fchannels", (False, True))
    @pytest.mark.parametrize("with_pad", (False, True))
    def test_save_and_load_state(self, dc_random, with_fchannels, with_pad):
        """
        Check that we can recover a datacube from a save file.
        """
        try:
            if with_fchannels:
                dc_random.freq_channels()
            else:
                dc_random.velocity_channels()
            if with_pad:
                dc_random.add_pad((3, 3))
            dc_random.save_state("test_savestate.hdf5", overwrite=True)
            loaded = DataCube.load_state("test_savestate.hdf5")
            for attr in (
                "n_px_x",
                "n_px_y",
                "n_channels",
                "padx",
                "pady",
            ):
                assert getattr(dc_random, attr) == getattr(loaded, attr)
            for attr in (
                "px_size",
                "channel_width",
                "velocity_centre",
                "ra",
                "dec",
            ):
                assert U.isclose(getattr(dc_random, attr), getattr(loaded, attr))
            for attr in (
                "channel_edges",
                "channel_mids",
                "_array",
            ):
                assert U.allclose(getattr(dc_random, attr), getattr(loaded, attr))
            assert str(dc_random.wcs) == str(loaded.wcs)
        except Exception:
            raise
        finally:
            os.remove("test_savestate.hdf5")

    def test_init_with_frequency_channel_spec(self, dc_random):
        """
        Check that we can specify channel spacing and central channel in frequency units.
        """
        const_kwargs = dict(
            n_px_x=dc_random.n_px_x,
            n_px_y=dc_random.n_px_y,
            n_channels=dc_random.n_channels,
        )
        f_velocity_centre = dc_random.velocity_centre.to(
            U.Hz, equivalencies=U.doppler_radio(HIfreq)
        )
        f_channel_width = (
            dc_random.velocity_centre - 0.5 * dc_random.channel_width
        ).to(U.Hz, equivalencies=U.doppler_radio(HIfreq)) - (
            dc_random.velocity_centre + 0.5 * dc_random.channel_width
        ).to(
            U.Hz, equivalencies=U.doppler_radio(HIfreq)
        )
        dc_vf = DataCube(
            velocity_centre=dc_random.velocity_centre,
            channel_width=f_channel_width,
            **const_kwargs,
        )
        dc_fv = DataCube(
            velocity_centre=f_velocity_centre,
            channel_width=dc_random.channel_width,
            **const_kwargs,
        )
        dc_ff = DataCube(
            velocity_centre=f_velocity_centre,
            channel_width=f_channel_width,
            **const_kwargs,
        )
        assert U.allclose(dc_vf.channel_edges, dc_random.channel_edges)
        assert U.allclose(dc_fv.channel_edges, dc_random.channel_edges)
        assert U.allclose(dc_ff.channel_edges, dc_random.channel_edges)

    def test_channels_equal_in_frequency(self, dc_zeros):
        """
        Expect channels to be equally spaced in frequency, check that this is the case.
        """
        dc_zeros.freq_channels()
        assert U.allclose(
            np.diff(np.diff(dc_zeros.channel_edges)), 0 * U.Hz, atol=1e-5 * U.Hz
        )
