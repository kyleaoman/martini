import pytest
import os
import numpy as np
from astropy import wcs
from astropy import units as U
from martini import DataCube
from martini.datacube import HIfreq

try:
    import h5py
except ImportError:
    have_h5py = False
else:
    have_h5py = True


def check_wcs_match(wcs1, wcs2):
    assert set(wcs1.to_header().keys()) == set(wcs2.to_header().keys())
    for k, v, _ in wcs1.to_header().cards:
        assert v == wcs2.to_header()[k]


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
        bandwidth = np.abs(dc_zeros.channel_mids[-1] - dc_zeros.channel_mids[0])
        assert U.isclose(bandwidth, (dc_zeros.n_channels - 1) * dc_zeros.channel_width)

    def test_channel_edges(self, dc_zeros):
        """
        Check that first and last channel edges are spaced as expected.
        """
        bandwidth = np.abs(dc_zeros.channel_edges[-1] - dc_zeros.channel_edges[0])
        assert bandwidth == dc_zeros.n_channels * dc_zeros.channel_width

    def test_iterators(self, dc_zeros):
        """
        Check that iterators over slices give us expected lengths.
        """
        assert len(list(dc_zeros.channel_maps)) == dc_zeros.n_channels
        assert len(list(dc_zeros.spatial_slices)) == dc_zeros.n_channels
        assert len(list(dc_zeros.spectra)) == dc_zeros.n_px_x * dc_zeros.n_px_y

    def test_freq_channels(self, dc_zeros):
        """
        Check that frequency channels match WCS.
        """
        spec_unit = U.Unit(dc_zeros.wcs.wcs.cunit[dc_zeros.wcs.wcs.spec], format="fits")
        mids = (
            dc_zeros.wcs.sub(("spectral",)).all_pix2world(
                np.arange(dc_zeros.n_channels), 0
            )
        ) * spec_unit
        edges = (
            dc_zeros.wcs.sub(("spectral",)).all_pix2world(
                np.arange(dc_zeros.n_channels + 1) - 0.5, 0
            )
        ) * spec_unit
        assert U.allclose(
            mids.to(U.Hz, equivalencies=U.doppler_radio(HIfreq)),
            dc_zeros.frequency_channel_mids,
        )
        assert U.allclose(
            edges.to(U.Hz, equivalencies=U.doppler_radio(HIfreq)),
            dc_zeros.frequency_channel_edges,
        )

    def test_velocity_channels(self, dc_zeros):
        """
        Check that velocity channels match WCS.
        """
        spec_unit = U.Unit(dc_zeros.wcs.wcs.cunit[dc_zeros.wcs.wcs.spec], format="fits")
        mids = (
            dc_zeros.wcs.sub(("spectral",)).all_pix2world(
                np.arange(dc_zeros.n_channels), 0
            )
        ) * spec_unit
        edges = (
            dc_zeros.wcs.sub(("spectral",)).all_pix2world(
                np.arange(dc_zeros.n_channels + 1) - 0.5, 0
            )
        ) * spec_unit
        assert U.allclose(
            mids.to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq)),
            dc_zeros.velocity_channel_mids,
        )
        assert U.allclose(
            edges.to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq)),
            dc_zeros.velocity_channel_edges,
        )

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

    @pytest.mark.parametrize("with_pad", (False, True))
    def test_copy(self, dc_random, with_pad):
        """
        Check that copying a datacube copies all required information.
        """
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
            "spectral_centre",
            "ra",
            "dec",
        ):
            assert U.isclose(
                getattr(dc_random, attr),
                getattr(copy, attr),
                atol=1e-6 * getattr(dc_random, attr).unit,
            )
        for attr in (
            "_channel_edges",
            "_channel_mids",
            "_array",
        ):
            if getattr(dc_random, attr) is not None:
                assert U.allclose(getattr(dc_random, attr), getattr(copy, attr))
            else:
                assert getattr(copy, attr) is None
        check_wcs_match(dc_random.wcs, copy.wcs)

    @pytest.mark.skipif(
        not have_h5py, reason="h5py (optional dependency) not available."
    )
    @pytest.mark.parametrize("with_pad", (False, True))
    def test_save_and_load_state(self, dc_random, with_pad):
        """
        Check that we can recover a datacube from a save file.
        """
        try:
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
                "spectral_centre",
                "ra",
                "dec",
            ):
                assert U.isclose(
                    getattr(dc_random, attr),
                    getattr(loaded, attr),
                    atol=1e-6 * getattr(dc_random, attr).unit,
                )
            for attr in (
                "channel_edges",
                "channel_mids",
                "_array",
            ):
                assert U.allclose(getattr(dc_random, attr), getattr(loaded, attr))
            check_wcs_match(dc_random.wcs, loaded.wcs)
        except Exception:
            raise
        finally:
            os.remove("test_savestate.hdf5")

    def test_init_with_mixed_spectral_centre_and_channel_width_units(self):
        """
        Check that we can specify channel spacing and central channel in mixed units.
        """
        const_kwargs = dict(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
        )
        spectral_centre = 3 * 70 * U.km / U.s
        channel_width = 4 * U.km / U.s
        f_channel_width = np.abs(
            (spectral_centre + 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
            - (spectral_centre - 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
        )
        f_spectral_centre = spectral_centre.to(
            U.Hz, equivalencies=U.doppler_radio(HIfreq)
        )
        dc_vv = DataCube(
            spectral_centre=spectral_centre,
            channel_width=channel_width,
            **const_kwargs,
        )
        dc_vf = DataCube(
            spectral_centre=spectral_centre,
            channel_width=f_channel_width,
            **const_kwargs,
        )
        dc_fv = DataCube(
            spectral_centre=f_spectral_centre,
            channel_width=channel_width,
            **const_kwargs,
        )
        dc_ff = DataCube(
            spectral_centre=f_spectral_centre,
            channel_width=f_channel_width,
            **const_kwargs,
        )
        # expect channels to match where units of channel_width match
        # channel width as velocity:
        assert U.allclose(
            dc_vv.channel_mids,
            dc_fv.channel_mids,
        )
        # channel width as frequency:
        assert U.allclose(
            dc_ff.channel_mids,
            dc_vf.channel_mids,
        )

    def test_channel_spacing(self, dc_zeros):
        """
        Expect channels to be equally spaced in units matching channel_width, check that
        this is the case.
        """
        assert U.get_physical_type(dc_zeros.channel_width) == U.get_physical_type(
            dc_zeros.channel_mids
        )
        assert U.get_physical_type(dc_zeros.channel_width) == U.get_physical_type(
            dc_zeros.channel_edges
        )
        assert U.allclose(
            np.diff(np.diff(dc_zeros.channel_edges)),
            0 * dc_zeros.channel_width.unit,
            atol=1e-5 * dc_zeros.channel_width.unit,
        )
        assert U.allclose(
            np.diff(np.diff(dc_zeros.channel_edges)),
            0 * dc_zeros.channel_width.unit,
            atol=1e-5 * dc_zeros.channel_width.unit,
        )
        if U.get_physical_type(dc_zeros.channel_width) == "frequency":
            assert U.allclose(
                np.diff(np.diff(dc_zeros.frequency_channel_edges)),
                0 * U.Hz,
                atol=1e-5 * U.Hz,
            )
            assert U.allclose(
                np.diff(np.diff(dc_zeros.frequency_channel_mids)),
                0 * U.Hz,
                atol=1e-5 * U.Hz,
            )
        elif U.get_physical_type(dc_zeros.channel_width) == "velocity":
            assert U.allclose(
                np.diff(np.diff(dc_zeros.velocity_channel_edges)),
                0 * U.m / U.s,
                atol=1e-5 * U.m / U.s,
            )
            assert U.allclose(
                np.diff(np.diff(dc_zeros.velocity_channel_mids)),
                0 * U.m / U.s,
                atol=1e-5 * U.m / U.s,
            )


class TestDataCubeFromWCS:

    def test_consistent_with_direct(self, dc_random):
        """
        Check that extracting WCS from a constructed DataCube and constructing
        a DataCube from that WCS is consistent with the original DataCube.

        Note that we shouldn't expect this to reproduce a padded cube, or use
        a WCS from a padded cube!
        """
        from_wcs = DataCube.from_wcs(dc_random.wcs)
        for attr in (
            "n_px_x",
            "n_px_y",
            "n_channels",
            "padx",
            "pady",
        ):
            assert getattr(dc_random, attr) == getattr(from_wcs, attr)
        for attr in (
            "px_size",
            "channel_width",
            "spectral_centre",
            "ra",
            "dec",
        ):
            assert U.isclose(
                getattr(dc_random, attr),
                getattr(from_wcs, attr),
                atol=1e-6 * getattr(dc_random, attr).unit,
            )
        # don't test for arrays matching, they should not:
        for attr in (
            "channel_edges",
            "channel_mids",
        ):
            assert U.allclose(
                getattr(dc_random, attr),
                getattr(from_wcs, attr),
                atol=1e-6 * getattr(dc_random, attr).unit,
            )
        check_wcs_match(dc_random.wcs, from_wcs.wcs)

    @pytest.mark.parametrize(
        "sample_header",
        (
            "IC_2574_NA_CUBE_THINGS.HEADER",
            "comb_10tracks_J1337_28_HI_r10_t90_mg095_2.image.header",
            "WALLABY_PILOT_CUTOUT_APPROX.HEADER",
            # as previous but with axes reordered:
            "comb_10tracks_J1337_28_HI_r10_t90_mg095_2.image.header.permuted",
        ),
    )
    def test_sample_headers(self, sample_header):
        """
        Check that some real-world FITS headers can be used to initialize a DataCube.
        """
        with open(os.path.join("tests/data/", sample_header), "r") as f:
            hdr = f.read()
        with pytest.warns(wcs.FITSFixedWarning):
            hdr_wcs = wcs.WCS(hdr)
        if hdr_wcs.wcs.specsys == "":
            with pytest.warns(
                UserWarning,
                match="Input WCS did not specify 'SPECSYS'",
            ):
                dc = DataCube.from_wcs(hdr_wcs)
        elif hdr_wcs.wcs.specsys == "BARYCENT":
            with pytest.warns(
                UserWarning, match="Assuming ICRS barycentric reference system."
            ):
                dc = DataCube.from_wcs(hdr_wcs)
        else:
            dc = DataCube.from_wcs(hdr_wcs)
        assert dc.stokes_axis == (hdr_wcs.naxis == 4)
        centre_coords = hdr_wcs.all_pix2world(
            [[n_px // 2 + (1 + n_px % 2) / 2 for n_px in hdr_wcs.pixel_shape]],
            1,  # origin, i.e. index pixels from 1
        ).squeeze()
        for i, (centre_coord, unit, spacing, len_ax) in enumerate(
            zip(
                centre_coords,
                hdr_wcs.world_axis_units,
                hdr_wcs.wcs.cdelt,
                hdr_wcs.pixel_shape,
            )
        ):
            if i == hdr_wcs.wcs.lng:
                assert U.isclose(dc.px_size, -spacing * U.Unit(unit, format="fits"))
                assert dc.n_px_x == len_ax
                assert U.isclose(dc.ra, centre_coord * U.Unit(unit, format="fits"))
            elif i == hdr_wcs.wcs.lat:
                assert U.isclose(dc.px_size, spacing * U.Unit(unit, format="fits"))
                assert dc.n_px_y == len_ax
                assert U.isclose(dc.dec, centre_coord * U.Unit(unit, format="fits"))
            elif i == hdr_wcs.wcs.spec:
                # This breaks if spacing is in Hz,
                # need the frequency difference at the velocity centre:
                hdr_specref = centre_coord * U.Unit(unit, format="fits")
                assert U.isclose(
                    dc.channel_width,
                    np.abs(
                        (hdr_specref + spacing * U.Unit(unit, format="fits")).to(
                            dc.channel_width.unit, equivalencies=U.doppler_radio(HIfreq)
                        )
                        - (hdr_specref).to(
                            dc.channel_width.unit, equivalencies=U.doppler_radio(HIfreq)
                        )
                    ),
                )
                assert dc.n_channels == len_ax
                assert U.isclose(
                    dc.spectral_centre,
                    hdr_specref.to(
                        dc.spectral_centre.unit, equivalencies=U.doppler_radio(HIfreq)
                    ),
                )
