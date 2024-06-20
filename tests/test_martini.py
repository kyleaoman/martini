import os
import pytest
import numpy as np
from martini.martini import Martini, GlobalProfile, _BaseMartini
from martini.datacube import DataCube, HIfreq
from martini.beams import GaussianBeam
from test_sph_kernels import simple_kernels
from martini.sph_kernels import _CubicSplineKernel, _GaussianKernel, DiracDeltaKernel
from martini.spectral_models import DiracDeltaSpectrum, GaussianSpectrum
from astropy import units as U
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import FK5, ICRS
from scipy.signal import fftconvolve

try:
    import multiprocess
except ImportError:
    have_multiprocess = False
else:
    have_multiprocess = True

try:
    import matplotlib
except ImportError:
    have_matplotlib = False
else:
    have_matplotlib = True


def check_mass_accuracy(m, out_mode):
    if out_mode == "hdf5":
        try:
            import h5py
        except ImportError:
            pytest.skip()

    # flux in channels
    F = (m.datacube._array * m.datacube.px_size**2).sum((0, 1)).squeeze()  # Jy

    # distance
    D = m.source.distance

    # channel width
    dv = np.abs(np.diff(m.datacube.velocity_channel_edges))

    # HI mass
    MHI = np.sum(
        2.36e5
        * U.Msun
        * D.to_value(U.Mpc) ** 2
        * F.to_value(U.Jy)
        * dv.to_value(U.km / U.s)
    ).to(U.Msun)

    # demand accuracy within 1% after source insertion
    assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

    m.convolve_beam()

    # radiant intensity
    Irad = m.datacube._array.sum((0, 1)).squeeze()  # Jy / beam

    # beam area, for an equivalent Gaussian beam
    A = np.pi * m.beam.bmaj * m.beam.bmin / 4 / np.log(2) / U.beam

    # distance
    D = m.source.distance

    # channel width
    dv = np.abs(np.diff(m.datacube.velocity_channel_edges))

    # flux
    F = (Irad / A).to(U.Jy / U.arcsec**2) * m.datacube.px_size**2

    # HI mass
    MHI = np.sum(
        2.36e5
        * U.Msun
        * D.to_value(U.Mpc) ** 2
        * F.to_value(U.Jy)
        * dv.to_value(U.km / U.s)
    ).to(U.Msun)

    # demand accuracy within 1% after beam convolution
    assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

    if out_mode == "fits":
        filename = "cube.fits"
        try:
            m.write_fits(filename)
            with fits.open(filename) as f:
                # distance
                D = m.source.distance

                # radiant intensity
                fits_wcs = wcs.WCS(f[0].header)
                Irad = U.Quantity(
                    f[0].data.T.sum((0, 1)).squeeze(), unit=f[0].header["BUNIT"]
                )

                A = (
                    np.pi
                    * (f[0].header["BMAJ"] * U.deg)
                    * (f[0].header["BMIN"] * U.deg)
                    / 4
                    / np.log(2)
                    / U.beam
                )
                px_area = U.Quantity(
                    np.abs(f[0].header["CDELT1"]), unit=f[0].header["CUNIT1"]
                ) * U.Quantity(
                    np.abs(f[0].header["CDELT2"]), unit=f[0].header["CUNIT2"]
                )

                # flux
                F = (Irad / A).to(U.Jy / U.arcsec**2) * px_area

                channel_edges = fits_wcs.sub(("spectral",)).all_pix2world(
                    np.arange(fits_wcs.sub(("spectral",)).pixel_shape[0] + 1) - 0.5,
                    0,
                ) * U.Unit(fits_wcs.wcs.cunit[fits_wcs.wcs.spec], format="fits")
                dv = np.abs(
                    np.diff(
                        channel_edges.squeeze().to(
                            U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                        )
                    )
                )

                # HI mass
                MHI = np.sum(
                    2.36e5
                    * U.Msun
                    * D.to_value(U.Mpc) ** 2
                    * F.to_value(U.Jy)
                    * dv.to_value(U.km / U.s)
                ).to(U.Msun)

            # demand accuracy within 1% in output fits file
            assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

        finally:
            if os.path.exists(filename):
                os.remove(filename)

    if out_mode == "hdf5":
        filename = "cube.hdf5"
        try:
            m.write_hdf5(filename)
            with h5py.File(filename, "r") as f:
                # distance
                D = m.source.distance

                # radiant intensity
                Irad = U.Quantity(
                    f["FluxCube"][()].sum((0, 1)).squeeze(),
                    unit=f["FluxCube"].attrs["FluxCubeUnit"],
                )

                A = (
                    np.pi
                    * (f["FluxCube"].attrs["BeamMajor_in_deg"] * U.deg)
                    * (f["FluxCube"].attrs["BeamMinor_in_deg"] * U.deg)
                    / 4
                    / np.log(2)
                    / U.beam
                )
                dv = np.abs(
                    np.diff(
                        f["velocity_channel_edges"]
                        * U.Unit(f["velocity_channel_edges"].attrs["Unit"])
                    )
                )
                px_area = U.Quantity(
                    np.abs(f["FluxCube"].attrs["deltaRA_in_RAUnit"]),
                    unit=f["FluxCube"].attrs["RAUnit"],
                ) * U.Quantity(
                    np.abs(f["FluxCube"].attrs["deltaDec_in_DecUnit"]),
                    unit=f["FluxCube"].attrs["DecUnit"],
                )

                # flux
                F = (Irad / A).to(U.Jy / U.arcsec**2) * px_area

                # HI mass
                MHI = np.sum(
                    2.36e5
                    * U.Msun
                    * D.to_value(U.Mpc) ** 2
                    * F.to_value(U.Jy)
                    * dv.to_value(U.km / U.s)
                ).to(U.Msun)

            # demand accuracy within 1% in output hdf5 file
            assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestMartini:
    @pytest.mark.parametrize("sph_kernel", simple_kernels)
    @pytest.mark.parametrize("spectral_model", (DiracDeltaSpectrum, GaussianSpectrum))
    @pytest.mark.parametrize("out_mode", ("fits", "hdf5"))
    def test_mass_accuracy(
        self, dc_zeros, sph_kernel, spectral_model, single_particle_source, out_mode
    ):
        """
        Check that the input mass in the particles gives the correct total mass in the
        datacube, by checking the conversion back to total mass. Covers testing
        Martini.insert_source_in_cube.
        """

        hsm_g = (
            0.1 * U.kpc if sph_kernel.__name__ == "DiracDeltaKernel" else 1.0 * U.kpc
        )
        source = single_particle_source(hsm_g=hsm_g)
        # single_particle_source has a mass of 1E4Msun, temperature of 1E4K
        m = Martini(
            source=source,
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model(),
            sph_kernel=sph_kernel(),
        )
        m.insert_source_in_cube(progressbar=False)
        check_mass_accuracy(m, out_mode)

    def test_convolve_beam(self, single_particle_source):
        """
        Check that beam convolution gives result matching manual calculation.
        """
        source = single_particle_source()
        datacube = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
        )
        beam = GaussianBeam()
        noise = None
        sph_kernel = _GaussianKernel()
        spectral_model = GaussianSpectrum()

        m = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
        )
        m.insert_source_in_cube()
        unconvolved_cube = m.datacube._array.copy()
        unit = unconvolved_cube.unit
        s = np.s_[..., 0] if m.datacube.stokes_axis else np.s_[...]
        for spatial_slice in iter(unconvolved_cube[s].transpose((2, 0, 1))):
            spatial_slice[...] = (
                fftconvolve(spatial_slice, m.beam.kernel, mode="same") * unit
            )
        convolved_cube = unconvolved_cube[
            m.datacube.padx : -m.datacube.padx, m.datacube.pady : -m.datacube.padx
        ]
        convolved_cube = convolved_cube.to(
            U.Jy * U.beam**-1,
            equivalencies=U.beam_angular_area(m.beam.area),
        )
        m.convolve_beam()
        assert U.allclose(m.datacube._array, convolved_cube)

    def test_add_noise(self, m_init):
        """
        Check that noise provided goes into the datacube when we call add_noise.
        """
        assert (m_init.datacube._array.sum() == 0).all()
        assert m_init.noise.seed is not None
        expected_noise = m_init.noise.generate(m_init.datacube, m_init.beam)
        m_init.noise.reset_rng()
        m_init.add_noise()
        assert U.allclose(
            m_init.datacube._array,
            expected_noise.to(
                U.Jy * U.arcsec**-2,
                equivalencies=U.beam_angular_area(m_init.beam.area),
            ).to(
                m_init.datacube._array.unit,
                equivalencies=[m_init.datacube.arcsec2_to_pix],
            ),
        )

    @pytest.mark.parametrize(
        ("ra_off", "ra_in"),
        (
            (0 * U.arcsec, True),
            (3 * U.arcsec, True),
            (9 * U.arcsec, False),
            (-3 * U.arcsec, True),
            (-9 * U.arcsec, False),
        ),
    )
    @pytest.mark.parametrize(
        ("dec_off", "dec_in"),
        (
            (0 * U.arcsec, True),
            (3 * U.arcsec, True),
            (9 * U.arcsec, False),
            (-3 * U.arcsec, True),
            (-9 * U.arcsec, False),
        ),
    )
    @pytest.mark.parametrize(
        ("v_off", "v_in"),
        (
            (0 * U.km / U.s, True),
            (3 * U.km / U.s, True),
            (7 * U.km / U.s, False),
            (-3 * U.km / U.s, True),
            (-7 * U.km / U.s, False),
        ),
    )
    @pytest.mark.parametrize("spatial", (True, False))
    @pytest.mark.parametrize("spectral", (True, False))
    def test_prune_particles(
        self,
        ra_off,
        ra_in,
        dec_off,
        dec_in,
        v_off,
        v_in,
        single_particle_source,
        spatial,
        spectral,
    ):
        """
        Check that a particle offset by a specific set of (RA, Dec, v) is inside/outside
        the cube as expected.
        """
        if spatial and spectral:
            expect_particle = all((ra_in, dec_in, v_in))
        elif spatial and not spectral:
            expect_particle = all((ra_in, dec_in))
        elif spectral and not spatial:
            expect_particle = v_in
        elif not spectral and not spatial:
            expect_particle = True
        # set distance so that 1kpc = 1arcsec
        distance = (1 * U.kpc / 1 / U.arcsec).to(U.Mpc, U.dimensionless_angles())
        source = single_particle_source(
            distance=distance, ra=ra_off, dec=dec_off, vpeculiar=v_off
        )
        datacube = DataCube(
            n_px_x=2,
            n_px_y=2,
            n_channels=2,
            spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
            ra=0 * U.deg,
            dec=0 * U.deg,
        )
        # pad size will be 5, so datacube is 12x12 pixels
        beam = GaussianBeam(bmaj=1 * U.arcsec, bmin=1 * U.arcsec, truncate=4)
        sph_kernel = _CubicSplineKernel()
        spectral_model = GaussianSpectrum(sigma=1 * U.km / U.s)
        # need to use _BaseMartini below to manipulate _prune_kwargs
        kwargs = dict(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
            _prune_kwargs=dict(spatial=spatial, spectral=spectral),
        )
        # if more than 1px (datacube) + 5px (pad) + 2px (sm_range) then expect to prune
        # if more than 1px (datacube) + 4px (4*spectrum_half_width) then expect to prune
        if not expect_particle:
            with pytest.raises(
                RuntimeError, match="No source particles in target region."
            ):
                _BaseMartini(**kwargs)
        else:
            assert _BaseMartini(**kwargs).source.npart == 1

    def test_reset(self, m_nn):
        """
        Check that resetting martini instance zeros out datacube.
        """
        cube_array = m_nn.datacube._array
        assert m_nn.datacube._array.sum() > 0
        m_nn.reset()
        assert m_nn.datacube._array.sum() == 0
        # check that can start over and get the same result w/o errors
        m_nn.insert_source_in_cube(progressbar=False)
        m_nn.convolve_beam()
        assert U.allclose(cube_array, m_nn.datacube._array)
        # check that can reset after doing nothing
        m_nn.reset()
        m_nn.reset()

    def test_reset_preserves_shape(self, single_particle_source, dc_zeros):
        m = Martini(
            source=single_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=DiracDeltaSpectrum(),
            sph_kernel=DiracDeltaKernel(),
        )
        expected_shape = m.datacube._array.shape
        m.reset()
        assert m.datacube._array.shape == expected_shape

    @pytest.mark.skipif(
        not have_matplotlib, reason="matplotlib (optional dependency) not available."
    )
    def test_preview(self, m_init):
        """
        Simply check that the preview visualisation runs without error.
        """
        # with default arguments
        with pytest.warns(UserWarning, match="singular"):
            # warning: single-particle source is used, so axis limits try to be equal
            m_init.preview()
        # with non-default arguments
        m_init.preview(
            max_points=1000,
            fig=2,
            lim="datacube",
            vlim="datacube",
            point_scaling="fixed",
            title="test",
        )

    def test_source_to_datacube_coord_transformation(self, single_particle_source):
        """
        Check that transformation is applied if source and datacube have different
        coordinate frames.
        """
        source = single_particle_source(hsm_g=0.01 * U.kpc)
        assert source.coordinate_frame.name == "icrs"
        datacube_icrs = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            channel_width=4 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=ICRS(),
        )
        m_icrs = Martini(
            source=source,
            datacube=datacube_icrs,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=DiracDeltaKernel(),
            spectral_model=DiracDeltaSpectrum(),
        )

        def centre_pixels_slice(m):
            datacube = m.datacube
            return m.datacube._array[
                datacube.n_px_x // 2
                - 1
                + datacube.padx : datacube.n_px_x // 2
                + 1
                + datacube.padx,
                datacube.n_px_y // 2
                - 1
                + datacube.pady : datacube.n_px_y // 2
                + 1
                + datacube.pady,
            ]

        assert np.sum(centre_pixels_slice(m_icrs).sum()) == 0
        m_icrs.insert_source_in_cube(progressbar=False)
        assert np.sum(centre_pixels_slice(m_icrs).sum()) > 0

        # ICRS is ~J2000 equinox. J1950 equinox is about a degree off,
        # so we should completely miss the cube (16 pix of 10 arcsec).
        datacube_fk5_J1950 = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            channel_width=4 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=FK5(equinox="J1950"),
        )
        with pytest.raises(RuntimeError, match="No source particles in target region."):
            Martini(
                source=source,
                datacube=datacube_fk5_J1950,
                beam=GaussianBeam(),
                noise=None,
                sph_kernel=DiracDeltaKernel(),
                spectral_model=DiracDeltaSpectrum(),
            )

    def test_source_to_datacube_specsys_transformation(self, single_particle_source):
        """
        Check that spectral reference transformation is applied if source and datacube
        have different specsys.
        """
        source = single_particle_source(hsm_g=0.01 * U.kpc)
        datacube_icrs = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            channel_width=4 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=ICRS(),
            specsys="icrs",
        )
        m_icrs = Martini(
            source=source,
            datacube=datacube_icrs,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=DiracDeltaKernel(),
            spectral_model=DiracDeltaSpectrum(),
        )

        def centre_channels_slice(m):
            datacube = m.datacube
            return m.datacube._array[
                :, :, datacube.n_channels // 2 - 1 : datacube.n_channels // 2 + 1
            ]

        assert np.sum(centre_channels_slice(m_icrs).sum()) == 0
        m_icrs.insert_source_in_cube(progressbar=False)
        assert np.sum(centre_channels_slice(m_icrs).sum()) > 0

        # ICRS and Galactocentric are offset by many km/s depending on direction
        # so with 4 channels of 1 km/s we should completely miss the cube
        datacube_galactocentric = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=4,
            channel_width=1 * U.km / U.s,
            px_size=10 * U.arcsec,
            spectral_centre=source.vsys,
            ra=source.ra,
            dec=source.dec,
            coordinate_frame=ICRS(),
            specsys="galactocentric",
        )
        assert datacube_galactocentric.wcs.wcs.specsys == "galactocentric"
        with pytest.raises(RuntimeError, match="No source particles in target region."):
            Martini(
                source=source,
                datacube=datacube_galactocentric,
                beam=GaussianBeam(),
                noise=None,
                sph_kernel=DiracDeltaKernel(),
                spectral_model=DiracDeltaSpectrum(),
            )


@pytest.mark.skipif(
    not have_multiprocess, reason="multiprocess (optional dependency) not available"
)
class TestParallel:
    def test_parallel_consistent_with_serial(self, many_particle_source, dc_zeros):
        """
        Check that running the source insertion loop in parallel gives the same result
        as running in serial.
        """

        m = Martini(
            source=many_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            sph_kernel=_GaussianKernel(),
            spectral_model=GaussianSpectrum(),
        )

        m.insert_source_in_cube(ncpu=1, progressbar=False)
        expected_result = m.datacube._array

        # check that we're not testing on a zero array
        assert m.datacube._array.sum() > 0

        m.reset()

        # check the reset was successful
        assert np.allclose(
            m.datacube._array.to_value(m.datacube._array.unit),
            0.0,
        )

        m.insert_source_in_cube(ncpu=2, progressbar=False)

        assert U.allclose(m.datacube._array, expected_result)


class TestGlobalProfile:
    @pytest.mark.parametrize("spectral_model", (DiracDeltaSpectrum, GaussianSpectrum))
    @pytest.mark.parametrize("ra", (0 * U.deg, 180 * U.deg))
    def test_mass_accuracy(self, spectral_model, single_particle_source, ra):
        """
        Check that the input mass in the particles gives the correct total mass in the
        spectrum, by checking the conversion back to total mass. Covers testing
        GlobalProfile.insert_source_in_spectrum.
        """

        # single_particle_source has a mass of 1E4Msun, temperature of 1E4K
        # we test both ra=0deg and ra=180deg to make sure all particles included
        source = single_particle_source(ra=ra)
        m = GlobalProfile(
            source=source,
            spectral_model=spectral_model(),
            n_channels=32,
            channel_width=10 * U.km * U.s**-1,
            spectral_centre=source.vsys,
        )

        m.insert_source_in_spectrum()

        # flux
        F = m.spectrum.sum()  # Jy

        # distance
        D = m.source.distance

        # channel width
        dv = m.channel_width

        # HI mass
        MHI = (
            2.36e5
            * U.Msun
            * D.to_value(U.Mpc) ** 2
            * F.to_value(U.Jy)
            * dv.to_value(U.km / U.s)
        ).to(U.Msun)

        # demand accuracy within 1% after source insertion
        assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

    @pytest.mark.parametrize(
        ("ra_off", "ra_in"),
        (
            (0 * U.arcsec, True),
            (3 * U.arcsec, True),
            (5 * U.deg, False),  # global profile uses 1 deg pixel
            (-3 * U.arcsec, True),
            (-5 * U.deg, False),  # global profile uses 1 deg pixel
        ),
    )
    @pytest.mark.parametrize(
        ("dec_off", "dec_in"),
        (
            (0 * U.arcsec, True),
            (3 * U.arcsec, True),
            (5 * U.deg, False),  # global profile uses 1 deg pixel
            (-3 * U.arcsec, True),
            (-5 * U.deg, False),  # global profile uses 1 deg pixel
        ),
    )
    @pytest.mark.parametrize(
        ("v_off", "v_in"),
        (
            (0 * U.km / U.s, True),
            (3 * U.km / U.s, True),
            (7 * U.km / U.s, False),
            (-3 * U.km / U.s, True),
            (-7 * U.km / U.s, False),
        ),
    )
    def test_prune_particles(
        self, ra_off, ra_in, dec_off, dec_in, v_off, v_in, single_particle_source
    ):
        """
        Check that a particle offset by a specific set of (RA, Dec, v) is inside/outside
        the cube as expected. GlobalProfile should ignore RA, Dec when pruning.
        """
        # GlobalProfile should ignore RA, Dec when pruning:
        expect_particle = v_in
        # set distance so that 1kpc = 1arcsec
        distance = (1 * U.kpc / 1 / U.arcsec).to(U.Mpc, U.dimensionless_angles())
        source = single_particle_source(
            distance=distance, ra=ra_off, dec=dec_off, vpeculiar=v_off
        )
        spectral_model = GaussianSpectrum(sigma=1 * U.km / U.s)
        kwargs = dict(
            source=source,
            spectral_model=spectral_model,
            n_channels=2,
            channel_width=1 * U.km / U.s,
            spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
        )
        # if more than 1px (datacube) + 4px (4*spectrum_half_width) then expect to prune
        if not expect_particle:
            with pytest.raises(
                RuntimeError, match="No source particles in target region."
            ):
                GlobalProfile(**kwargs)
        else:
            assert GlobalProfile(**kwargs).source.npart == 1

    def test_reset(self, gp):
        """
        Check that resetting global profile instance zeros out datacube and spectrum.
        """
        cube_array = gp._datacube._array
        assert gp._datacube._array.sum() > 0
        spectrum = gp.spectrum
        assert spectrum.sum() > 0
        gp.reset()
        assert gp._datacube._array.sum() == 0
        assert not hasattr(gp, "_spectrum")
        # check that can start over and get the same result w/o errors
        gp.insert_source_in_spectrum()
        assert U.allclose(cube_array, gp._datacube._array)
        assert U.allclose(spectrum, gp.spectrum)
        # check that can reset after doing nothing
        gp.reset()
        gp.reset()

    @pytest.mark.skipif(
        not have_matplotlib, reason="matplotlib (optional dependency) not available."
    )
    def test_preview(self, gp):
        """
        Simply check that the preview visualisation runs without error.
        """
        # with default arguments
        with pytest.warns(UserWarning, match="singular"):
            # warning: single-particle source is used, so axis limits try to be equal
            gp.preview()
        # with non-default arguments
        gp.preview(
            max_points=1000,
            fig=2,
            lim="datacube",
            vlim="datacube",
            point_scaling="fixed",
            title="test",
        )

    def test_channel_modes(self, single_particle_source):
        """
        Check that channels have expected units in both modes (frequency, velocity).
        """
        source = single_particle_source()
        channel_width = 10 * U.km * U.s**-1
        m = GlobalProfile(
            source=source,
            spectral_model=GaussianSpectrum(sigma="thermal"),
            n_channels=32,
            channel_width=channel_width,
            spectral_centre=source.vsys,
        )
        expected_units = channel_width.unit
        # these will raise if there's a problem:
        m.channel_edges.to(expected_units)
        m.channel_mids.to(expected_units)

    @pytest.mark.skipif(
        not have_matplotlib, reason="matplotlib (optional dependency) not available."
    )
    def test_view_spectrum(self, gp):
        """
        Simply check that plotting spectrum runs without error.
        """
        # with default arguments
        gp.plot_spectrum()
        # with non-default arguments
        gp.plot_spectrum(fig=2, title="test", show_vsys=False)


class TestMartiniWithDataCubeFromWCS:

    @pytest.mark.parametrize("out_mode", ("fits", "hdf5"))
    def test_source_insertion(self, dc_wcs, single_particle_source, out_mode):
        datacube = dc_wcs
        distance = (
            datacube.spectral_centre.to(
                U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
            )
            / (70 * U.km / U.s / U.Mpc)
        ).to(U.Mpc)
        source = single_particle_source(
            ra=datacube.ra,
            dec=datacube.dec,
            distance=distance,
            hsm_g=(3 * datacube.px_size * distance).to(
                U.kpc, equivalencies=U.dimensionless_angles()
            ),
        )
        beam = GaussianBeam(bmaj=3 * datacube.px_size, bmin=3 * datacube.px_size)
        m = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            spectral_model=GaussianSpectrum(sigma="thermal"),
            sph_kernel=_CubicSplineKernel(),
        )
        m.insert_source_in_cube(progressbar=False)
        check_mass_accuracy(m, out_mode)
