import os
import pytest
import numpy as np
import h5py
from martini.martini import Martini
from martini.datacube import DataCube, HIfreq
from martini.beams import GaussianBeam
from test_sph_kernels import simple_kernels
from martini.sph_kernels import CubicSplineKernel, GaussianKernel
from martini.spectral_models import DiracDeltaSpectrum, GaussianSpectrum
from astropy import units as U
from astropy.io import fits
from scipy.signal import fftconvolve


class TestMartini:
    @pytest.mark.parametrize("sph_kernel", simple_kernels)
    @pytest.mark.parametrize("spectral_model", (DiracDeltaSpectrum, GaussianSpectrum))
    def test_mass_accuracy(
        self, dc, sph_kernel, spectral_model, single_particle_source
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
            datacube=dc,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model(),
            sph_kernel=sph_kernel(),
        )

        m.insert_source_in_cube(progressbar=False)

        # flux
        F = m.datacube._array.sum() * m.datacube.px_size**2  # Jy

        # distance
        D = m.source.distance

        # channel width
        dv = m.datacube.channel_width

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

        m.convolve_beam()

        # radiant intensity
        Irad = m.datacube._array.sum()  # Jy / beam

        # beam area, for an equivalent Gaussian beam
        A = np.pi * m.beam.bmaj * m.beam.bmin / 4 / np.log(2) / U.beam

        # distance
        D = m.source.distance

        # channel width
        dv = m.datacube.channel_width

        # flux
        F = (Irad / A).to(U.Jy / U.arcsec**2) * m.datacube.px_size**2

        # HI mass
        MHI = (
            2.36e5
            * U.Msun
            * D.to_value(U.Mpc) ** 2
            * F.to_value(U.Jy)
            * dv.to_value(U.km / U.s)
        ).to(U.Msun)

        # demand accuracy within 1% after beam convolution
        assert U.isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

        for channel_mode in ("velocity", "frequency"):
            filename = f"cube_{channel_mode}.fits"
            try:
                m.write_fits(filename, channels=channel_mode)
                with fits.open(filename) as f:
                    # distance
                    D = m.source.distance

                    # radiant intensity
                    Irad = U.Quantity(f[0].data.sum(), unit=f[0].header["BUNIT"])

                    A = (
                        np.pi
                        * (f[0].header["BMAJ"] * U.deg)
                        * (f[0].header["BMIN"] * U.deg)
                        / 4
                        / np.log(2)
                        / U.beam
                    )
                    dchannel = np.abs(
                        U.Quantity(f[0].header["CDELT3"], unit=f[0].header["CUNIT3"])
                    )
                    if channel_mode == "velocity":
                        dv = dchannel
                    elif channel_mode == "frequency":
                        channelmid = U.Quantity(
                            f[0].header["CRVAL3"], unit=f[0].header["CUNIT3"]
                        )
                        dv = (channelmid - 0.5 * dchannel).to(
                            U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                        ) - (channelmid + 0.5 * dchannel).to(
                            U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                        )
                    px_area = U.Quantity(
                        np.abs(f[0].header["CDELT1"]), unit=f[0].header["CUNIT1"]
                    ) * U.Quantity(
                        np.abs(f[0].header["CDELT2"]), unit=f[0].header["CUNIT2"]
                    )

                    # flux
                    F = (Irad / A).to(U.Jy / U.arcsec**2) * px_area

                    # HI mass
                    MHI = (
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

        for channel_mode in ("velocity", "frequency"):
            filename = f"cube_{channel_mode}.hdf5"
            try:
                m.write_hdf5(filename, channels=channel_mode)
                with h5py.File(filename, "r") as f:
                    # distance
                    D = m.source.distance

                    # radiant intensity
                    Irad = U.Quantity(
                        f["FluxCube"][()].sum(),
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
                    dchannel = U.Quantity(
                        np.abs(f["FluxCube"].attrs["deltaV_in_VUnit"]),
                        unit=f["FluxCube"].attrs["VUnit"],
                    )
                    if channel_mode == "velocity":
                        dv = dchannel
                    elif channel_mode == "frequency":
                        channelmid = U.Quantity(
                            f["FluxCube"].attrs["V0_in_VUnit"],
                            unit=f["FluxCube"].attrs["VUnit"],
                        )
                        dv = (channelmid - 0.5 * dchannel).to(
                            U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                        ) - (channelmid + 0.5 * dchannel).to(
                            U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
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
                    MHI = (
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

    def test_convolve_beam(self, single_particle_source):
        """
        Check that beam convolution gives result matching manual calculation.
        """
        source = single_particle_source()
        datacube = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=16,
            velocity_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
        )
        beam = GaussianBeam()
        noise = None
        sph_kernel = GaussianKernel()
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
    def test_prune_particles(
        self, ra_off, ra_in, dec_off, dec_in, v_off, v_in, single_particle_source
    ):
        """
        Check that a particle offset by a specific set of (RA, Dec, v) is inside/outside
        the cube as expected.
        """
        expect_particle = all((ra_in, dec_in, v_in))
        # set distance so that 1kpc = 1arcsec
        distance = (1 * U.kpc / 1 / U.arcsec).to(U.Mpc, U.dimensionless_angles())
        source = single_particle_source(
            distance=distance, ra=ra_off, dec=dec_off, vpeculiar=v_off
        )
        datacube = DataCube(
            n_px_x=2,
            n_px_y=2,
            n_channels=2,
            velocity_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
            px_size=1 * U.arcsec,
            channel_width=1 * U.km / U.s,
            ra=0 * U.deg,
            dec=0 * U.deg,
        )
        # pad size will be 5, so datacube is 12x12 pixels
        beam = GaussianBeam(bmaj=1 * U.arcsec, bmin=1 * U.arcsec, truncate=4)
        sph_kernel = CubicSplineKernel()
        spectral_model = GaussianSpectrum(sigma=1 * U.km / U.s)
        kwargs = dict(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
        )
        # if more than 1px (datacube) + 5px (pad) + 2px (sm_range) then expect to prune
        # if more than 1px (datacube) + 4px (4*spectrum_half_width) then expect to prune
        if not expect_particle:
            with pytest.raises(
                RuntimeError, match="No source particles in target region."
            ):
                Martini(**kwargs)
        else:
            assert Martini(**kwargs).source.npart == 1

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
