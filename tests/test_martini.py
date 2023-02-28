import os
import pytest
import numpy as np
import h5py
from martini import Martini
from martini.datacube import HIfreq
from martini.beams import GaussianBeam
from martini.sources import _SingleParticleSource as SingleParticleSource
from test_sph_kernels import simple_kernels
from martini.spectral_models import DiracDeltaSpectrum, GaussianSpectrum
from astropy import units as U
from astropy.units import isclose
from astropy.io import fits

# Should probably actually just inline this into martini to monitor how
# much of the input mass ends up in the datacube?


class TestMartiniUtils:
    @pytest.mark.xfail
    def test_gen_particle_coords():
        raise NotImplementedError


class TestMartini:
    @pytest.mark.parametrize("sph_kernel", simple_kernels)
    @pytest.mark.parametrize("spectral_model", (DiracDeltaSpectrum, GaussianSpectrum))
    def test_mass_accuracy(self, dc, sph_kernel, spectral_model):
        """
        Check that the input mass in the particles gives the correct total mass in the
        datacube, by checking the conversion back to total mass.
        """

        hsm_g = (
            0.1 * U.kpc if sph_kernel.__name__ == "DiracDeltaKernel" else 1.0 * U.kpc
        )

        # SingleParticleSource has a mass of 1E4Msun, temperature of 1E4K
        m = Martini(
            source=SingleParticleSource(hsm_g=hsm_g),
            datacube=dc,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model(),
            sph_kernel=sph_kernel(),
        )

        m.insert_source_in_cube(printfreq=None)

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
        assert isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

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
        assert isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

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
                        dv = np.abs(
                            dchannel.to(
                                U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                            )
                            - (0 * U.Hz).to(
                                U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                            )
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
                assert isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

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
                        dv = np.abs(
                            dchannel.to(
                                U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
                            )
                            - (0 * U.Hz).to(
                                U.km / U.s, equivalencies=U.doppler_radio(HIfreq)
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
                    MHI = (
                        2.36e5
                        * U.Msun
                        * D.to_value(U.Mpc) ** 2
                        * F.to_value(U.Jy)
                        * dv.to_value(U.km / U.s)
                    ).to(U.Msun)

                # demand accuracy within 1% in output fits file
                assert isclose(MHI, m.source.mHI_g.sum(), rtol=1e-2)

            finally:
                if os.path.exists(filename):
                    os.remove(filename)

    @pytest.mark.xfail
    def test_convolve_beam():
        raise NotImplementedError

    @pytest.mark.xfail
    def test_add_noise():
        raise NotImplementedError

    @pytest.mark.xfail
    def test_prune_particles():
        raise NotImplementedError

    @pytest.mark.xfail
    def test_insert_source_in_cube():
        raise NotImplementedError

    def test_reset(self, m):
        assert m.datacube._array.sum() > 0
        m.reset()
        assert m.datacube._array.sum() == 0
