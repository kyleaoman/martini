import pytest
import numpy as np
from martini import Martini
from martini.beams import GaussianBeam
from martini.sources import _SingleParticleSource as SingleParticleSource
from martini.sph_kernels import DiracDeltaKernel
from martini.spectral_models import DiracDeltaSpectrum
from astropy import units as U

# Should probably actually just inline this into martini to monitor how
# much of the input mass ends up in the datacube?


class TestMartiniUtils:
    @pytest.mark.xfail
    def test_gen_particle_coords():
        raise NotImplementedError


class TestMartini:
    def test_unconvolved_mass_accuracy(self, dc):
        """
        Check that the input mass in the particles gives the correct total mass in the
        datacube, by checking the conversion back to total mass.
        """

        raise RuntimeError  # parametrize test over spectral models & sph kernels:
        spectral_model = DiracDeltaSpectrum()
        sph_kernel = DiracDeltaKernel()
        source = SingleParticleSource()
        raise RuntimeError  # avoid monkey-patching hsm_g here:
        source.hsm_g = np.array([0.1]) * U.kpc

        # SingleParticleSource has a mass of 1E4Msun,
        # smoothing length of 1kpc, temperature of 1E4K
        M = Martini(
            source=source,
            datacube=dc,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
        )

        M.insert_source_in_cube(printfreq=None)

        # radiant intensity
        F = M.datacube._array.sum()  # Jy / arcsec^2

        # distance
        D = M.source.distance

        # channel width
        dv = M.datacube.channel_width

        # flux
        F = F * M.datacube.px_size**2  # Jy

        # HI mass
        MHI = (
            2.36e5
            * U.Msun
            * D.to(U.Mpc).value ** 2
            * F.to(U.Jy).value
            * dv.to(U.km / U.s).value
        ).to(U.Msun)

        # demand accuracy within 1%
        assert np.isclose(
            MHI.to(U.Msun).value, M.source.mHI_g.sum().to(U.Msun).value, rtol=1e-2
        )

    def test_mass_accuracy(self, dc):
        """
        Check that the input mass in the particles gives the correct total flux density in
        the datacube, by checking the conversion back to total mass.
        """

        raise RuntimeError  # parametrize test over spectral models & sph kernels:
        spectral_model = DiracDeltaSpectrum()
        sph_kernel = DiracDeltaKernel()
        source = SingleParticleSource()
        raise RuntimeError  # avoid monkey-patching hsm_g here:
        source.hsm_g = np.array([0.1]) * U.kpc

        # SingleParticleSource has a mass of 1E4Msun,
        # smoothing length of 1kpc, temperature of 1E4K
        M = Martini(
            source=source,
            datacube=dc,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
        )

        M.insert_source_in_cube(printfreq=None)
        M.convolve_beam()

        # radiant intensity
        Irad = M.datacube._array.sum()  # Jy / beam

        # beam area, for an equivalent Gaussian beam
        A = np.pi * M.beam.bmaj * M.beam.bmin / 4 / np.log(2) / U.beam

        # distance
        D = M.source.distance

        # channel width
        dv = M.datacube.channel_width

        # flux
        F = (Irad / A).to(U.Jy / U.arcsec**2) * M.datacube.px_size**2

        # HI mass
        MHI = (
            2.36e5
            * U.Msun
            * D.to(U.Mpc).value ** 2
            * F.to(U.Jy).value
            * dv.to(U.km / U.s).value
        ).to(U.Msun)
        print("MHI", MHI)

        # demand accuracy within 1%
        assert np.isclose(
            MHI.to(U.Msun).value, M.source.mHI_g.sum().to(U.Msun).value, rtol=1e-2
        )

    @pytest.mark.xfail
    def test_fileoutput_mass_accuracy():
        raise NotImplementedError

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

    @pytest.mark.xfail
    def test_reset():
        raise NotImplementedError
