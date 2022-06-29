import pytest
from astropy import units as U
from martini.martini import Martini
from martini.datacube import DataCube
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.sources import _SingleParticleSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import GaussianKernel

GaussianKernel.noFWHMwarn = True


@pytest.fixture(scope="function")
def m():

    source = _SingleParticleSource()
    datacube = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        velocity_centre=source.distance * 70 * U.km / U.s / U.Mpc
    )
    beam = GaussianBeam()
    noise = GaussianNoise(rms=1.e-9 * U.Jy * U.arcsec ** -2)
    sph_kernel = GaussianKernel()
    spectral_model = GaussianSpectrum()

    M = Martini(
        source=source,
        datacube=datacube,
        beam=beam,
        noise=noise,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model,
    )
    M.insert_source_in_cube()
    M.add_noise()
    M.convolve_beam()
    yield M
