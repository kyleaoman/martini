import pytest
import numpy as np
from astropy import units as U
from martini.martini import Martini
from martini.datacube import DataCube
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.sources import _SingleParticleSource, SPHSource
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
        velocity_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
    )
    beam = GaussianBeam()
    noise = GaussianNoise(rms=1.0e-9 * U.Jy * U.arcsec**-2, seed=0)
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
    m.insert_source_in_cube(printfreq=None)
    m.add_noise()
    m.convolve_beam()
    yield m


@pytest.fixture(scope="function")
def m_init():

    source = _SingleParticleSource()
    datacube = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        velocity_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
    )
    beam = GaussianBeam()
    noise = GaussianNoise(rms=1.0e-9 * U.Jy * U.arcsec**-2, seed=0)
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
    yield m


@pytest.fixture(scope="function")
def m_nn():

    source = _SingleParticleSource()
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
    m.insert_source_in_cube(printfreq=None)
    m.convolve_beam()
    yield m


@pytest.fixture(scope="function")
def dc():

    dc = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        velocity_centre=3 * 70 * U.km / U.s,
    )

    dc._array[...] = (
        np.random.rand(dc._array.size).reshape(dc._array.shape) * dc._array.unit
    )

    yield dc


@pytest.fixture(scope="function")
def s():

    n_g = 1000
    phi = np.random.rand(n_g, 1) * 2 * np.pi
    R = np.random.rand(n_g, 1)
    xyz_g = (
        np.hstack(
            (
                R * np.cos(phi) * 3,
                R * np.sin(phi) * 3,
                (np.random.rand(n_g, 1) * 2 - 1) * 0.01,
            )
        )
        * U.kpc
    )
    vxyz_g = (
        np.hstack(
            (
                -R * np.sin(phi) * 100,
                R * np.cos(phi) * 100,
                (np.random.rand(n_g, 1) * 2 - 1) * 0.01,
            )
        )
        * U.km
        / U.s
    )
    T_g = np.ones(n_g) * 1e4 * U.K
    mHI_g = np.ones(n_g) * 1e9 * U.Msun / n_g
    hsm_g = 0.5 * U.kpc
    particles = dict(
        xyz_g=xyz_g,
        vxyz_g=vxyz_g,
        mHI_g=mHI_g,
        T_g=T_g,
        hsm_g=hsm_g,
    )
    s = SPHSource(**particles)

    yield s
