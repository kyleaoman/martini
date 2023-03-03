import pytest
import numpy as np
from astropy import units as U
from martini.martini import Martini
from martini.datacube import DataCube
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.sources import SPHSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import GaussianKernel

GaussianKernel.noFWHMwarn = True


def sps_sourcegen(
    T_g=np.ones(1) * 1.0e4 * U.K,
    mHI_g=np.ones(1) * 1.0e4 * U.Msun,
    xyz_g=np.ones((1, 3)) * 1.0e-3 * U.kpc,
    vxyz_g=np.zeros((1, 3)) * U.km * U.s**-1,
    hsm_g=np.ones(1) * U.kpc,
    distance=3 * U.Mpc,
    ra=0 * U.deg,
    dec=0 * U.deg,
    vpeculiar=0 * U.km / U.s,
):
    """
    Creates a single particle test source.

    A simple test source consisting of a single particle will be created. The
    particle has a mass of 10^4 Msun, a SPH smoothing length of 1 kpc, a
    temperature of 10^4 K, a position offset by (x, y, z) = (1 pc, 1 pc, 1 pc)
    from the source centroid, a peculiar velocity of 0 km/s, and will be placed
    in the Hubble flow assuming h = 0.7 at a distance of 3 Mpc. The particle has
    a 1 kpc smoothing length.
    """
    return SPHSource(
        T_g=T_g,
        mHI_g=mHI_g,
        xyz_g=xyz_g,
        vxyz_g=vxyz_g,
        hsm_g=hsm_g,
        distance=distance,
        ra=ra,
        dec=dec,
        vpeculiar=vpeculiar,
    )


def cross_sourcegen(
    T_g=np.arange(4) * 1.0e4 * U.K,
    mHI_g=np.ones(4) * 1.0e4 * U.Msun,
    xyz_g=np.array([[0, 1, 0], [0, 0, 2], [0, -3, 0], [0, 0, -4]]) * U.kpc,
    vxyz_g=np.array([[0, 0, 1], [0, -1, 0], [0, 0, -1], [0, 1, 0]]) * U.km * U.s**-1,
    hsm_g=np.ones(4) * U.kpc,
    distance=3 * U.Mpc,
    ra=0 * U.deg,
    dec=0 * U.deg,
    vpeculiar=0 * U.km / U.s,
):
    """
    Creates a source consisting of 4 particles arrayed in an asymmetric cross.

    A simple test source consisting of four particles will be created. Each has
    a mass of 10^4 Msun, a SPH smoothing length of 1 kpc, and will be placed in the Hubble
    flow assuming h=.7 and a distance of 3 Mpc. Particle coordinates in kpc are
    [[0,  1,  0],
    [0,  0,  2],
    [0, -3,  0],
    [0,  0, -4]]
    and velocities in km/s are
    [[0,  0,  1],
    [0, -1,  0],
    [0,  0, -1],
    [0,  1,  0]]
    Particles temperatures are [1, 2, 3, 4] * 1e4 Kelvin.
    """
    return SPHSource(
        T_g=T_g,
        mHI_g=mHI_g,
        xyz_g=xyz_g,
        vxyz_g=vxyz_g,
        hsm_g=hsm_g,
        distance=distance,
        ra=ra,
        dec=dec,
        vpeculiar=vpeculiar,
    )


@pytest.fixture(scope="function")
def m():
    source = sps_sourcegen()
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
    source = sps_sourcegen()
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
    source = sps_sourcegen()
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


@pytest.fixture(scope="function")
def cross_source():
    yield cross_sourcegen


@pytest.fixture(scope="function")
def single_particle_source():

    yield sps_sourcegen
