import pytest
import os
import numpy as np
from astropy import units as U
from astropy import wcs
from martini.martini import Martini, GlobalProfile
from martini.datacube import DataCube, HIfreq
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.sources import SPHSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import _GaussianKernel


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
    in the Hubble flow assuming h = 0.7 at a distance of 3 Mpc.
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


def mps_sourcegen(
    T_g=np.ones(100) * 1.0e4 * U.K,
    mHI_g=np.ones(100) * 1.0e4 * U.Msun,
    xyz_g=(np.random.rand(300).reshape((100, 3)) - 0.5) * 10 * U.kpc,
    vxyz_g=(np.random.rand(300).reshape((100, 3)) - 0.5) * 40 * U.km * U.s**-1,
    hsm_g=np.ones(100) * U.kpc,
    distance=3 * U.Mpc,
    ra=0 * U.deg,
    dec=0 * U.deg,
    vpeculiar=0 * U.km / U.s,
):
    """
    Creates a 100-particle test source.

    A simple test source consisting of 100 particles will be created. The
    particles have a mass of 10^4 Msun, a SPH smoothing length of 1 kpc, a
    temperature of 10^4 K, random positions between -5 and 5 kpc offset from the source
    centroid along each axis, velocity offsets between -20 and 20 km/s around systemic,
    a peculiar velocity of 0 km/s, and will be placed in the Hubble flow assuming
    h = 0.7 at a distance of 3 Mpc.
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


def adaptive_kernel_test_sourcegen(
    T_g=np.ones(4) * 1.0e4 * U.K,
    mHI_g=np.ones(4) * 1.0e4 * U.Msun,
    xyz_g=np.ones((4, 3)) * 1.0e-3 * U.kpc,
    vxyz_g=np.zeros((4, 3)) * U.km * U.s**-1,
    hsm_g=np.array([3.0, 1.0, 0.55, 0.1]) * U.kpc,
    distance=3 * U.Mpc,
    ra=0 * U.deg,
    dec=0 * U.deg,
    vpeculiar=0 * U.km / U.s,
):
    """
    Creates a 4-particle test source.

    A simple test source consisting of 4 particles will be created. The
    particles have a mass of 10^4 Msun, a temperature of 10^4 K, a position offset by
    (x, y, z) = (1 pc, 1 pc, 1 pc) from the source centroid, a peculiar velocity of
    0 km/s, and will be placed in the Hubble flow assuming h = 0.7 at a distance of
    3 Mpc. The smoothing lengths are respectively: (4.0, 3.0, 1.0, 0.1) kpc. Normally
    the first two should use the preferred kernel (except for very small _GaussianKernel
    truncations where only the first would work), the third should fall back to
    a _GaussianKernel with a large truncation radius, and the last should fall back to
    a DiracDeltaKernel. Assumes 1kpc pixels, which is what we'll use for testing.
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


@pytest.fixture(scope="function", params=[True, False])
def m(request):
    source = sps_sourcegen()
    spectral_centre = source.distance * source.h * 100 * U.km / U.s / U.Mpc
    channel_width = 4 * U.km / U.s
    if request.param:
        channel_width = np.abs(
            (spectral_centre + 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
            - (spectral_centre - 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
        )
    datacube = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        spectral_centre=spectral_centre,
    )
    beam = GaussianBeam(bmaj=20.0 * U.arcsec, bmin=15.0 * U.arcsec)
    noise = GaussianNoise(rms=1.0e-9 * U.Jy * U.beam**-1, seed=0)
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
    m.insert_source_in_cube(progressbar=False)
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
        spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
    )
    beam = GaussianBeam(bmaj=20.0 * U.arcsec, bmin=15.0 * U.arcsec)
    noise = GaussianNoise(rms=1.0e-9 * U.Jy * U.beam**-1, seed=0)
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
    yield m


@pytest.fixture(scope="function")
def m_nn():
    source = sps_sourcegen()
    datacube = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
    )
    beam = GaussianBeam(bmaj=20.0 * U.arcsec, bmin=15.0 * U.arcsec)
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
    m.insert_source_in_cube(progressbar=False)
    m.convolve_beam()
    yield m


@pytest.fixture(
    scope="function",
    params=[(True, True), (True, False), (False, True), (False, False)],
)
def dc_random(request):
    stokes_axis, freq_channels = request.param
    spectral_centre = 3 * 70 * U.km / U.s
    channel_width = 4 * U.km / U.s
    if freq_channels:
        channel_width = np.abs(
            (spectral_centre + 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
            - (spectral_centre - 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
        )
        spectral_centre = spectral_centre.to(
            U.Hz, equivalencies=U.doppler_radio(HIfreq)
        )
    dc = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        channel_width=channel_width,
        spectral_centre=spectral_centre,
        stokes_axis=stokes_axis,
    )

    dc._array[...] = (
        np.random.rand(dc._array.size).reshape(dc._array.shape) * dc._array.unit
    )

    yield dc


@pytest.fixture(
    scope="function",
    params=[
        "IC_2574_NA_CUBE_THINGS.HEADER",
        "comb_10tracks_J1337_28_HI_r10_t90_mg095_2.image.header",
        "WALLABY_PILOT_CUTOUT_APPROX.HEADER",
        # as previous but with axes reordered:
        "comb_10tracks_J1337_28_HI_r10_t90_mg095_2.image.header.permuted",
    ],
)
def dc_wcs(request):
    with open(os.path.join("tests/data/", request.param), "r") as f:
        hdr = f.read()
    with pytest.warns(wcs.FITSFixedWarning):
        hdr_wcs = wcs.WCS(hdr)
    dc = DataCube.from_wcs(hdr_wcs)
    yield dc


@pytest.fixture(
    scope="function",
    params=[(True, True), (True, False), (False, True), (False, False)],
)
def dc_zeros(request):
    stokes_axis, freq_channels = request.param
    spectral_centre = 3 * 70 * U.km / U.s
    channel_width = 4 * U.km / U.s
    if freq_channels:
        channel_width = np.abs(
            (spectral_centre + 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
            - (spectral_centre - 0.5 * channel_width).to(
                U.Hz, equivalencies=U.doppler_radio(HIfreq)
            )
        )
        spectral_centre = spectral_centre.to(
            U.Hz, equivalencies=U.doppler_radio(HIfreq)
        )
    dc = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        channel_width=channel_width,
        spectral_centre=spectral_centre,
        stokes_axis=stokes_axis,
    )

    yield dc


@pytest.fixture(scope="function")
def adaptive_kernel_test_datacube():
    dc = DataCube(
        n_px_x=16,
        n_px_y=16,
        n_channels=16,
        px_size=((1 * U.kpc) / (3 * U.Mpc)).to(U.arcsec, U.dimensionless_angles()),
        spectral_centre=3 * 70 * U.km / U.s,
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


@pytest.fixture(scope="function")
def many_particle_source():
    yield mps_sourcegen


@pytest.fixture(scope="function")
def adaptive_kernel_test_source():
    yield adaptive_kernel_test_sourcegen


@pytest.fixture(scope="function")
def gp():
    source = sps_sourcegen()
    spectral_model = GaussianSpectrum()

    m = GlobalProfile(
        source=source,
        spectral_model=spectral_model,
        n_channels=16,
        spectral_centre=source.distance * source.h * 100 * U.km / U.s / U.Mpc,
    )
    m.insert_source_in_spectrum()
    yield m
