from . import Martini, DataCube
from .beams import GaussianBeam
from .noise import GaussianNoise
from .spectral_models import GaussianSpectrum
from .sph_kernels import CubicSplineKernel
from .sources import SPHSource
import astropy.units as U
import numpy as np
from scipy.optimize import fsolve


def demo(cubefile="testcube.fits", beamfile="testbeam.fits", hdf5file="testcube.hdf5"):
    """
    Demonstrates basic usage of MARTINI.

    Creates a (very!) crude toy model of a galaxy with a linearly rising
    rotation curve, exponential disk profile, exponential vertical structure. A
    basic configuration of MARTINI is initialized and used to create and output
    a datacube and an image of the beam.

    Parameters
    ----------
    cubefile : string
        File to write demonstration datacube.

    beamfile : string
        File to write demonstration beam.
    """

    # ------make a toy galaxy----------
    N = 500
    phi = np.random.rand(N) * 2 * np.pi
    r = []
    for L in np.random.rand(N):

        def f(r):
            return L - 0.5 * (2 - np.exp(-r) * (np.power(r, 2) + 2 * r + 2))

        r.append(fsolve(f, 1.0)[0])
    r = np.array(r)
    # exponential disk
    r *= 3 / np.sort(r)[N // 2]
    z = -np.log(np.random.rand(N))
    # exponential scale height
    z *= 0.5 / np.sort(z)[N // 2] * np.sign(np.random.rand(N) - 0.5)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    xyz_g = np.vstack((x, y, z)) * U.kpc
    # linear rotation curve
    vphi = 100 * r / 6.0
    vx = -vphi * np.sin(phi)
    vy = vphi * np.cos(phi)
    # small pure random z velocities
    vz = (np.random.rand(N) * 2.0 - 1.0) * 5
    vxyz_g = np.vstack((vx, vy, vz)) * U.km * U.s**-1
    T_g = np.ones(N) * 8e3 * U.K
    mHI_g = np.ones(N) / N * 5.0e9 * U.Msun
    # ~mean interparticle spacing smoothing
    hsm_g = np.ones(N) * 4 / np.sqrt(N) * U.kpc
    # ---------------------------------

    source = SPHSource(
        distance=3.0 * U.Mpc,
        rotation={"L_coords": (60.0 * U.deg, 0.0 * U.deg)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        h=0.7,
        T_g=T_g,
        mHI_g=mHI_g,
        xyz_g=xyz_g,
        vxyz_g=vxyz_g,
        hsm_g=hsm_g,
    )

    datacube = DataCube(
        n_px_x=128,
        n_px_y=128,
        n_channels=32,
        px_size=10.0 * U.arcsec,
        channel_width=10.0 * U.km * U.s**-1,
        velocity_centre=source.vsys,
    )

    beam = GaussianBeam(
        bmaj=30.0 * U.arcsec, bmin=30.0 * U.arcsec, bpa=0.0 * U.deg, truncate=4.0
    )

    noise = GaussianNoise(rms=3.0e-5 * U.Jy * U.beam**-1)

    spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

    sph_kernel = CubicSplineKernel()

    M = Martini(
        source=source,
        datacube=datacube,
        beam=beam,
        noise=noise,
        spectral_model=spectral_model,
        sph_kernel=sph_kernel,
    )

    M.insert_source_in_cube()
    M.add_noise()
    M.convolve_beam()
    M.write_beam_fits(beamfile, channels="velocity")
    M.write_fits(cubefile, channels="velocity")
    print(f"Wrote demo fits output to {cubefile}, and beam image to {beamfile}.")
    try:
        M.write_hdf5(hdf5file, channels="velocity")
    except ModuleNotFoundError:
        print("h5py package not present, skipping hdf5 output demo.")
    else:
        print(f"Wrote demo hdf5 output to {hdf5file}.")

    return
