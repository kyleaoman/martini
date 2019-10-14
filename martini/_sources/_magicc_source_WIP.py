import pynbody
import astropy.units as U
import astropy.constants as C
from kyleaoman_utilities.neutral_fractions import atomic_frac
import numpy as np
from martini import Martini, DataCube
from martini.beams import GaussianBeam
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import DiracDeltaKernel
from martini.sources import SPHSource
from multiprocessing import Pool
from astropy.coordinates.matrix_utilities import rotation_matrix


def generate_datacube(az_rot):

    fH = .75
    fHe = .25

    s = pynbody.load('snapshot/g1536.01024')
    h = s.halos()
    h1 = h[1]
    h1.physical_units()
    with pynbody.analysis.halo.center(h1, mode='hyb'):
        xyz_g = np.array(h1.gas['pos']) * U.kpc
        vxyz_g = np.array(h1.gas['vel']) * U.km / U.s
        m_g = np.array(h1.gas['mass']) * U.Msun
        rho_g = np.array(h1.gas['rho']) * U.Msun * U.kpc ** -3
        T_g = np.array(h1.gas['temp']) * U.K
        hsm_g = np.array(h1.gas['eps']) * U.kpc  # grav soft or sph smooth?
        Habundance_g = 1 - np.array(h1.gas['metals']) - fHe
        mu = 1 / (fH + .25 * fHe)
        mHI_g = atomic_frac(
            0,
            rho_g * Habundance_g / (mu * C.m_p),
            T_g,
            rho_g,
            Habundance_g,
            onlyA1=True
        ) * m_g

    source = SPHSource(
        distance=3.657 * U.Mpc,
        rotation={'L_coords': (60. * U.deg, az_rot * U.deg)},
        ra=0. * U.deg,
        dec=0. * U.deg,
        h=.7,
        T_g=T_g,
        mHI_g=mHI_g,
        xyz_g=xyz_g,
        vxyz_g=vxyz_g,
        hsm_g=hsm_g
    )

    datacube = DataCube(
        n_px_x=512,
        n_px_y=512,
        n_channels=100,
        px_size=6. * U.arcsec,
        channel_width=4. * U.km * U.s ** -1,
        velocity_centre=source.vsys
    )

    beam = GaussianBeam(
        bmaj=12. * U.arcsec,
        bmin=12. * U.arcsec,
        bpa=0. * U.deg,
        truncate=4.
    )

    spectral_model = GaussianSpectrum(
        sigma=7. * U.km / U.s
    )

    sph_kernel = DiracDeltaKernel()

    M = Martini(
        source=source,
        datacube=datacube,
        beam=beam,
        noise=None,
        spectral_model=spectral_model,
        sph_kernel=sph_kernel,
        logtag='[{:03d}]: '.format(az_rot)
    )

    M.insert_source_in_cube(skip_validation=True)
    M.convolve_beam()
    M.write_fits('fitsfiles/g1536.01024_az{:03d}.fits'.format(az_rot),
                 channels='velocity')
    return


az_rots = list(np.arange(0, 360, 15))
pool = Pool(24)
pool.map(generate_datacube, az_rots, chunksize=1)
