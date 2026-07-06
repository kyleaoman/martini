import matplotlib.pyplot as plt
import numpy as np
from martini.mochi import Mochi
from martini.martini import Martini
from martini.beams import GaussianBeam
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import CubicSplineKernel
import astropy.units as U
from copy import deepcopy


class TestMochi:
    def test_scratch(self, many_particle_source, dc_zeros):
        # This will try to run with 4 variants of dc_zeros.
        # Use dc_zeros1 (stokes + vchannels) for now.
        martini_dc = deepcopy(dc_zeros)
        hsm_g = 0.3 * U.kpc
        mochi = Mochi(
            source=many_particle_source(
                hsm_g=hsm_g,
            ),  # refined cube changes by factors of 2 with this choice
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=GaussianSpectrum(),
            sph_kernel=CubicSplineKernel(),
        )
        mochi.insert_source_in_cube()

        martini = Martini(
            source=many_particle_source(hsm_g=hsm_g),
            datacube=martini_dc,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=GaussianSpectrum(),
            sph_kernel=CubicSplineKernel(),
        )
        martini.insert_source_in_cube()

        plt.imsave("martini.png", np.sum(martini.datacube._array[..., 0], axis=2))
        plt.imsave("mochi.png", np.sum(mochi.datacube._array, axis=2))
        # assert np.allclose(
        #     martini.datacube._array[..., 0].to_value(U.Jy / U.arcsec**2),
        #     mochi.datacube._array.to_value(U.Jy / U.beam),
        # )
