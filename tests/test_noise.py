import numpy as np
from martini.noise import GaussianNoise
from martini.beams import GaussianBeam
from martini import DataCube
from astropy import units as U


class TestNoise:
    def test_noise_shape(self):
        """
        Check that we generate noise with correct shape.
        """
        rms = 1.0 * U.Jy * U.beam**-1
        noise_generator = GaussianNoise(rms=rms)
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        beam = GaussianBeam()
        noise = noise_generator.generate(datacube, beam)
        assert noise.shape == datacube._array.shape

    def test_noise_amplitude(self, m_init):
        """
        Check that we generate noise with correct amplitude.
        """
        target_rms = m_init.noise.target_rms
        m_init.insert_source_in_cube(progressbar=False)
        m_init.datacube._array[...] = 0 * U.Jy * U.arcsec**-2
        m_init.add_noise()
        m_init.convolve_beam()
        measured_rms = np.sqrt(np.mean(np.power(m_init.datacube._array, 2)))
        assert U.isclose(measured_rms, target_rms, rtol=0.1)

    def test_noise_seed(self):
        """
        Check that if we use a seed we get repeatable results.
        """

        rms = 1.0 * U.Jy * U.beam**-1
        seed = 0
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        beam = GaussianBeam()
        noise_generator1 = GaussianNoise(rms=rms, seed=seed)
        noise_generator2 = GaussianNoise(rms=rms, seed=seed)
        assert U.allclose(
            noise_generator1.generate(datacube, beam),
            noise_generator2.generate(datacube, beam),
        )

    def test_noise_noseed(self):
        """
        Check that if we use seed=None we get unpredictable results.
        """

        rms = 1.0 * U.Jy * U.beam**-1
        seed = None
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        beam = GaussianBeam()
        noise_generator1 = GaussianNoise(rms=rms, seed=seed)
        noise_generator2 = GaussianNoise(rms=rms, seed=seed)
        assert not U.allclose(
            noise_generator1.generate(datacube, beam),
            noise_generator2.generate(datacube, beam),
        )

    def test_reset_rng(self):
        """
        Check that when we reset the rng we get the same results.
        """
        rms = 1.0 * U.Jy * U.beam**-1
        seed = 0
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        beam = GaussianBeam()
        noise_generator = GaussianNoise(rms=rms, seed=seed)
        noise1 = noise_generator.generate(datacube, beam)
        noise_generator.reset_rng()
        noise2 = noise_generator.generate(datacube, beam)

        assert noise_generator.seed is not None
        assert U.allclose(noise1, noise2)
