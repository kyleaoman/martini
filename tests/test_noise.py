import numpy as np
from martini.noise import GaussianNoise
from martini import DataCube
from astropy import units as U


class TestNoise:
    def test_noise_shape(self):
        """
        Check that we generate noise with correct shape.
        """
        rms = 1.0 * U.Jy * U.arcsec**-2
        noise_generator = GaussianNoise(rms=rms)
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        noise = noise_generator.generate(datacube)
        assert noise.shape == datacube._array.shape

    def test_noise_amplitude(self):
        """
        Check that we generate noise with correct amplitude.
        """
        rms = 1.0 * U.Jy * U.arcsec**-2
        noise_generator = GaussianNoise(rms=rms)
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        noise = noise_generator.generate(datacube)
        measured_rms = np.sqrt(np.mean(np.power(noise, 2)))
        assert U.isclose(measured_rms, rms, rtol=0.1)

    def test_noise_seed(self):
        """
        Check that if we use a seed we get repeatable results.
        """

        rms = 1.0 * U.Jy * U.arcsec**-2
        seed = 0
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        noise_generator1 = GaussianNoise(rms=rms, seed=seed)
        noise_generator2 = GaussianNoise(rms=rms, seed=seed)
        assert U.allclose(
            noise_generator1.generate(datacube), noise_generator2.generate(datacube)
        )

    def test_noise_noseed(self):
        """
        Check that if we use seed=None we get unpredictable results.
        """

        rms = 1.0 * U.Jy * U.arcsec**-2
        seed = None
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        noise_generator1 = GaussianNoise(rms=rms, seed=seed)
        noise_generator2 = GaussianNoise(rms=rms, seed=seed)
        assert not U.allclose(
            noise_generator1.generate(datacube), noise_generator2.generate(datacube)
        )

    def test_reset_rng(self):
        """
        Check that when we reset the rng we get the same results.
        """
        rms = 1.0 * U.Jy * U.arcsec**-2
        seed = 0
        datacube = DataCube(n_px_x=256, n_px_y=256, n_channels=64)
        noise_generator = GaussianNoise(rms=rms, seed=seed)
        noise1 = noise_generator.generate(datacube)
        noise_generator.reset_rng()
        noise2 = noise_generator.generate(datacube)

        assert noise_generator.seed is not None
        assert U.allclose(noise1, noise2)
