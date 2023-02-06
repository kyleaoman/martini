import numpy as np
from math import isclose
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
        measured_rms = np.sqrt(np.mean(np.power(noise.to_value(rms.unit), 2)))
        assert isclose(measured_rms, rms.to_value(rms.unit), rel_tol=0.1)
