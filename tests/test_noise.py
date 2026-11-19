"""Test the functionality of the noise modules."""

import numpy as np
from martini.noise import GaussianNoise
from martini.beams import GaussianBeam
from martini import DataCube
from astropy import units as U


class TestNoise:
    """Test the functionality of the noise modules."""

    def test_noise_shape(self):
        """Check that we generate noise with correct shape."""
        rms = 1.0 * U.Jy * U.beam**-1
        noise_generator = GaussianNoise(rms=rms)
        datacube = DataCube(
            n_px_x=256,
            n_px_y=256,
            n_channels=64,
            px_size=15 * U.arcsec,
            channel_width=4 * U.km / U.s,
        )
        beam = GaussianBeam()
        noise = noise_generator.generate(datacube, beam)
        assert noise.shape == datacube._array.shape

    def test_noise_amplitude(self, m_init):
        """Check that we generate noise with correct amplitude."""
        target_rms = m_init.noise.target_rms
        m_init.insert_source_in_cube(progressbar=False)
        m_init.datacube._array[...] = 0 * U.Jy * U.arcsec**-2
        m_init.add_noise()
        m_init.convolve_beam()
        measured_rms = np.sqrt(np.mean(np.power(m_init.datacube._array, 2)))
        assert U.isclose(measured_rms, target_rms, rtol=0.1)

    def test_noise_seed(self):
        """Check that if we use a seed we get repeatable results."""
        rms = 1.0 * U.Jy * U.beam**-1
        seed = 0
        datacube = DataCube(
            n_px_x=256,
            n_px_y=256,
            n_channels=64,
            px_size=15 * U.arcsec,
            channel_width=4 * U.km / U.s,
        )
        beam = GaussianBeam()
        noise_generator1 = GaussianNoise(rms=rms, seed=seed)
        noise_generator2 = GaussianNoise(rms=rms, seed=seed)
        assert U.allclose(
            noise_generator1.generate(datacube, beam),
            noise_generator2.generate(datacube, beam),
        )

    def test_noise_noseed(self):
        """Check that if we use seed=None we get unpredictable results."""
        rms = 1.0 * U.Jy * U.beam**-1
        seed = None
        datacube = DataCube(
            n_px_x=256,
            n_px_y=256,
            n_channels=64,
            px_size=15 * U.arcsec,
            channel_width=4 * U.km / U.s,
        )
        beam = GaussianBeam()
        noise_generator1 = GaussianNoise(rms=rms, seed=seed)
        noise_generator2 = GaussianNoise(rms=rms, seed=seed)
        assert not U.allclose(
            noise_generator1.generate(datacube, beam),
            noise_generator2.generate(datacube, beam),
        )

    def test_reset_rng(self):
        """Check that when we reset the rng we get the same results."""
        rms = 1.0 * U.Jy * U.beam**-1
        seed = 0
        datacube = DataCube(
            n_px_x=256,
            n_px_y=256,
            n_channels=64,
            px_size=15 * U.arcsec,
            channel_width=4 * U.km / U.s,
        )
        beam = GaussianBeam()
        noise_generator = GaussianNoise(rms=rms, seed=seed)
        noise1 = noise_generator.generate(datacube, beam)
        noise_generator.reset_rng()
        noise2 = noise_generator.generate(datacube, beam)

        assert noise_generator.seed is not None
        assert U.allclose(noise1, noise2)


class _JyPerBeamNoise(GaussianNoise):
    """Gaussian noise returned in Jy/beam, as a user-defined noise model might do."""

    def generate(self, datacube, beam):
        """
        Create a cube containing Gaussian noise in units of Jy/beam.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            The data cube.

        beam : ~martini.beams._BaseBeam
            The beam.

        Returns
        -------
        ~astropy.units.Quantity
            Noise realization in Jy/beam.
        """
        noise = super().generate(datacube, beam)
        return noise.to(
            U.Jy * U.arcsec**-2, equivalencies=[datacube.arcsec2_to_pix]
        ).to(U.Jy * U.beam**-1, equivalencies=U.beam_angular_area(beam.area))


class TestNoiseUnits:
    """Test that noise is generated in, or converted to, the data cube's units."""

    def test_generated_in_cube_units(self):
        """Check noise is generated in the cube's units, matching the Jy/beam noise."""
        rms = 1.0 * U.Jy * U.beam**-1
        seed = 3
        datacube = DataCube(
            n_px_x=32,
            n_px_y=32,
            n_channels=8,
            px_size=15 * U.arcsec,
            channel_width=4 * U.km / U.s,
        )
        beam = GaussianBeam()
        beam.init_kernel(datacube)
        noise = GaussianNoise(rms=rms, seed=seed).generate(datacube, beam)
        assert noise.unit == datacube.current_units
        # expected: noise drawn in Jy/beam (pre-convolution rms), then converted:
        sig_maj = (beam.bmaj / 2 / np.sqrt(2 * np.log(2)) / datacube.px_size).to_value(
            U.dimensionless_unscaled
        )
        sig_min = (beam.bmin / 2 / np.sqrt(2 * np.log(2)) / datacube.px_size).to_value(
            U.dimensionless_unscaled
        )
        rms_pre = rms * 2.19568 * np.sqrt(np.pi * sig_maj * sig_min)
        expected = rms_pre * np.random.default_rng(seed).standard_normal(
            size=datacube._array.shape, dtype=datacube.cube_dtype
        )
        expected = expected.to(
            U.Jy * U.arcsec**-2, equivalencies=U.beam_angular_area(beam.area)
        ).to(datacube.current_units, equivalencies=[datacube.arcsec2_to_pix])
        assert U.allclose(noise, expected)

    def test_other_units_converted(self, m_init):
        """Check that noise models returning Jy/beam give the same data cube."""
        rms = m_init.noise.target_rms
        initial = m_init.datacube._array.copy()
        m_init.noise = GaussianNoise(rms=rms, seed=5)
        m_init.add_noise()
        expected = m_init.datacube._array.copy()
        m_init.datacube._array[...] = initial
        m_init.noise = _JyPerBeamNoise(rms=rms, seed=5)
        m_init.add_noise()
        assert m_init.datacube._array.unit == expected.unit
        assert U.allclose(m_init.datacube._array, expected)
