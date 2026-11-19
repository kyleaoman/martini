"""Test the functionality of the noise modules."""

import pytest
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


class _ConvolvedNoise(GaussianNoise):
    """
    Noise already convolved with the beam, with units of flux density per beam.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to :class:`~martini.noise.GaussianNoise`.

    unit : ~astropy.units.UnitBase
        Units (of flux density per beam) of the generated noise.

    **kwargs : dict
        Keyword arguments passed to :class:`~martini.noise.GaussianNoise`.
    """

    def __init__(self, *args, unit=U.Jy * U.beam**-1, **kwargs):
        self.unit = unit
        super().__init__(*args, **kwargs)

    def generate(self, datacube, beam):
        """
        Create a cube containing (white) noise with the target rms per beam.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            The data cube.

        beam : ~martini.beams._BaseBeam
            The beam.

        Returns
        -------
        ~astropy.units.Quantity
            Noise realization in units of flux density per beam.
        """
        noise = self.rng.standard_normal(size=datacube._array.shape)
        return (noise * self.target_rms).to(self.unit)


class _PerAreaNoise(GaussianNoise):
    """Noise to be convolved with the beam, returned in Jy/arcsec^2."""

    def generate(self, datacube, beam):
        """
        Create a cube containing noise in units of Jy/arcsec^2.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            The data cube.

        beam : ~martini.beams._BaseBeam
            The beam.

        Returns
        -------
        ~astropy.units.Quantity
            Noise realization in Jy/arcsec^2.
        """
        noise = super().generate(datacube, beam)
        return noise.to(U.Jy * U.arcsec**-2, equivalencies=[datacube.arcsec2_to_pix])


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

    def test_per_area_units_converted(self, m_init):
        """Check that per-area noise in other units gives the same data cube."""
        rms = m_init.noise.target_rms
        initial = m_init.datacube._array.copy()
        m_init.noise = GaussianNoise(rms=rms, seed=5)
        m_init.add_noise()
        expected = m_init.datacube._array.copy()
        m_init.datacube._array[...] = initial
        m_init.noise = _PerAreaNoise(rms=rms, seed=5)
        m_init.add_noise()
        assert m_init.datacube._array.unit == expected.unit
        assert U.allclose(m_init.datacube._array, expected)

    @pytest.mark.parametrize("unit", (U.Jy * U.beam**-1, U.mJy * U.beam**-1))
    def test_convolved_noise_after_beam(self, m_init, unit):
        """Check that convolved noise is added after beam convolution."""
        rms = m_init.noise.target_rms
        m_init.insert_source_in_cube(progressbar=False)
        m_init.convolve_beam()
        before = m_init.datacube._array.copy()
        m_init.noise = _ConvolvedNoise(rms=rms, seed=7, unit=unit)
        m_init.add_noise()
        expected = _ConvolvedNoise(rms=rms, seed=7).generate(m_init.datacube, None)
        assert m_init.datacube._array.unit == U.Jy * U.beam**-1
        assert U.allclose(m_init.datacube._array - before, expected)

    def test_convolved_noise_before_beam(self, m_init):
        """Check that convolved noise added before beam convolution is an error."""
        m_init.noise = _ConvolvedNoise(rms=m_init.noise.target_rms)
        with pytest.raises(ValueError, match="after `convolve_beam`"):
            m_init.add_noise()

    def test_gaussian_noise_after_beam(self, m_init):
        """Check that noise to be convolved added after beam convolution is an error."""
        m_init.insert_source_in_cube(progressbar=False)
        m_init.convolve_beam()
        with pytest.raises(ValueError, match="before convolving the beam"):
            m_init.add_noise()

    def test_per_area_noise_after_beam(self, m_init):
        """Check the error from Martini for per-area noise after beam convolution."""
        m_init.insert_source_in_cube(progressbar=False)
        m_init.convolve_beam()
        m_init.noise = _PerAreaNoise(rms=m_init.noise.target_rms)
        m_init.noise.generate = lambda datacube, beam: U.Quantity(  # bypass own check
            np.zeros(datacube._array.shape), U.Jy * U.arcsec**-2
        )
        with pytest.raises(ValueError, match="before `convolve_beam`"):
            m_init.add_noise()
