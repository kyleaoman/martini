import pytest
import numpy as np
from martini import DataCube
from martini.spectral_models import GaussianSpectrum, DiracDeltaSpectrum
from martini.sources import _SingleParticleSource, _CrossSource
from astropy import units as U

spectral_models = GaussianSpectrum, DiracDeltaSpectrum


def test_docstrings():
    raise RuntimeError("This file needs docstrings.")


class TestGaussianSpectrum:
    @pytest.mark.parametrize("sigma", ("thermal", 7.0 * U.km / U.s))
    def test_init_spectra(self, sigma):
        source = _SingleParticleSource(distance=1 * U.Mpc)  # D=1Mpc
        spectral_model = GaussianSpectrum(sigma=sigma)
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectral_model.init_spectra(source, datacube)
        expected_flux = (
            source.mHI_g[0] / 2.36e5 * U.Jy * U.km * U.s**-1 / U.Msun
        )  # D=1Mpc
        flux = spectral_model.spectra[0].sum() * datacube.channel_width
        assert U.isclose(flux, expected_flux, rtol=1.0e-5)

    def test_half_width_constant(self):
        sigma = 7.0 * U.km / U.s
        source = _SingleParticleSource()
        spectral_model = GaussianSpectrum(sigma=sigma)
        assert U.isclose(spectral_model.half_width(source), sigma)

    def test_half_width_thermal(self):
        expected_sigma = 9.0853727258 * U.km / U.s  # @1E4K
        source = _SingleParticleSource()
        spectral_model = GaussianSpectrum(sigma="thermal")
        assert U.isclose(spectral_model.half_width(source)[0], expected_sigma)

    @pytest.mark.parametrize("sigma", ("thermal", 7.0 * U.km / U.s))
    def test_spectral_function(self, sigma):
        source = _SingleParticleSource()
        spectral_model = GaussianSpectrum(sigma=sigma)
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectral_model.init_spectral_function_extra_data(source, datacube)
        spectrum = spectral_model.spectral_function(
            datacube.channel_edges[:-1],
            datacube.channel_edges[1:],
            U.Quantity([source.vsys]),  # expected from _SingleParticleSource
        )
        assert U.isclose(spectrum.sum(), 1.0 * U.dimensionless_unscaled, rtol=1.0e-4)

    @pytest.mark.parametrize("sigma", ("thermal", 7.0 * U.km / U.s))
    @pytest.mark.parametrize("source", (_SingleParticleSource(), _CrossSource()))
    def test_spectral_function_extra_data(self, sigma, source):
        spectral_model = GaussianSpectrum(sigma=sigma)
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectral_model.init_spectral_function_extra_data(source, datacube)
        extra_data = spectral_model.spectral_function_extra_data
        assert set(extra_data.keys()) == {"sigma"}
        for column in extra_data["sigma"].T:
            assert U.allclose(column, spectral_model.half_width(source))
        expected_rows = 1 if sigma != "thermal" else source.npart
        assert extra_data["sigma"].shape == (expected_rows, datacube.n_channels)


class TestDiracDeltaSpectrum:
    def test_init_spectra(self):
        source = _SingleParticleSource(distance=1 * U.Mpc)  # D=1Mpc
        spectral_model = DiracDeltaSpectrum()
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectral_model.init_spectra(source, datacube)
        expected_flux = (
            source.mHI_g[0] / 2.36e5 * U.Jy * U.km * U.s**-1 / U.Msun
        )  # D=1Mpc
        flux = spectral_model.spectra[0].sum() * datacube.channel_width
        assert U.isclose(flux, expected_flux, rtol=1.0e-5)

    def test_half_width(self):
        source = _SingleParticleSource()
        spectral_model = DiracDeltaSpectrum()
        assert U.isclose(spectral_model.half_width(source), 0 * U.km / U.s)

    def test_spectral_function(self):
        source = _SingleParticleSource()
        spectral_model = DiracDeltaSpectrum()
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectrum = spectral_model.spectral_function(
            datacube.channel_edges[:-1],
            datacube.channel_edges[1:],
            U.Quantity([source.vsys]),  # expected from _SingleParticleSource
        )
        assert U.isclose(spectrum.sum(), 1.0 * U.dimensionless_unscaled, rtol=1.0e-4)

    def test_spectral_function_extra_data(self):
        source = _SingleParticleSource()
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectral_model = DiracDeltaSpectrum()
        spectral_model.init_spectral_function_extra_data(source, datacube)
        extra_data = spectral_model.spectral_function_extra_data
        assert len(extra_data) == 0


class TestSpectrumPrecision:
    @pytest.mark.parametrize("SpectralModel", spectral_models)
    @pytest.mark.parametrize("dtype", (np.float64, np.float32))
    def test_spectrum_precision(self, SpectralModel, dtype):
        source = _SingleParticleSource()
        spectral_model = SpectralModel(spec_dtype=dtype)
        datacube = DataCube()
        spectral_model.init_spectra(source, datacube)
        assert spectral_model.spectra.dtype == dtype
