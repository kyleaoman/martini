import pytest
from math import isclose
from martini import DataCube
from martini.spectral_models import GaussianSpectrum, DiracDeltaSpectrum
from martini.sources import _SingleParticleSource
from astropy import units as U


class TestGaussianSpectrum:
    @pytest.mark.parametrize("sigma", ("thermal", 7.0 * U.km / U.s))
    def test_init_spectra(self, sigma):
        source = _SingleParticleSource(distance=1 * U.Mpc)  # D=1Mpc
        spectral_model = GaussianSpectrum(sigma=sigma)
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectral_model.init_spectra(source, datacube)
        expected_flux = source.mHI_g[0].to_value(U.Msun) / 2.36E5  # D=1Mpc
        flux = (
            spectral_model.spectra[0].to_value(U.Jy).sum()
            * datacube.channel_width.to_value(U.km / U.s)
        )
        assert isclose(flux, expected_flux, rel_tol=1.e-5)

    def test_half_width_constant(self):
        sigma = 7.0 * U.km / U.s
        source = _SingleParticleSource()
        spectral_model = GaussianSpectrum(sigma=sigma)
        assert isclose(
            spectral_model.half_width(source).to_value(sigma.unit),
            sigma.to_value(sigma.unit)
        )

    def test_half_width_thermal(self):
        expected_sigma = 9.0853727258 * U.km / U.s  # @1E4K
        source = _SingleParticleSource()
        spectral_model = GaussianSpectrum(sigma="thermal")
        assert isclose(
            spectral_model.half_width(source)[0].to_value(expected_sigma.unit),
            expected_sigma.to_value(expected_sigma.unit)
        )

    @pytest.mark.parametrize("sigma", ("thermal", 7.0 * U.km / U.s))
    def test_spectral_function(self, sigma):
        source = _SingleParticleSource()
        spectral_model = GaussianSpectrum(sigma=sigma)
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectrum = spectral_model.spectral_function(
            datacube.channel_edges[:-1],
            datacube.channel_edges[1:],
            U.Quantity([source.vsys]),  # expected from _SingleParticleSource
            **spectral_model.spectral_function_kwargs(source),
        )
        assert isclose(spectrum.sum(), 1.0, rel_tol=1.e-4)

    @pytest.mark.parametrize("sigma", ("thermal", 7.0 * U.km / U.s))
    def test_spectral_function_kwargs(self, sigma):
        source = _SingleParticleSource()
        spectral_model = GaussianSpectrum(sigma=sigma)
        kwargs = spectral_model.spectral_function_kwargs(source)
        assert set(kwargs.keys()) == {"sigma"}
        assert kwargs["sigma"] == spectral_model.half_width(source)


class TestDiracDeltaSpectrum:
    def test_init_spectra(self):
        source = _SingleParticleSource(distance=1 * U.Mpc)  # D=1Mpc
        spectral_model = DiracDeltaSpectrum()
        datacube = DataCube(
            n_channels=64, channel_width=4 * U.km / U.s, velocity_centre=source.vsys
        )
        spectral_model.init_spectra(source, datacube)
        expected_flux = source.mHI_g[0].to_value(U.Msun) / 2.36E5  # D=1Mpc
        flux = (
            spectral_model.spectra[0].to_value(U.Jy).sum()
            * datacube.channel_width.to_value(U.km / U.s)
        )
        assert isclose(flux, expected_flux, rel_tol=1.e-5)

    def test_half_width(self):
        source = _SingleParticleSource()
        spectral_model = DiracDeltaSpectrum()
        assert isclose(
            spectral_model.half_width(source).to_value(U.km / U.s),
            0
        )

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
            **spectral_model.spectral_function_kwargs(source)
        )
        assert isclose(spectrum.sum(), 1.0, rel_tol=1.e-4)

    def test_spectral_function_kwargs(self):
        source = _SingleParticleSource()
        spectral_model = DiracDeltaSpectrum()
        kwargs = spectral_model.spectral_function_kwargs(source)
        assert len(kwargs) == 0
