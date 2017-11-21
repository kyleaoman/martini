import numpy as np
import astropy.units as U
from astropy import constants as C
from scipy.special import erf as _erf
erf = lambda z: _erf(z.to(U.dimensionless_unscaled).value)
from abc import ABCMeta, abstractmethod

class _BaseSpectrum(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self.spectra = None
        return
        
    @abstractmethod
    def init_spectra(self, source, datacube):
        pass

    @abstractmethod
    def half_width(self, source):
        pass

class GaussianSpectrum(_BaseSpectrum):
    
    def __init__(self, sigma=7. * U.km * U.s ** -1):
        self.sigma_mode = sigma
        super(GaussianSpectrum, self).__init__()
        return

    def Gaussian_integral(self, a, b, mu=0.0, sigma=1.0):
        return .5 * (
            erf((b - mu) / (np.sqrt(2.) * sigma)) - \
            erf((a - mu) / (np.sqrt(2.) * sigma))
        )

    def init_spectra(self, source, datacube):
        channel_edges = datacube.channel_edges
        channel_widths = np.diff(channel_edges).to(U.km * U.s ** -1)
        mu = source.sky_coordinates.radial_velocity
        self.sigma = self.half_width(source)
        A = source.mHI_g * np.power(source.sky_coordinates.distance.to(U.Mpc), -2)
        MHI_Jy = (
            U.solMass * U.Mpc ** -2 * (U.km * U.s ** -1) ** -1, 
            U.Jy, 
            lambda x: (1 / 2.36E5) * x, 
            lambda x: 2.36E5 * x
        )
        self.spectra = (A[..., np.newaxis] * self.Gaussian_integral(
            np.tile(channel_edges[:-1], mu.shape + (1,)),
            np.tile(channel_edges[1:], mu.shape + (1,)),
            mu = np.tile(mu, np.shape(channel_edges[:-1]) + (1,) * mu.ndim).T,
            sigma = np.tile(self.sigma, np.shape(channel_edges[:-1]) + (1,) * mu.ndim).T
        ) / channel_widths).to(U.Jy, equivalencies=[MHI_Jy])
        return

    def half_width(self, source):
        if self.sigma_mode == 'thermal':
            return np.sqrt(C.k_B * source.T_g / C.m_p).to(U.km * U.s ** -1)
        else:
            return self.sigma_mode
