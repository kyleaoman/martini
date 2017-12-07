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
        
    def init_spectra(self, source, datacube):
        channel_edges = datacube.channel_edges
        channel_widths = np.diff(channel_edges).to(U.km * U.s ** -1)
        vmids = source.sky_coordinates.radial_velocity
        A = source.mHI_g * np.power(source.sky_coordinates.distance.to(U.Mpc), -2)
        MHI_Jy = ( 
            U.solMass * U.Mpc ** -2 * (U.km * U.s ** -1) ** -1, 
            U.Jy, 
            lambda x: (1 / 2.36E5) * x, 
            lambda x: 2.36E5 * x
        )
        # note that the unit is U.Jy * U.pix ** -2 for consistency with other places in the code,
        # i.e. the per pixel is measured in units of pixel areas, where 1 * U.pix is the pixel side
        # length
        spectral_function_kwargs = {k: np.tile(v, np.shape(channel_edges[:-1]) + (1,) * vmids.ndim).T
                                    for k, v in self.spectral_function_kwargs(source).items()}
        self.spectra = (A[..., np.newaxis] * self.spectral_function(
            np.tile(channel_edges[:-1], vmids.shape + (1,)),
            np.tile(channel_edges[1:], vmids.shape + (1,)),
            np.tile(vmids, np.shape(channel_edges[:-1]) + (1,) * vmids.ndim).T,
            **spectral_function_kwargs
        ) / channel_widths).to(U.Jy, equivalencies=[MHI_Jy])

    @abstractmethod
    def half_width(self, source):
        pass

    @abstractmethod
    def spectral_function(self, a, b, vmids, **kwargs):
        pass

    @abstractmethod
    def spectral_function_kwargs(self, source):
        pass

class GaussianSpectrum(_BaseSpectrum):
    
    def __init__(self, sigma=7. * U.km * U.s ** -1):
        self.sigma_mode = sigma
        super(GaussianSpectrum, self).__init__()
        return

    def spectral_function(self, a, b, vmids, sigma=1.0):
        return .5 * (
            erf((b - vmids) / (np.sqrt(2.) * sigma)) - \
            erf((a - vmids) / (np.sqrt(2.) * sigma))
        )

    def spectral_function_kwargs(self, source):
        return {'sigma': self.half_width(source)}

    def half_width(self, source):
        if self.sigma_mode == 'thermal':
            return np.sqrt(C.k_B * source.T_g / C.m_p).to(U.km * U.s ** -1)
        else:
            return self.sigma_mode

class DiracDeltaSpectrum(_BaseSpectrum):

    def __init__(self):
        super(DiracDeltaSpectrum, self).__init__()
        return

    def spectral_function(self, a, b, vmids):
        return np.heaviside(vmids - a, 1.) * np.heaviside(b - vmids, 0.)

    def spectral_function_kwargs(self, source):
        return dict()

    def half_width(self, source):
        return 0 * U.km * U.s ** -1
