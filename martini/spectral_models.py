import numpy as np
import astropy.units as U
from scipy.special import erf as _erf
erf = lambda z: _erf(z.to(U.dimensionless_unscaled).value)

def _Gaussian_integral(a, b, mu=1.0, sigma=1.0):
    return .5 * (erf((b - mu) / (np.sqrt(2.) * sigma)) - erf((a - mu) / (np.sqrt(2.) * sigma)))

def GaussianSpectrum(datacube, source, mask, weights):
    mu = source.sky_coordinates.radial_velocity[mask]
    A = source.mHI_g[mask] * weights
    sigma = 7. * U.km * U.s ** -1
    bins = datacube.channel_edges
    spectrum_no_sum = A[..., np.newaxis] * \
                      _Gaussian_integral(
                          np.tile(bins[:-1], mu.shape + (1,)), 
                          np.tile(bins[1:], mu.shape + (1,)), 
                          mu=np.tile(mu, np.shape(bins[:-1]) + (1,) * mu.ndim).T, 
                          sigma=np.tile(sigma, np.shape(bins[:-1]) + (1,) * mu.ndim).T
                      )
    if mu.ndim > 0:
        return np.sum(spectrum_no_sum, axis=-2)
    else:
        return spectrum_no_sum
