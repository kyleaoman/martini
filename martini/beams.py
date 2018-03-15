from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from astropy.io import fits
from astropy import wcs
from scipy.interpolate import RectBivariateSpline
import warnings

f_HI = 1.420405751*U.GHz

class _BaseBeam(object):
    __metaclass__ = ABCMeta

    def __init__(self, bmaj=15.*U.arcsec, bmin=15.*U.arcsec, bpa=0.*U.deg):
        #some beams need information from the datacube; in this make their call to 
        #_BaseBeam.__init__ with bmaj == bmin == bpa == None and define a
        #init_beam_header, to be called after the ra, dec, vel, etc. of the datacube
        #are known
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.px_size = None
        self.kernel = None

        return

    def needs_pad(self):
        return self.kernel.shape[0] // 2, self.kernel.shape[1] // 2

    def init_kernel(self, datacube):
        self.px_size = datacube.px_size
        self.vel = datacube.velocity_centre
        self.ra = datacube.ra
        self.dec = datacube.dec
        if (self.bmaj == None) or (self.bmin == None) or (self.bpa == None):
            self.init_beam_header()
        npx_x, npx_y = self.kernel_size_px()
        px_centres_x = (np.arange(-npx_x, npx_x + 1)) * self.px_size
        px_centres_y = (np.arange(-npx_y, npx_y + 1)) * self.px_size
        self.kernel = self.f_kernel()(*np.meshgrid(px_centres_x, px_centres_y))[..., np.newaxis, np.newaxis]

        self.arcsec_to_beam = (
            U.Jy * U.arcsec ** -2,
            U.Jy * U.beam ** -1,
            lambda x: x * (np.pi * self.bmaj * self.bmin),
            lambda x: x / (np.pi * self.bmaj * self.bmin)
        )

        #can turn 2D beam into a 3D beam here; use above for central channel then shift in frequency up and down for other channels
        #then probably need to adjust convolution step to do the 2D convolution on a stack
        
        return
        
    @abstractmethod
    def f_kernel(self):
        pass

    @abstractmethod
    def kernel_size_px(self):
        pass
        
    @abstractmethod
    def init_beam_header(self):
        pass

class GaussianBeam(_BaseBeam):

    def __init__(self, bmaj=15.*U.arcsec, bmin=15.*U.arcsec, bpa=0.*U.deg, truncate=4.):
        self.truncate = truncate
        super(GaussianBeam, self).__init__(bmaj=bmaj, bmin=bmin, bpa=bpa)
        return

    def f_kernel(self):
        fwhm_to_sigma = lambda fwhm: fwhm / (2. * np.sqrt(2. * np.log(2.)))
        sigmamaj = fwhm_to_sigma(self.bmaj)
        sigmamin = fwhm_to_sigma(self.bmin)

        a = np.power(np.cos(self.bpa), 2) / (2. * np.power(sigmamin, 2)) \
            + np.power(np.sin(self.bpa), 2) / (2. * np.power(sigmamaj, 2))
        b = -np.sin(2. * self.bpa) / (4 * np.power(sigmamin, 2)) \
            + np.sin(2. * self.bpa) / (4 * np.power(sigmamaj, 2)) #signs set for CCW rotation (PA)
        c = np.power(np.sin(self.bpa), 2) / (2. * np.power(sigmamin, 2)) \
            + np.power(np.cos(self.bpa), 2) / (2. * np.power(sigmamaj, 2))
        A = np.power(2. * np.pi * sigmamin * sigmamaj, -1)

        return lambda x, y: A * np.exp(-a * np.power(x, 2) - 2. * b * x * y - c * np.power(y, 2))
        
    def kernel_size_px(self):
        size = np.ceil((self.bmaj * self.truncate).to(U.pix, U.pixel_scale(self.px_size / U.pix))).value + 1
        return size, size

class WSRTBeam(_BaseBeam):
    
    beamfile = '/Users/users/koman/Data/beam00_freq02.fits'

    def __init__(self):
        super(WSRTBeam, self).__init__(bmaj=None, bmin=None, bpa=None)

    def _load_beamfile(self):
        with fits.open(self.beamfile) as f:
            return f[0].header, np.transpose(f[0].data, axes=(1, 2, 0)) * U.Jy / U.beam

    def _centroid(self):
        bheader, bdata = self._load_beamfile()
        c = wcs.WCS(header=bheader)
        RAgrid, Decgrid, freqgrid = c.wcs_pix2world(
            *np.meshgrid(*tuple([np.arange(N) for N in bdata.shape])), 
            0
        )
        RAgrid *= U.deg
        Decgrid *= U.deg
        freqgrid *= U.Hz
        return [A[tuple(np.array(bdata.shape) // 2)] for A in (RAgrid, Decgrid, freqgrid)]

    def init_beam_header(self):
        self.bmaj = 15. * U.arcsec / np.sin(self.dec)
        self.bmin = 15. * U.arcsec
        self.bpa = 90. * U.deg
        return

    def f_kernel(self):
        freq = self.vel.to(U.GHz, equivalencies=U.doppler_radio(f_HI))
        bheader, bdata = self._load_beamfile()
        centroid = self._centroid()
        bpx_ra = np.abs(bheader['CDELT1'] * U.deg).to(U.arcsec) 
        bpx_dec = np.abs(bheader['CDELT2'] * U.deg).to(U.arcsec)
        dRAs = np.arange(-(bdata.shape[0] // 2), bdata.shape[0] // 2 + 1) * bpx_ra * (centroid[2] / freq).to(U.dimensionless_unscaled)
        dDecs = np.arange(-(bdata.shape[1] // 2), bdata.shape[1] // 2 + 1) * bpx_dec * np.sin(self.dec) / np.sin(centroid[1]) * (centroid[2] / freq).to(U.dimensionless_unscaled)
        interpolator = RectBivariateSpline(dRAs, dDecs, bdata[..., 0], kx=1, ky=1, s=0)
        return lambda x, y: interpolator(y, x, grid=False) #RectBivariateSpline is a wrapper around Fortran code and causes a transpose...

    def kernel_size_px(self):
        if self.px_size > 12. * U.arcsec:
            warnings.warn("Using WSRT beam with datacube pixel size >> 8 arcsec, beam interpolation may fail.")
        freq = self.vel.to(U.GHz, equivalencies=U.doppler_radio(f_HI))
        bheader, bdata = self._load_beamfile()
        centroid = self._centroid()
        aspect_x, aspect_y = np.floor(bdata.shape[0] // 2 * np.sin(self.dec)), bdata.shape[1] // 2
        aspect_x *= np.abs((bheader['CDELT1'] * U.deg)).to(U.arcsec) * (centroid[2] / freq).to(U.dimensionless_unscaled)
        aspect_y *= (bheader['CDELT2'] * U.deg).to(U.arcsec) * (centroid[2] / freq).to(U.dimensionless_unscaled)
        return tuple([(a.to(U.pix, U.pixel_scale(self.px_size / U.pix))).value + 1 for a in (aspect_x, aspect_y)])
