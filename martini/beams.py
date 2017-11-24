from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U

class _BaseBeam(object):
    __metaclass__ = ABCMeta

    def __init__(self, bmaj=15.*U.arcsec, bmin=15.*U.arcsec, bpa=0.*U.deg):
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.px_size = None
        self.kernel = None

        return

    def needs_pad(self):
        return self.kernel.shape[0] // 2

    def init_kernel(self, datacube):
        self.px_size = datacube.px_size
        px_centres = (np.arange(-self.kernel_size_px(), self.kernel_size_px() + 1)) * self.px_size
        self.kernel = self.f_kernel()(*np.meshgrid(px_centres, px_centres))[..., np.newaxis, np.newaxis]
        self.arcsec_to_beam = (
            U.Jy * U.arcsec ** -2,
            U.Jy * U.beam ** -1,
            lambda x: x * (np.pi * self.bmaj * self.bmin),
            lambda x: x / (np.pi * self.bmaj * self.bmin)
        )
        return
        
    @abstractmethod
    def f_kernel(self):
        pass

    @abstractmethod
    def kernel_size_px(self):
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
        return np.ceil((self.bmaj * self.truncate).to(U.pix, U.pixel_scale(self.px_size / U.pix))).value + 1
