from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U

class _BaseBeam(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.px_size = None
        self.kernel = None
        return

    def needs_pad(self):
        return self.kernel.shape[0] // 2

    def init_kernel(self, datacube):
        self.px_size = datacube.px_size
        px_centres = (np.arange(-self.kernel_size_px(), self.kernel_size_px() + 1)) * self.px_size
        self.kernel = self.f_kernel()(*np.meshgrid(px_centres, px_centres))[..., np.newaxis]
        return
        
    @abstractmethod
    def f_kernel(self):
        pass

    @abstractmethod
    def kernel_size_px(self):
        pass

class GaussianBeam(_BaseBeam):

    def __init__(self, bmaj=15.*U.arcsec, bmin=15.*U.arcsec, bpa=0.*U.deg, truncate=4.):

        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.truncate = truncate

        super(GaussianBeam, self).__init__()

        return

    def f_kernel(self):

        a = np.power(np.cos(self.bpa), 2) / (2. * np.power(self.bmin, 2)) \
            + np.power(np.sin(self.bpa), 2) / (2. * np.power(self.bmaj, 2))
        b = -np.sin(2. * self.bpa) / (4 * np.power(self.bmin, 2)) \
            + np.sin(2. * self.bpa) / (4 * np.power(self.bmaj, 2)) #signs set for CCW rotation (PA)
        c = np.power(np.sin(self.bpa), 2) / (2. * np.power(self.bmin, 2)) \
            + np.power(np.cos(self.bpa), 2) / (2. * np.power(self.bmaj, 2))
        A = np.power(2. * np.pi * self.bmin * self.bmaj, -1)

        return lambda x, y: A * np.exp(-a * np.power(x, 2) - 2. * b * x * y - c * np.power(y, 2))
        
    def kernel_size_px(self):
        return np.ceil(self.bmaj * self.truncate / self.px_size).value + 1
