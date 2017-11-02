from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U

class BaseBeam(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.px_size = None
        self.kernel = None
        return

    def needs_pad(self):
        return self.kernel.shape[0] // 2

    def init_kernel(self, datacube):
        self.px_size = datacube.px_size
        self.kernel = self.calculate_kernel()

    @abstractmethod
    def calculate_kernel(self):
        pass

class GaussianBeam(BaseBeam):

    def __init__(self, bmaj=15.*U.arcsec, bmin=15.*U.arcsec, bpa=0.*U.deg, truncate=4.):

        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.truncate = truncate

        super(GaussianBeam, self).__init__()

        return

    def calculate_kernel(self):
        k = np.zeros((3, 3))
        k[1, 1] = 1
        return k #PLACEHOLDER!
