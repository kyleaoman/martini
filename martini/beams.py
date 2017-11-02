import numpy as np
import astropy.units as U

class BaseBeam():
    def __init__(self):
        return

    def convolve(datacube):
        for spatial_slice in datacube.spatial_slices():
            pass

class GaussianBeam(BaseBeam):

    def __init__(self, bmaj=15.*U.arcsec, bmin=15.*U.arcsec, bpa=0.*U.deg, truncate=4.):

        super(GaussianBeam, self).__init__()
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa

        self.kernel = None

        return
