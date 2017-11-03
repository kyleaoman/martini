from ._integrals import Gaussian_integral, WendlandC2_line_integral
from scipy.ndimage import convolve

class Martini():

    def __init__(self, source=None, datacube=None, beam=None, baselines=None, noise=None):
        self.source = source
        self.datacube = datacube
        self.beam = beam
        self.baselines = baselines
        self.noise = noise

        self.beam.init_kernel(self.datacube)
        self.datacube.add_pad(self.beam.needs_pad())
        
        return

    def convolve_beam(self):
        unit = self.datacube._array.unit
        self.datacube._array = convolve(
            self.datacube._array, 
            self.beam.kernel,
            mode='constant',
            cval=0.0
        ) * unit
        self.datacube.drop_pad()
        return
