from ._integrals import Gaussian_integral, WendlandC2_line_integral

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
        #DO CONVOLUTION!
        return
