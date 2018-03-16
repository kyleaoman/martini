from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U

class _BaseSPHKernel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def px_weight(self, dr2, h):
        pass

    @abstractmethod
    def validate(self, sm_lengths):
        pass

class WendlandC2Kernel(_BaseSPHKernel):
    
    def __init__(self):
        super().__init__()
        return

    def px_weight(self, dij, h):
        dr2 = np.power(dij, 2).sum(axis=0) 
        retval = np.zeros(h.shape)
        R2 =  dr2 / (h * h)
        retval[R2 == 0] = 2. / 3.
        use = np.logical_and(R2 < 1, R2 > 0)
        R2 = R2[use]
        A = np.sqrt(1 - R2)
        retval[use] = 5 * R2 * R2 * (.5 * R2 + 3) * np.log((1 + A) / np.sqrt(R2)) + A * (-27. / 2. * R2 * R2 - 14. / 3. * R2 + 2. / 3.)
        return retval / .2992 / np.power(h, 2) #.2992 is normalization s.t. kernel integral = 1 for particle mass = 1

    def validate(self, sm_lengths):
        if (sm_lengths < 2 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.WendlandC2Kernel.validate: SPH smoothing lengths "
                               "must be >= 2 px in size for WendlandC2 kernel integral approximation "
                               "accuracy.")
        return

class DiracDeltaKernel(_BaseSPHKernel):
    
    def __init__(self, ignore_smoothing=False):
        self.ignore_smoothing = ignore_smoothing
        super().__init__()
        return
        
    def px_weight(self, dij, h):
        return np.where((np.abs(dij) < 0.5 * U.pix).all(axis=0), np.ones(h.shape), np.zeros(h.shape))

    def validate(self, sm_lengths):
        if (sm_lengths > 1 * U.pix).any() and (self.ignore_smoothing == False):
            raise RuntimeError("Martini.sph_kernels.DiracDeltaKernel.validate: SPH smoothing lengths "
                               "must be <= 1 px in size for DiracDelta kernel to be a reasonable "
                               "approximation. Initialize with 'ignore_smoothing=True' to override.")
        return
