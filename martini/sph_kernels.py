from abc import ABCMeta, abstractmethod
import numpy as np

#arguments in each case should be:
# - squared distance
# - kernel size

class _BaseSPHKernel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def line_integral(self, dr2, h):
        pass

class WendlandC2(_BaseSPHKernel):
    
    def __init__(self):
        super(WendlandC2, self).__init__()
        return

    def line_integral(self, dr2, h):
        retval = np.zeros(h.shape)
        R2 =  dr2 / (h * h)
        retval[R2 == 0] = 2. / 3.
        use = np.logical_and(R2 < 1, R2 > 0)
        R2 = R2[use]
        A = np.sqrt(1 - R2)
        retval[use] = 5 * R2 * R2 * (.5 * R2 + 3) * np.log((1 + A) / np.sqrt(R2)) + A * (-27. / 2. * R2 * R2 - 14. / 3. * R2 + 2. / 3.)
        return retval / .2992 / np.power(h, 2) #.2992 is normalization s.t. kernel integral = 1 for particle mass = 1
