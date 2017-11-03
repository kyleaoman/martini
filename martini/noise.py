from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
import numpy as np

class _BaseNoise(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        return

    @abstractmethod
    def f_noise(self):
        pass

class GaussianNoise(_BaseNoise):
    
    def __init__(self, rms=1.0 * U.Jy):
        
        self.rms = rms
        
        super(GaussianNoise, self).__init__()

        return

    def f_noise(self):
        return lambda datacube: np.abs(np.random.normal(scale=self.rms.value, size=datacube._array.shape)) * self.rms.unit
