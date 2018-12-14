from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U


class _BaseNoise(object):
    """
    Abstract base class to inherit for classes implementing a noise model.

    Classes inheriting from _BaseNoise must implement a method 'generate'
    which receives one argument, a martini.DataCube instance. This method
    should return an astropy.units.Quantity array with the same shape as the
    datacube array, and units of Jy (or compatible).

    See Also
    --------
    GaussianNoise (simple example implementing a Gaussian noise model)
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def generate(self, datacube):
        """
        Abstract method; create a cube containing noise.
        """
        pass


class GaussianNoise(_BaseNoise):
    """
    Implementation of a simple Gaussian noise model.

    Provides a 'generate' method producing a cube of Gaussian Noise.

    Parameters
    ----------
    rms : astropy.units.Quantity with dimensions of flux density (e.g. Jy)
        Root mean square amplitude of the noise field.

    Returns
    -------
    out : GaussianNoise
        An appropriately initialized GaussianNoise object.
    """

    def __init__(self, rms=1.0 * U.Jy):

        self.rms = rms

        super().__init__()

        return

    def generate(self, datacube):
        """
        Create a cube containing Gaussian noise.

        Some numpy functions such as np.random.normal strip units, so need to
        handle them explicitly.

        Parameters
        ----------
        datacube : martini.DataCube instance
            This method will be called passing the martini.DataCube instance as
            an argument; its attributes can thus be accessed here.
            datacube._array.shape is particularly relevant.

        Returns
        -------
        out : astropy.units.Quantity array, with dimensions of flux density
            Noise realization with size matching the DataCube._array.
        """

        return np.random.normal(
            scale=self.rms.value,
            size=datacube._array.shape
        ) * self.rms.unit
