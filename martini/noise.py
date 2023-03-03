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

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        return

    @abstractmethod
    def generate(self, datacube):
        """
        Abstract method; create a cube containing noise.

        Any random number generation should use the self.rng generator.
        """
        pass

    def reset_rng(self):
        """
        Reset the random number generator to its initial state.

        If the seed is None (the default value), this has no effect.
        """
        self.rng = np.random.default_rng(seed=self.seed)
        return


class GaussianNoise(_BaseNoise):
    """
    Implementation of a simple Gaussian noise model.

    Provides a `generate` method producing a cube of Gaussian noise.

    Parameters
    ----------
    rms : Quantity, with dimensions of flux density per solid angle
        Root mean square amplitude of the noise field.

    seed : int or None
        Seed for random number generator. If None, results will be unpredictable,
        if an integer is given results will be repeatable. (Default: None)
    """

    def __init__(
        self,
        rms=1.0 * U.Jy * U.arcsec**-2,
        seed=None,
    ):
        self.rms = rms

        super().__init__(seed=seed)

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
        out : Quantity, with dimensions of flux density
            Noise realization with size matching the DataCube._array.
        """
        rms_unit = self.rms.unit
        return (
            self.rng.normal(
                scale=self.rms.to_value(rms_unit), size=datacube._array.shape
            )
            * rms_unit
        )
