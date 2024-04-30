from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U


class _BaseNoise(object):
    """
    Abstract base class to inherit for classes implementing a noise model.

    Classes inheriting from :class:`~martini.noise._BaseNoise` must implement a
    method :meth:`~martini.noise._BaseNoise.generate` which receives one argument,
    a :class:`~martini.datacube.DataCube` instance. This method
    should return an :class:`~astropy.units.Quantity` array with the same shape as the
    datacube array, and units of Jy (or compatible).

    See Also
    --------
    ~martini.noise.GaussianNoise
    """

    __metaclass__ = ABCMeta

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        return

    @abstractmethod
    def generate(self, datacube, beam):
        """
        Abstract method; create a cube containing noise.

        Any random number generation should use the :attr:`~martini.noise._BaseNoise.rng`
        generator.
        """
        pass

    def reset_rng(self):
        """
        Reset the random number generator to its initial state.

        If the seed is ``None`` (the default value), this has no effect.
        """
        self.rng = np.random.default_rng(seed=self.seed)
        return


class GaussianNoise(_BaseNoise):
    """
    Implementation of a simple Gaussian noise model.

    Provides a :meth:`~martini.noise.GaussianNoise.generate` method producing a cube of
    Gaussian noise.

    Parameters
    ----------
    rms : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of flux density per beam.
        Desired root mean square amplitude of the noise field after convolution with the
        beam. (Default: ``1.0 * U.Jy * U.beam ** -1``)

    seed : int, optional
        Seed for random number generator. If ``None``, results will be unpredictable,
        if an integer is given results will be repeatable. (Default: ``None``)
    """

    def __init__(
        self,
        rms=1.0 * U.Jy * U.beam**-1,
        seed=None,
    ):
        self.target_rms = rms

        super().__init__(seed=seed)

        return

    def generate(self, datacube, beam):
        """
        Create a cube containing Gaussian noise.

        Some numpy functions such as :func:`numpy.random.normal` strip units, so need to
        handle them explicitly.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            This method will be called passing the :class:`~martini.datacube.DataCube`
            instance as an argument; its attributes can thus be accessed here.
            ``datacube._array.shape`` is particularly relevant.

        beam : ~martini.beams._BaseBeam
            This method will be called passing the object derived from
            :class:`~martini.beams._BaseBeam` (for example a
            :class:`~martini.beams.GaussianBeam`) as an argument. Its attributes can thus
            be accessed here. The beam size is needed to estimate the pre-convolution rms
            required to obtain the desired post-convolution rms.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of flux density.
            Noise realization with size matching the
            :attr:`~martini.datacube.DataCube._array`.
        """

        sig_maj = (beam.bmaj / 2 / np.sqrt(2 * np.log(2)) / datacube.px_size).to_value(
            U.dimensionless_unscaled
        )
        sig_min = (beam.bmin / 2 / np.sqrt(2 * np.log(2)) / datacube.px_size).to_value(
            U.dimensionless_unscaled
        )
        # Approximate rms reduction by convolution:
        # new_rms * 2 * sqrt(pi) * beam_std ~ old_rms for a circular beam
        # for an elliptical beam beam_std = sqrt(beam_std_maj * beam_std_min)
        # Approximation turns out to be low by ~ 10%, correct for this:
        rms = self.target_rms * 2.19568 * np.sqrt(np.pi * sig_maj * sig_min)
        rms_unit = rms.unit
        return (
            self.rng.normal(scale=rms.to_value(rms_unit), size=datacube._array.shape)
            * rms_unit
        )
