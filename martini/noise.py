"""Provides classes to generate noise for data cubes."""

from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from numpy.random._generator import Generator
from martini.beams import _BaseBeam
from martini.datacube import DataCube


class _BaseNoise(object):
    """
    Abstract base class to inherit for classes implementing a noise model.

    Classes inheriting from :class:`~martini.noise._BaseNoise` must implement a
    method :meth:`~martini.noise._BaseNoise.generate` which receives two arguments,
    a :class:`~martini.datacube.DataCube` instance and a beam instance. This method
    should return an :class:`~astropy.units.Quantity` array with the same shape as the
    datacube array, with units of flux density per beam or per unit area. Returning
    the current units of the data cube (``datacube.current_units``) avoids a unit
    conversion of the noise cube when it is added to the data cube.

    Parameters
    ----------
    seed : int, optional
        The integer to seed the random number generator.

    See Also
    --------
    martini.noise.GaussianNoise
    """

    __metaclass__ = ABCMeta
    seed: int
    rng: Generator

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        return

    @abstractmethod
    def generate(self, datacube: DataCube, beam: _BaseBeam) -> U.Quantity:
        """
        Abstract method; create a cube containing noise.

        Any random number generation should use the :attr:`~martini.noise._BaseNoise.rng`
        generator.

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
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of flux density per beam
            or per unit area. Noise realization with size matching the
            :attr:`~martini.datacube.DataCube._array`.
        """
        pass  # pragma: no cover

    def reset_rng(self) -> None:
        """Reset the random number generator to its initial state."""
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
        beam.

    seed : int, optional
        Seed for random number generator.
    """

    target_rms: U.Quantity[U.Jy * U.beam**-1]

    @U.quantity_input
    def __init__(
        self,
        rms: U.Quantity[U.Jy * U.beam**-1] = 1.0 * U.Jy * U.beam**-1,
        seed: int = 0,
    ) -> None:
        self.target_rms = rms

        super().__init__(seed=seed)

        return

    def generate(self, datacube: DataCube, beam: _BaseBeam) -> U.Quantity:
        """
        Create a cube containing Gaussian noise.

        The noise is generated in the current units of the data cube (so that it can be
        added to the data cube without converting units).

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
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, in the current units of the data cube.
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
        # Convert the rms (rather than the noise cube) to the units of the data cube:
        beam_area_equivalencies = U.beam_angular_area(beam.area)
        rms = rms.to(U.Jy * U.arcsec**-2, equivalencies=beam_area_equivalencies).to(
            datacube.current_units,
            equivalencies=beam_area_equivalencies + [datacube.arcsec2_to_pix],
        )
        noise = self.rng.standard_normal(
            size=datacube._array.shape, dtype=datacube.cube_dtype
        )
        noise *= datacube.cube_dtype(rms.value)  # in place to avoid a copy
        return U.Quantity(noise, rms.unit, copy=False)
