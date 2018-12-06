import numpy as np
import astropy.units as U
from astropy import constants as C
from scipy.special import erf as _erf
from abc import ABCMeta, abstractmethod


def erf(z):
    return _erf(z.to(U.dimensionless_unscaled).value)


class _BaseSpectrum(object):
    """
    Abstract base class for implementions of spectral models to inherit from.

    Classes inheriting from _BaseSpectrum must implement three methods:
    'half_width', 'spectral_function' and 'spectral_function_kwargs'.

    'half_width' should define a characteristic width for the model, measured
    from the peak to the characteristic location. Note that particles whose
    spectra within +/- 4 half-widths of the peak do not intersect the DataCube
    bandpass will be discarded to speed computation.

    'spectral_function' should define the model spectrum. Arguments which
    depend on the martini.sources.SPHSource (or derived class) properties
    should make use of the 'spectral_function_kwargs' method. The spectrum
    should integrate to 1, the amplitude is handled separately.

    'spectral_function_kwargs' should provide a helper to pass properties from
    martini.source.SPHSource (or derived class) to the 'spectral_function'.
    This is required because the source object is not accessible at class
    initialization.

    See Also
    --------
    GaussianSpectrum (simple example of a derived class using source properties
                      in spectral_function)
    DiracDeltaSpectrum
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.spectra = None
        return

    def init_spectra(self, source, datacube):
        """
        Pre-compute the spectrum of each particle.

        The spectral model defined in 'spectral_function' is evaluated using
        the channel edges from the DataCube instance and the particle
        velocities of the SPHSource (or derived class) instance provided.
        Additional particle properties can be accessed via the
        'spectral_function_kwargs' helper method.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object containing arrays of particle properties.

        datacube : martini.DataCube instance
            DataCube object defining the observational parameters, including
            spectral channels.
        """

        channel_edges = datacube.channel_edges
        channel_widths = np.diff(channel_edges).to(U.km * U.s ** -1)
        vmids = source.sky_coordinates.radial_velocity
        A = source.mHI_g * \
            np.power(source.sky_coordinates.distance.to(U.Mpc), -2)
        MHI_Jy = (
            U.solMass * U.Mpc ** -2 * (U.km * U.s ** -1) ** -1,
            U.Jy,
            lambda x: (1 / 2.36E5) * x,
            lambda x: 2.36E5 * x
        )
        spectral_function_kwargs = \
            {k: np.tile(v, np.shape(channel_edges[:-1]) + (1,) * vmids.ndim).T
             for k, v in self.spectral_function_kwargs(source).items()}
        raw_spectra = self.spectral_function(
            np.tile(
                channel_edges.value[:-1],
                vmids.shape + (1,)
            ) * channel_edges.unit,
            np.tile(
                channel_edges.value[1:],
                vmids.shape + (1,)
            ) * channel_edges.unit,
            np.tile(
                vmids.value,
                np.shape(channel_edges[:-1]) + (1,) * vmids.ndim
            ).T * vmids.unit,
            **spectral_function_kwargs
        ).to(U.dimensionless_unscaled).value
        self.spectra = (A[..., np.newaxis] * raw_spectra / channel_widths)\
            .to(U.Jy, equivalencies=[MHI_Jy])

        return

    @abstractmethod
    def half_width(self, source):
        """
        Abstract method; calculate the half-width of the spectrum, either
        globally or per-particle.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            The source object will be provided to allow access to particle
            properties.
        """
        pass

    @abstractmethod
    def spectral_function(self, a, b, vmids, **kwargs):
        """
        Abstract method; implementation of the spectral model.

        Should calculate the flux in each spectral channel, calculation should
        be vectorized (numpy).

        Parameters
        ----------
        a : astropy.units.Quantity array, with dimensions of velocity
            Lower spectral channel edge(s).

        b : astropy.units.Quantity array, with dimensions of velocity
            Upper spectral channel edge(s).

        vmids : astropy.units.Quantity array, with dimensions of velocity
            Particle velocities along the line of sight.

        **kwargs : various
            See spectral_function_kwargs.

        See Also
        --------
        spectral_function_kwargs
        """

        pass

    @abstractmethod
    def spectral_function_kwargs(self, source):
        """
        Abstract method; helper method to pass additional arguments to the
        spectral_function.

        Should return a dict containing the kwarg names as keys with the values
        to pass as associated values.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            The source object will be provided so that its attributes can be
            accessed.

        See Also
        --------
        GaussianSpectrum (example implementation)
        """

        pass


class GaussianSpectrum(_BaseSpectrum):
    """
    Class implementing a Gaussian model for the spectrum of the HI line.

    The line is modelled as a Gaussian of either fixed width, or of width
    scaling with the particle temperature as sqrt(k_B * T / m_p), centered
    at the particle velocity.

    Parameters
    ----------
    sigma : astropy.units.Quantity, with dimensions of velocity, or string:
            'thermal'
        Width of the Gaussian modelling the line (constant for all particles),
        or specify 'thermal' for width equal to sqrt(k_B * T / m_p) where k_B
        is Boltzmann's constant, T is the particle temperature and m_p is the
        particle mass.

    Returns
    -------
    out : GaussianSpectrum
        An appropriately initialized GaussianSpectrum object.

    See Also
    --------
    _BaseSpectrum
    DiracDeltaSpectrum
    """

    def __init__(self, sigma=7. * U.km * U.s ** -1):

        self.sigma_mode = sigma
        super().__init__()

        return

    def spectral_function(self, a, b, vmids, sigma=1.0 * U.km * U.s ** -1):
        """
        Evaluate a Gaussian integral in a channel.

        Parameters
        ----------
        a : astropy.units.Quantity, with dimensions of velocity
            Lower spectral channel edge(s).

        b : astropy.units.Quantity, with dimensions of velocity
            Upper spectral channel edge(s).

        vmids : astropy.units.Quantity array, with dimensions of velocity
            Particle velocities along the line of sight.

        sigma : astropy.units.Quantity array, or astropy.units.Quantity, with
                dimensions of velocity
            Velocity dispersion for HI line width, either for each particle or
            constant.

        Returns
        -------
        out : astropy.units.Quantity, dimesionless
            The evaluated spectral model.
        """

        # erf strips units
        return .5 * (
            erf((b - vmids) / (np.sqrt(2.) * sigma)) -
            erf((a - vmids) / (np.sqrt(2.) * sigma))
        ) * U.dimensionless_unscaled

    def spectral_function_kwargs(self, source):
        """
        Helper function to pass particle velocity dispersions to the
        spectral_function.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object.

        Returns
        -------
        out : dict
            Keyword arguments for the spectral_function.
        """

        return {'sigma': self.half_width(source)}

    def half_width(self, source):
        """
        Calculate velocity dispersions from particle temperatures, or return
        constant.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object, making particle properties available.

        Returns
        -------
        out : astropy.unit.Quantity, with dimensions of velocity
            Velocity dispersion (constant, or per particle).
        """

        if self.sigma_mode == 'thermal':
            return np.sqrt(C.k_B * source.T_g / C.m_p).to(U.km * U.s ** -1)
        else:
            return self.sigma_mode


class DiracDeltaSpectrum(_BaseSpectrum):
    """
    Class implemeting a Dirac-delta model for the spectrum of the HI line.

    The line is modelled as a Dirac-delta function, centered at the particle
    velocity.

    Returns
    -------
    out : DiracDeltaSpectrum
        An appropriately initialized DiracDeltaSpectrum object.
    """

    def __init__(self):
        super().__init__()
        return

    def spectral_function(self, a, b, vmids):
        """
        Evaluate a Dirac-delta function in a channel.

        Parameters
        ----------
        a : astropy.units.Quantity, with dimensions of velocity
            Lower spectral channel edge(s).

        b : astropy.units.Quantity, with dimensions of velocity
            Upper spectral channel edge(s).

        vmids : astropy.units.Quantity array, with dimensions of velocity
            Particle velocities along the line of sight.

        Returns
        -------
        out : astropy.units.Quantity, dimesionless
            The evaluated spectral model.
        """

        return np.heaviside(vmids - a, 1.) * np.heaviside(b - vmids, 0.)

    def spectral_function_kwargs(self, source):
        """
        No additional kwargs.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object, making particle properties available.

        Returns
        -------
        out : dict
            Empty; no additional kwargs.
        """

        return dict()

    def half_width(self, source):
        """
        Dirac-delta function has 0 width.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object, making particle properties available.

        Returns
        -------
        out : astropy.units.Quantity
            Velocity dispersion of 0 km/s.
        """

        return 0 * U.km * U.s ** -1
