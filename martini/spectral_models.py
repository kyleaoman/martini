import numpy as np
import astropy.units as U
from astropy import constants as C
from scipy.special import erf
from abc import ABCMeta, abstractmethod


class _BaseSpectrum(metaclass=ABCMeta):
    """
    Abstract base class for implementions of spectral models to inherit from.

    Classes inheriting from _BaseSpectrum must implement two methods:
    `half_width` and `spectral_function`.

    `half_width` should define a characteristic width for the model, measured
    from the peak to the characteristic location. Note that particles whose
    spectra within +/- 4 half-widths of the peak do not intersect the DataCube
    bandpass will be discarded to speed computation.

    `spectral_function` should define the model spectrum. The spectrum should integrate to
    1, the amplitude is handled separately.

    They may also override the function `init_spectral_function_extra_data` to make
    information that depends on the martini.sources.SPHSource (or derived class)
    or martini.datacube.DataCube properties available internally. This is required because
    the source object is not accessible at class initialization.

    See Also
    --------
    GaussianSpectrum (simple example of a derived class using source properties
                      in spectral_function)
    DiracDeltaSpectrum
    """

    def __init__(self, spec_dtype=np.float64):
        self.spectral_function_extra_data = None
        self.spectra = None
        self.spec_dtype = spec_dtype
        return

    def init_spectra(self, source, datacube):
        """
        Pre-compute the spectrum of each particle.

        The spectral model defined in `spectral_function` is evaluated using
        the channel edges from the DataCube instance and the particle
        velocities of the SPHSource (or derived class) instance provided.
        Initializes additional particle properties by calling
        `init_spectral_function_extra_data` which then becomes accessible via
        `spectral_function_extra_data`.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object containing arrays of particle properties.

        datacube : martini.DataCube instance
            DataCube object defining the observational parameters, including
            spectral channels.
        """

        channel_edges = datacube.channel_edges
        channel_widths = np.diff(channel_edges).to(U.km * U.s**-1)
        vmids = source.skycoords.radial_velocity
        A = source.mHI_g * np.power(source.skycoords.distance.to(U.Mpc), -2)
        MHI_Jy = (
            U.Msun * U.Mpc**-2 * (U.km * U.s**-1) ** -1,
            U.Jy,
            lambda x: (1 / 2.36e5) * x,
            lambda x: 2.36e5 * x,
        )
        self.init_spectral_function_extra_data(source, datacube)
        channel_edges_unit = channel_edges.unit
        vmids_unit = vmids.unit
        raw_spectra = self.spectral_function(
            (
                np.tile(
                    channel_edges.to_value(channel_edges_unit)[:-1], vmids.shape + (1,)
                )
                * channel_edges.unit
            ).astype(self.spec_dtype),
            (
                np.tile(
                    channel_edges.to_value(channel_edges_unit)[1:], vmids.shape + (1,)
                )
                * channel_edges.unit
            ).astype(self.spec_dtype),
            (
                np.tile(
                    vmids.to_value(vmids_unit),
                    np.shape(channel_edges[:-1]) + (1,) * vmids.ndim,
                ).T
                * vmids.unit
            ).astype(self.spec_dtype),
        ).to_value(U.dimensionless_unscaled)
        self.spectra = (
            A.astype(self.spec_dtype)[..., np.newaxis]
            * raw_spectra
            / channel_widths.astype(self.spec_dtype)
        ).to(U.Jy, equivalencies=[MHI_Jy])

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
    def spectral_function(self, a, b, vmids):
        """
        Abstract method; implementation of the spectral model.

        Should calculate the flux in each spectral channel, calculation should
        be vectorized (numpy).

        Parameters
        ----------
        a : Quantity, with dimensions of velocity
            Lower spectral channel edge(s).

        b : Quantity, with dimensions of velocity
            Upper spectral channel edge(s).

        vmids : Quantity, with dimensions of velocity
            Particle velocities along the line of sight.

        See Also
        --------
        init_spectral_function_extra_data
        """

        pass

    def init_spectral_function_extra_data(self, source, datacube):
        """
        Initialize extra data needed by spectral function. Default is no extra data.

        Derived classes should override this function, if needed, to populate the dict
        with any information from the source that is required by the spectral_function,
        then call super().init_spectral_function_extra_data.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object, making particle properties available.

        See Also
        --------
        GaussianSpectrum.init_spectral_function_extra_data (for an example with extra
                                                            data)
        """
        if self.spectral_function_extra_data is None:
            self.spectral_function_extra_data = dict()
        self.spectral_function_extra_data = {
            k: np.tile(
                v,
                np.shape(datacube.channel_edges[:-1])
                + (1,) * source.skycoords.radial_velocity.ndim,
            )
            .astype(self.spec_dtype)
            .T
            for k, v in self.spectral_function_extra_data.items()
        }
        return


class GaussianSpectrum(_BaseSpectrum):
    """
    Class implementing a Gaussian model for the spectrum of the HI line.

    The line is modelled as a Gaussian of either fixed width, or of width
    scaling with the particle temperature as sqrt(k_B * T / m_p), centered
    at the particle velocity.

    Parameters
    ----------
    sigma : Quantity, with dimensions of velocity, or string {'thermal'}, \
    optional
        Width of the Gaussian modelling the line (constant for all particles),
        or specify 'thermal' for width equal to sqrt(k_B * T / m_p) where k_B
        is Boltzmann's constant, T is the particle temperature and m_p is the
        particle mass. (Default is 7 km/s.)

    See Also
    --------
    _BaseSpectrum
    DiracDeltaSpectrum
    """

    def __init__(self, sigma=7.0 * U.km * U.s**-1, spec_dtype=np.float64):
        self.sigma_mode = sigma
        super().__init__(spec_dtype=spec_dtype)

        return

    def spectral_function(self, a, b, vmids):
        """
        Evaluate a Gaussian integral in a channel. Requires sigma to be available
        from `spectral_function_extra_data`.

        Parameters
        ----------
        a : Quantity, with dimensions of velocity
            Lower spectral channel edge(s).

        b : Quantity, with dimensions of velocity
            Upper spectral channel edge(s).

        vmids : Quantity, with dimensions of velocity
            Particle velocities along the line of sight.

        Returns
        -------
        out : Quantity, dimensionless
            The evaluated spectral model.
        """

        assert self.spectral_function_extra_data is not None
        sigma = self.spectral_function_extra_data["sigma"]

        return 0.5 * (
            erf((b - vmids) / (np.sqrt(2.0) * sigma))
            - erf((a - vmids) / (np.sqrt(2.0) * sigma))
        )

    def init_spectral_function_extra_data(self, source, datacube):
        """
        Helper function to expose particle velocity dispersions to `spectral_function`.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object.

        datacube: martini.datacube.DataCube instance
            DataCube object.

        """

        self.spectral_function_extra_data = dict(sigma=self.half_width(source))
        super().init_spectral_function_extra_data(source, datacube)
        return

    def half_width(self, source):
        """
        Calculate 1D velocity dispersions from particle temperatures, or return
        constant.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object, making particle properties available.

        Returns
        -------
        out : Quantity, with dimensions of velocity
            Velocity dispersion (constant, or per particle).
        """

        if self.sigma_mode == "thermal":
            # 3D velocity dispersion of an ideal gas is sqrt(3 * kB * T / mp)
            # So 1D velocity dispersion is sqrt(kB * T / mp)
            return np.sqrt(C.k_B * source.T_g / C.m_p).to(U.km * U.s**-1)
        else:
            return self.sigma_mode


class DiracDeltaSpectrum(_BaseSpectrum):
    """
    Class implemeting a Dirac-delta model for the spectrum of the HI line.

    The line is modelled as a Dirac-delta function, centered at the particle
    velocity.
    """

    def __init__(self, spec_dtype=np.float64):
        super().__init__(spec_dtype=spec_dtype)
        return

    def spectral_function(self, a, b, vmids):
        """
        Evaluate a Dirac-delta function in a channel.

        Parameters
        ----------
        a : Quantity, with dimensions of velocity
            Lower spectral channel edge(s).

        b : Quantity, with dimensions of velocity
            Upper spectral channel edge(s).

        vmids : Quantity, with dimensions of velocity
            Particle velocities along the line of sight.

        Returns
        -------
        out : Quantity, dimesionless
            The evaluated spectral model.
        """

        return np.heaviside(vmids - a, 1.0) * np.heaviside(b - vmids, 0.0)

    def half_width(self, source):
        """
        Dirac-delta function has 0 width.

        Parameters
        ----------
        source : martini.sources.SPHSource (or derived class) instance
            Source object, making particle properties available.

        Returns
        -------
        out : Quantity
            Velocity dispersion of 0 km/s.
        """

        return 0 * U.km * U.s**-1
