"""
Provides classes for modelling the 21-cm spectral line emitted by a SPH particle.
"""

import numpy as np
import astropy.units as U
from astropy import constants as C
from scipy.special import erf
from abc import ABCMeta, abstractmethod


class _BaseSpectrum(metaclass=ABCMeta):
    """
    Abstract base class for implementions of spectral models to inherit from.

    Classes inheriting from :class:`~martini.spectral_models._BaseSpectrum` must implement
    two methods: :meth:`~martini.spectral_models._BaseSpectrum.half_width` and
    :meth:`~martini.spectral_models._BaseSpectrum.spectral_function`.

    :meth:`~martini.spectral_models._Base_spectrum.half_width` should define a
    characteristic width for the model, measured from the peak to the characteristic
    location. Note that particles whose spectra within +/- 4 half-widths of the peak do
    not intersect the data cube bandpass will be discarded to speed computation.

    :meth:`~martini.spectral_models._BaseSpectrum.spectral_function` should define the
    model spectrum. The spectrum should integrate to 1, the amplitude is handled
    separately.

    They may also override the method
    :meth:`~martini.spectral_models._BaseSpectrum.init_spectral_function_extra_data` to
    make information that depends on the :class:`~martini.sources.sph_source.SPHSource`
    (or derived class) or :class:`~martini.datacube.DataCube` properties available
    internally. This is required because the source object is not accessible at class
    initialization.

    Parameters
    ----------
    ncpu : int, optional
        Number of cpus to use for evaluation of particle spectra. Defaults to ``1`` if not
        provided. (Default: ``None``)

    spec_dtype : type, optional
        Data type of the arrays storing spectra of each particle, can be used to manage
        memory usage by adjusting precision.

    See Also
    --------
    martini.spectral_models.GaussianSpectrum
    martini.spectral_models.DiracDeltaSpectrum
    """

    def __init__(self, ncpu=None, spec_dtype=np.float64):
        self.ncpu = ncpu if ncpu is not None else 1
        self.spectral_function_extra_data = None
        self.spectra = None
        self.spec_dtype = spec_dtype
        return

    def init_spectra(self, source, datacube):
        """
        Pre-compute the spectrum of each particle.

        The spectral model defined in
        :meth:`~martini.spectral_models._BaseSpectrum.spectral_function` is evaluated
        using the channel edges from the :class:`~martini.datacube.DataCube` instance and
        the particle velocities of the :class:`~martini.sources.sph_source.SPHSource` (or
        derived class) instance provided.

        If the instance of this class was initialized with ``ncpu > 1`` then a
        process pool is created to distribute subsets of the calculation in
        parallel. To minimize overhead form serializing large amounts of
        data in :mod:`multiprocess` communications, each parallel process inherits the
        entire line-of-sight velocity array (cheap because of copy-on-write
        behaviour), then masks its copy to the subset to operate on.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object containing arrays of particle properties.

        datacube : ~martini.datacube.DataCube
            :class:`~martini.datacube.DataCube` object defining the observational
            parameters, including spectral channels.
        """

        self.channel_edges = datacube.velocity_channel_edges
        channel_widths = np.abs(np.diff(self.channel_edges).to(U.km * U.s**-1))
        self.vmids = source.skycoords.radial_velocity
        A = source.mHI_g * np.power(source.skycoords.distance.to(U.Mpc), -2)

        if self.ncpu == 1:
            self.spectra = self.evaluate_spectra(source, datacube)
        else:
            from multiprocess.pool import ThreadPool

            with ThreadPool(processes=self.ncpu) as pool:
                self.spectra = np.vstack(
                    pool.map(
                        lambda mask: self.evaluate_spectra(source, datacube, mask=mask),
                        [
                            (
                                np.s_[
                                    icpu
                                    * len(self.vmids)
                                    // self.ncpu : (icpu + 1)
                                    * len(self.vmids)
                                    // self.ncpu
                                ]
                                if icpu is not None
                                else np.s_[...]
                            )
                            for icpu in range(self.ncpu)
                        ],
                    )
                )
        # ensure that self.spectra array is modified in place, keep memory usage minimal:
        self.spectra <<= U.dimensionless_unscaled
        np.multiply(
            A.astype(self.spec_dtype)[..., np.newaxis], self.spectra, out=self.spectra
        )
        np.divide(
            self.spectra, channel_widths.astype(self.spec_dtype), out=self.spectra
        )

        def MHI_to_Jy_inplace(x):
            """
            Apply the HI mass to flux density conversion, with no memory overhead.

            The conversion is:
            M_HI/Msun = 2.36x10^5 * (D/Mpc)^2 * (S_21/Jy km s^-1)

            Parameters
            ----------
            x : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity`, with dimensions of
                mass / length^2 / velocity.
            """
            input_units = U.Msun * U.Mpc**-2 * (U.km * U.s**-1) ** -1
            np.divide(x, 2.36e5, out=x)
            x *= U.Jy / input_units
            return

        MHI_to_Jy_inplace(self.spectra)

        return

    def evaluate_spectra(self, source, datacube, mask=np.s_[...]):
        """
        The main portion of the calculation of the spectra.

        Separated into this function so that it can be called by a parallel
        process pool. Initializes additional particle properties by calling
        :meth:`~martini.spectral_models._BaseSpectrum.init_spectral_function_extra_data`
        which then becomes accessible via
        :attr:`~martini.spectral_models._BaseSpectrum.spectral_function_extra_data`.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object containing arrays of particle properties.

        datacube : ~martini.datacube.DataCube
            :class:`~martini.datacube.DataCube` object defining the observational
            parameters, including spectral channels.

        mask : slice, optional
            Slice defining the subset of particles to operate on.
            (Default: ``np.s_[...]``)
        """
        vmids = self.vmids[mask]
        self.init_spectral_function_extra_data(source, datacube, mask=mask)
        if all(np.diff(self.channel_edges) > 0):
            lower_edges_slice = np.s_[:-1]
            upper_edges_slice = np.s_[1:]
        elif all(np.diff(self.channel_edges) < 0):
            lower_edges_slice = np.s_[1:]
            upper_edges_slice = np.s_[:-1]
        else:
            raise ValueError("Channel edges are not monotonic sequence.")
        return (
            self.spectral_function(
                self.channel_edges[np.newaxis, lower_edges_slice].astype(
                    self.spec_dtype
                ),
                self.channel_edges[np.newaxis, upper_edges_slice].astype(
                    self.spec_dtype
                ),
                vmids[:, np.newaxis].astype(self.spec_dtype),
            )
            << U.dimensionless_unscaled
        )

    @abstractmethod
    def half_width(self, source):
        """
        Abstract method; calculate the half-width of the spectrum, either globally or
        per-particle.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source object will be provided to allow access to particle
            properties.
        """
        pass

    @abstractmethod
    def spectral_function(self, a, b, vmids):
        """
        Abstract method; implementation of the spectral model.

        Should calculate the flux in each spectral channel, calculation should
        be vectorized (with :mod:`numpy`).

        Parameters
        ----------
        a : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Lower spectral channel edge(s).

        b : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Upper spectral channel edge(s).

        vmids : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Particle velocities along the line of sight.

        See Also
        --------
        martini.spectral_models._BaseSpectrum.init_spectral_function_extra_data
        """

        pass

    def init_spectral_function_extra_data(self, source, datacube, mask=np.s_[...]):
        """
        Initialize extra data needed by spectral function. Default is no extra data.

        Derived classes should override this function, if needed, to populate the dict
        with any information from the source that is required by the
        :meth:`~martini.spectral_models._BaseSpectrum.spectral_function`,
        then call ``super().init_spectral_function_extra_data``.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object, making particle properties available.

        datacube : ~martini.datacube.DataCube
            :class:`~martini.datacube.DataCube` object defining the observational
            parameters, including spectral channels.

        mask : slice, optional
            Slice defining the subset of particles to operate on.
            (Default: ``np.s_[...]``)

        See Also
        --------
        martini.spectral_models.GaussianSpectrum.init_spectral_function_extra_data
        """
        if self.spectral_function_extra_data is None:
            self.spectral_function_extra_data = dict()
        self.spectral_function_extra_data = {
            k: (v[mask, np.newaxis] if not v.isscalar else v).astype(self.spec_dtype)
            for k, v in self.spectral_function_extra_data.items()
        }
        return


class GaussianSpectrum(_BaseSpectrum):
    """
    Class implementing a Gaussian model for the spectrum of the HI line.

    The line is modelled as a Gaussian of either fixed width, or of width
    scaling with the particle temperature as :math:`\\sqrt{k_B T / m_p}`, centered
    at the particle velocity.

    Parameters
    ----------
    sigma : ~astropy.units.Quantity or str, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity, or string
        ``"thermal"``.
        Width of the Gaussian modelling the line (constant for all particles),
        or specify ``"thermal"`` for width equal to :math:`\\sqrt{k_B T / m_p}` where
        :math:`k_B` is Boltzmann's constant, :math:`T` is the particle temperature and
        :math:`m_p` is the particle mass. (Default: ``7 U.km * U.s**-1``)

    ncpu : int, optional
        Number of cpus to use for evaluation of particle spectra. Defaults to ``1`` if not
        provided. (Default: ``None``)

    spec_dtype : type, optional
        Data type of the arrays storing spectra of each particle, can be used to manage
        memory usage by adjusting precision.

    See Also
    --------
    martini.spectral_models._BaseSpectrum
    martini.spectral_models.DiracDeltaSpectrum
    """

    def __init__(self, sigma=7.0 * U.km * U.s**-1, ncpu=None, spec_dtype=np.float64):
        self.sigma_mode = sigma
        super().__init__(ncpu=ncpu, spec_dtype=spec_dtype)

        return

    def spectral_function(self, a, b, vmids):
        """
        Evaluate a Gaussian integral in a channel. Requires sigma to be available from
        :attr:`~martini.spectral_models.GaussianSpectrum.spectral_function_extra_data`.

        Parameters
        ----------
        a : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Lower spectral channel edge(s).

        b : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Upper spectral channel edge(s).

        vmids : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Particle velocities along the line of sight.

        Returns
        -------
        out : ~astropy.units.Quantity
            The evaluated spectral model (dimensionless).
        """

        assert self.spectral_function_extra_data is not None
        sigma = self.spectral_function_extra_data["sigma"]

        # work in-place as much as possible to limit memory usage:
        def term_in_place(x, vmids, sigma):
            """
            Evaluate partial expression for spectrum, working in-place in memory.

            Parameters
            ----------
            x : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity`, with dimensions of velocity.

            vmids : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity`, with dimensions of velocity.

            sigma : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity`, with dimensions of velocity.

            Returns
            -------
            out : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity` (dimensionless).
            """
            term = x - vmids  # individually small, broadcast 2D array here
            np.divide(term, np.sqrt(self.spec_dtype(2.0)), out=term)
            np.divide(term, sigma, out=term)
            term <<= U.dimensionless_unscaled
            erf(term, out=term)
            return term

        spectrum = term_in_place(b, vmids, sigma)
        np.subtract(
            spectrum,
            term_in_place(a, vmids, sigma),
            out=spectrum,
        )
        np.multiply(self.spec_dtype(0.5), spectrum, out=spectrum)
        return spectrum

    def init_spectral_function_extra_data(self, source, datacube, mask=np.s_[...]):
        """
        Helper function to expose particle velocity dispersions to
        :meth:`~martini.spectral_models.GaussianSpectrum.spectral_function`.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object.

        datacube : ~martini.datacube.DataCube
            :class:`~martini.datacube.DataCube` object.

        mask : slice, optional
            Slice defining the subset of particles to operate on.
            (Default: ``np.s_[...]``)
        """

        self.spectral_function_extra_data = dict(sigma=self.half_width(source))
        super().init_spectral_function_extra_data(source, datacube, mask=mask)
        return

    def half_width(self, source):
        """
        Calculate 1D velocity dispersions from particle temperatures, or return
        constant.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object, making particle properties available.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
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

    Parameters
    ----------
    ncpu : int, optional
        Number of cpus to use for evaluation of particle spectra. Defaults to ``1`` if not
        provided. (Default: ``None``)

    spec_dtype : type, optional
        Data type of the arrays storing spectra of each particle, can be used to manage
        memory usage by adjusting precision.
    """

    def __init__(self, ncpu=None, spec_dtype=np.float64):
        super().__init__(ncpu=ncpu, spec_dtype=spec_dtype)
        return

    def spectral_function(self, a, b, vmids):
        """
        Evaluate a Dirac-delta function in a channel.

        Parameters
        ----------
        a : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Lower spectral channel edge(s).

        b : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Upper spectral channel edge(s).

        vmids : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Particle velocities along the line of sight.

        Returns
        -------
        out : ~astropy.units.Quantity
            The evaluated spectral model (dimensionless).
        """

        def term_in_place(x1, x2):
            """
            Evaluate partial expression for spectrum, working in-place in memory.

            Parameters
            ----------
            x1 : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity`, with dimensions of velocity.

            x2 : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity`, with dimensions of velocity.

            Returns
            -------
            out : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity` (dimensionless).
            """
            term = x1 - x2  # individually small, broadcast 2D array here
            np.heaviside(term, 1.0, out=term)
            term <<= U.dimensionless_unscaled
            return term

        spectrum = term_in_place(vmids, a)
        np.multiply(spectrum, term_in_place(b, vmids), out=spectrum)
        return spectrum

    def half_width(self, source):
        """
        Dirac-delta function has 0 width.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object, making particle properties available.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Velocity dispersion of ``0 * U.km * U.s**-1``.
        """

        return 0 * U.km * U.s**-1
