"""Provides classes for modelling the 21-cm spectral line emitted by a SPH particle."""

from types import EllipsisType
from typing import Type
import numpy as np
from scipy.special import erf
import astropy.units as U
from astropy import constants as C
from abc import ABCMeta, abstractmethod
from martini.datacube import DataCube
from martini.sources import SPHSource
from martini._util import NUMBA_AVAILABLE, numba_threads

if NUMBA_AVAILABLE:
    import math
    from numba import njit, prange


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
    :meth:`~martini.spectral_models._BaseSpectrum.get_spectral_function_extra_data` to
    make information that depends on the :class:`~martini.sources.sph_source.SPHSource`
    (or derived class) or :class:`~martini.datacube.DataCube` properties available
    internally. This is required because the source object is not accessible at class
    initialization.

    Parameters
    ----------
    spec_dtype : type, optional
        Data type of the arrays storing spectra of each particle, can be used to manage
        memory usage by adjusting precision.

    See Also
    --------
    martini.spectral_models.GaussianSpectrum
    martini.spectral_models.DiracDeltaSpectrum
    """

    spec_dtype: type
    _allow_numba: bool = True  # intended for switching off in tests

    def __init__(self, spec_dtype: type = np.float64) -> None:
        self.spec_dtype = spec_dtype
        return

    @U.quantity_input
    def _eval_spectra(
        self,
        source: SPHSource,
        datacube: DataCube,
        ncpu: int = 1,
        mask: slice = slice(None),
    ) -> U.Quantity[U.Jy]:
        """
        Evaluate the spectra of each particle, optionally a masked subset.

        The spectral model defined in
        :meth:`~martini.spectral_models._BaseSpectrum.spectral_function` is evaluated
        using the channel edges from the :class:`~martini.datacube.DataCube` instance and
        the particle velocities of the :class:`~martini.sources.sph_source.SPHSource` (or
        derived class) instance provided.

        If ``ncpu > 1`` and :mod:`numba` is installed then the calculation is
        multi-threaded (but this usually doesn't speed it up much, likely because the
        bottleneck is memory access).

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object containing arrays of particle properties.

        datacube : ~martini.datacube.DataCube
            :class:`~martini.datacube.DataCube` object defining the observational
            parameters, including spectral channels.

        ncpu : int
            Number of threads to use for evaluation of spectra. Can be set to ``-1`` to
            use as many threads as available cores. Currently speedup for multiple cores
            is limited, see full documentation for details.

        mask : slice
            Evaluate spectra only for a subset of particles, enables processing particles
            in batches.
        """
        channel_edges = datacube.velocity_channel_edges[mask]
        channel_widths = np.abs(np.diff(channel_edges.to(U.km * U.s**-1)))
        assert source.skycoords is not None
        vmids = source.skycoords.radial_velocity[mask]
        A = source.mHI_g[mask] * np.power(source.skycoords.distance.to(U.Mpc), -2)
        extra_data = self.get_spectral_function_extra_data(source, datacube, mask=mask)
        if all(np.diff(channel_edges) > 0):
            lower_edges_slice: slice = np.s_[:-1]
            upper_edges_slice: slice = np.s_[1:]
        elif all(np.diff(channel_edges) < 0):
            lower_edges_slice = np.s_[1:]
            upper_edges_slice = np.s_[:-1]
        else:
            raise ValueError("Channel edges are not monotonic sequence.")
        spectra = self.spectral_function(
            channel_edges[np.newaxis, lower_edges_slice].astype(self.spec_dtype),
            channel_edges[np.newaxis, upper_edges_slice].astype(self.spec_dtype),
            vmids[:, np.newaxis].astype(self.spec_dtype),
            extra_data=extra_data,
            ncpu=ncpu,
        )
        # ensure that spectra array is modified in place, keep memory usage minimal:
        spectra <<= U.dimensionless_unscaled
        np.multiply(A.astype(self.spec_dtype)[..., np.newaxis], spectra, out=spectra)
        np.divide(spectra, channel_widths.astype(self.spec_dtype), out=spectra)

        @U.quantity_input
        def MHI_to_Jy_inplace(x: U.Quantity[U.Msun / U.Mpc**2 / (U.km / U.s)]) -> None:
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

        MHI_to_Jy_inplace(spectra)

        return spectra

    @abstractmethod
    @U.quantity_input
    def half_width(self, source: SPHSource) -> U.Quantity[U.km / U.s]:
        """
        Abstract method; get the half-width of the spectrum, globally or per-particle.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source object will be provided to allow access to particle
            properties.
        """
        pass  # pragma: no cover

    @abstractmethod
    @U.quantity_input
    def spectral_function(
        self,
        a: U.Quantity[U.km / U.s],
        b: U.Quantity[U.km / U.s],
        vmids: U.Quantity[U.km / U.s],
        extra_data: dict | None = None,
        ncpu: int = 1,
    ) -> U.dimensionless_unscaled:
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

        extra_data : dict, optional
            ``dict`` containing arrays of extra data for the spectral function
            evaluation.

        ncpu : int
            Number of threads to use in evaluation.

        See Also
        --------
        martini.spectral_models._BaseSpectrum.get_spectral_function_extra_data
        """
        pass  # pragma: no cover

    def get_spectral_function_extra_data(
        self,
        source: SPHSource,
        datacube: DataCube,
        mask: slice | EllipsisType = np.s_[...],
        extra_data: dict[str, U.Quantity] | None = None,
    ) -> dict[str, U.Quantity]:
        """
        Initialize extra data needed by spectral function. Default is no extra data.

        Derived classes should override this function, if needed, to populate the dict
        with any information from the source that is required by the
        :meth:`~martini.spectral_models._BaseSpectrum.spectral_function`,
        then call ``super().get_spectral_function_extra_data`` with the ``extra_data``
        argument set to the dictionary that they loaded. This function then handles
        setting up the array for broadcasting.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object, making particle properties available.

        datacube : ~martini.datacube.DataCube
            :class:`~martini.datacube.DataCube` object defining the observational
            parameters, including spectral channels.

        mask : slice, optional
            Slice defining the subset of particles to operate on.

        extra_data : dict, optional
            ``dict`` containing additional data arrays needed for the spectral function
            evaluation. The dictionary entries should contain scalars or 1D arrays of
            particle properties.

        Returns
        -------
        dict
            The extra data that have been read in and prepared for use.

        See Also
        --------
        martini.spectral_models.GaussianSpectrum.get_spectral_function_extra_data
        """
        if extra_data is None:
            return {}
        return {
            k: (v[mask, np.newaxis] if not v.isscalar else v).astype(self.spec_dtype)
            for k, v in extra_data.items()
        }


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _gaussian_numba_kernel(
        a: np.ndarray,
        b: np.ndarray,
        vmids: np.ndarray,
        sigma: np.ndarray,
        spec_dtype: Type[np.number],
    ) -> np.ndarray:
        """
        Fast evaluation for the Gaussian spectral function.

        Parameters
        ----------
        a : np.ndarray
            Lower boundaries of the spectral channels, shape (1, N).

        b : np.ndarray
            Upper boundaries of the spectral channels, shape (1, N).

        vmids : np.ndarray
            Spectral centres of particles, shape (M, 1).

        spec_dtype : Type[np.number]
            The data type for output.

        Returns
        -------
        np.ndarray
            Evaluated spectra, shape (M, N).
        """
        M = vmids.shape[0]
        N = a.shape[1]
        spectrum = np.zeros((M, N), dtype=spec_dtype)
        sqrt_2 = math.sqrt(2.0)
        is_row_varying = sigma.ndim == 2 and sigma.shape[0] > 1
        s_global = 0.0 if is_row_varying else float(sigma.flat[0])
        for i in prange(M):  # type: ignore[attr-defined]
            v = vmids[i, 0]
            s_val = sigma[i, 0] if is_row_varying else s_global
            denom = sqrt_2 * s_val
            for j in range(N):
                val_b = (b[0, j] - v) / denom
                val_a = (a[0, j] - v) / denom
                spectrum[i, j] = (math.erf(val_b) - math.erf(val_a)) * 0.5
        return spectrum


class GaussianSpectrum(_BaseSpectrum):
    r"""
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
        :math:`m_p` is the particle mass.

    spec_dtype : type, optional
        Data type of the arrays storing spectra of each particle, can be used to manage
        memory usage by adjusting precision.

    See Also
    --------
    martini.spectral_models._BaseSpectrum
    martini.spectral_models.DiracDeltaSpectrum
    """

    def __init__(
        self,
        sigma: str | U.Quantity[U.km / U.s] = 7.0 * U.km * U.s**-1,
        spec_dtype: type = np.float64,
    ) -> None:
        self.sigma_mode = sigma
        super().__init__(spec_dtype=spec_dtype)

        return

    @U.quantity_input
    def spectral_function(
        self,
        a: U.Quantity[U.km / U.s],
        b: U.Quantity[U.km / U.s],
        vmids: U.Quantity[U.km / U.s],
        extra_data: dict[str, U.Quantity] | None = None,
        ncpu: int = 1,
    ) -> U.dimensionless_unscaled:
        """
        Evaluate a Gaussian integral in a channel.

        Requires sigma to be available from
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

        extra_data : dict, optional
            ``dict`` containing arrays of extra data for the spectral function
            evaluation.

        ncpu : int
            Number of threads to use in evaluation.

        Returns
        -------
        ~astropy.units.Quantity
            The evaluated spectral model (dimensionless).
        """
        assert extra_data is not None
        sigma = extra_data["sigma"]

        if (
            NUMBA_AVAILABLE
            and self._allow_numba
            and a.ndim == 2
            and b.ndim == 2
            and vmids.ndim == 2
        ):
            a_val = np.asarray(a.to_value(U.km / U.s))
            b_val = np.asarray(b.to_value(U.km / U.s))
            vmids_val = np.asarray(vmids.to_value(U.km / U.s))
            sigma_val = np.asarray(sigma.to_value(U.km / U.s))

            with numba_threads(ncpu):
                raw_spectrum = _gaussian_numba_kernel(
                    a_val, b_val, vmids_val, sigma_val, self.spec_dtype
                )
            return raw_spectrum * U.dimensionless_unscaled

        # work in-place as much as possible to limit memory usage:
        @U.quantity_input
        def term_in_place(
            x: U.Quantity[U.km / U.s],
            vmids: U.Quantity[U.km / U.s],
            sigma: U.Quantity[U.km / U.s],
        ) -> U.dimensionless_unscaled:
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
            ~astropy.units.Quantity
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

    def get_spectral_function_extra_data(
        self,
        source: SPHSource,
        datacube: DataCube,
        mask: slice | EllipsisType = np.s_[...],
        extra_data: dict[str, U.Quantity] | None = None,
    ) -> dict[str, U.Quantity]:
        """
        Expose particle velocity dispersions.

        Access to these is needed by
        :meth:`~martini.spectral_models.GaussianSpectrum.spectral_function`.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object.

        datacube : ~martini.datacube.DataCube
            :class:`~martini.datacube.DataCube` object.

        mask : slice, optional
            Slice defining the subset of particles to operate on.

        extra_data : dict, optional
            ``dict`` containing arrays of extra data for the spectral function
            evaluation.

        Returns
        -------
        dict
            The extra data that have been read in and prepared for use.
        """
        extra_data = {"sigma": self.half_width(source)}
        return super().get_spectral_function_extra_data(
            source, datacube, mask=mask, extra_data=extra_data
        )

    @U.quantity_input
    def half_width(self, source: SPHSource) -> U.Quantity[U.km / U.s]:
        """
        Get 1D velocity dispersions from particle temperatures, or return constant.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object, making particle properties available.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Velocity dispersion (constant, or per particle).
        """
        if self.sigma_mode == "thermal":
            # 3D velocity dispersion of an ideal gas is sqrt(3 * kB * T / mp)
            # So 1D velocity dispersion is sqrt(kB * T / mp)
            return np.sqrt(C.k_B * source.T_g / C.m_p).to(U.km * U.s**-1)
        else:
            return self.sigma_mode


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _diracdelta_numba_kernel(
        a: np.ndarray, b: np.ndarray, vmids: np.ndarray, spec_dtype: Type[np.number]
    ) -> np.ndarray:
        """
        Fast evaluation for the Dirac-delta spectral function.

        Parameters
        ----------
        a : np.ndarray
            Lower boundaries of the spectral channels, shape (1, N).

        b : np.ndarray
            Upper boundaries of the spectral channels, shape (1, N).

        vmids : np.ndarray
            Spectral centres of particles, shape (M, 1).

        spec_dtype : Type[np.number]
            The data type for output.

        Returns
        -------
        np.ndarray
            Evaluated spectra, shape (M, N).
        """
        M = vmids.shape[0]
        N = a.shape[1]
        spectra = np.zeros((M, N), dtype=spec_dtype)
        for i in prange(M):  # type: ignore[attr-defined]
            v = vmids[i, 0]
            for j in range(N):
                t1 = 1.0 if (v - a[0, j]) >= 0.0 else 0.0
                t2 = 1.0 if (b[0, j] - v) >= 0.0 else 0.0
                spectra[i, j] = t1 * t2
        return spectra


class DiracDeltaSpectrum(_BaseSpectrum):
    """
    Class implemeting a Dirac-delta model for the spectrum of the HI line.

    The line is modelled as a Dirac-delta function, centered at the particle
    velocity.

    Parameters
    ----------
    spec_dtype : type, optional
        Data type of the arrays storing spectra of each particle, can be used to manage
        memory usage by adjusting precision.
    """

    def __init__(self, spec_dtype: type = np.float64) -> None:
        super().__init__(spec_dtype=spec_dtype)
        return

    @U.quantity_input
    def spectral_function(
        self,
        a: U.Quantity[U.km / U.s],
        b: U.Quantity[U.km / U.s],
        vmids: U.Quantity[U.km / U.s],
        extra_data: dict[str, U.Quantity] | None = None,
        ncpu: int = 1,
    ) -> U.dimensionless_unscaled:
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

        extra_data : dict, optional
            ``dict`` containing arrays of extra data for the spectral function
            evaluation.

        ncpu : int
            Number of threads to use in evaluation.

        Returns
        -------
        ~astropy.units.Quantity
            The evaluated spectral model (dimensionless).
        """
        if (
            NUMBA_AVAILABLE
            and self._allow_numba
            and a.ndim == 2
            and b.ndim == 2
            and vmids.ndim == 2
        ):
            a_val = np.asarray(a.to_value(U.km / U.s))
            b_val = np.asarray(b.to_value(U.km / U.s))
            vmids_val = np.asarray(vmids.to_value(U.km / U.s))
            with numba_threads(ncpu):
                raw_spectrum = _diracdelta_numba_kernel(
                    a_val, b_val, vmids_val, self.spec_dtype
                )
            return raw_spectrum * U.dimensionless_unscaled

        @U.quantity_input
        def term_in_place(
            x1: U.Quantity[U.km / U.s], x2: U.Quantity[U.km / U.s]
        ) -> U.dimensionless_unscaled:
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
            ~astropy.units.Quantity
                :class:`~astropy.units.Quantity` (dimensionless).
            """
            term = x1 - x2  # individually small, broadcast 2D array here
            np.heaviside(term, 1.0, out=term)
            term <<= U.dimensionless_unscaled
            return term

        spectrum = term_in_place(vmids, a)
        np.multiply(spectrum, term_in_place(b, vmids), out=spectrum)
        return spectrum

    @U.quantity_input
    def half_width(self, source: SPHSource) -> U.Quantity[U.km / U.s]:
        """
        Dirac-delta function has 0 width.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            Source object, making particle properties available.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of velocity.
            Velocity dispersion of ``0 * U.km * U.s**-1``.
        """
        return 0 * U.km * U.s**-1
