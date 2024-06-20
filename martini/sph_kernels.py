from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from scipy.special import erf
from scipy.optimize import fsolve


def find_fwhm(f):
    """
    Solve for the FWHM of the input function.

    Intended for SPH kernels and therefore assumes maximum occurs at :math:`f(0)` and
    symmetry around :math:`f(0)`.

    Parameters
    ----------
    f : callable
        Function for which to find the FWHM.

    Returns
    -------
    out : float
        FWHM of the input function.
    """
    return 2 * fsolve(lambda q: f(q) - f(np.zeros(1)) / 2, 0.5)[0]


class _BaseSPHKernel(object):
    """
    Abstract base class for classes implementing SPH kernels to inherit from.

    Classes inheriting from :class:`~martini.sph_kernels._BaseSPHKernel` must implement
    three methods: :meth:`~martini.sph_kernels._BaseSPHKernel.kernel`,
    :meth:`martini.sph_kernels._BaseSPHKernel._kernel_integral` and
    :meth:`~martini.sph_kernels._BaseSPHKernel_validate`.

    :meth:`~martini.sph_kernels._BaseSPHKernel.kernel` should define the kernel function,
    normalized such that its volume integral is ``1``.

    :meth:`~martini.sph_kernels._BaseSPHKernel._kernel_integral` should define the
    integral of the kernel over a pixel given the distance between the pixel centre and
    the particle centre, and the smoothing length (both in units of pixels). The integral
    should be normalized so that evaluated over the entire kernel it is equal to ``1``.

    :meth:`~martini.sph_kernels._BaseSPHKernel._validate` should check whether any
    approximations converge to sufficient accuracy (for instance, depending on the
    ratio of the pixel size and smoothing length), and raise an error if not. It should
    return a boolean array with ``True`` for particles which pass validation, and
    ``False`` otherwise.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self._rescale = 1
        return

    def _px_weight(self, dij, mask=None):
        """
        Calculate kernel integral using scaled smoothing lengths.

        This is the method that should be called by other modules in
        :mod:`martini`, rather than
        :meth:`~martini.sph_kernels._BaseSPHKernel._kernel_integral`.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels^-2.
            Integral of smoothing kernel over pixel, per unit pixel area.
        """
        if mask is not None:
            try:
                rescale = self._rescale[mask]
            except (TypeError, IndexError):
                rescale = self._rescale
            rescaled_h = self.sm_lengths[mask] * rescale
        else:
            rescaled_h = self.sm_lengths * self._rescale
        return self._kernel_integral(dij, rescaled_h, mask=mask)

    def _confirm_validation(self, noraise=False, quiet=False):
        """
        Verify kernel accuracy using scaled smoothing lengths.

        This is the method that should be called by other modules in
        :mod:`martini`, rather than :meth:`~martini.sph_kernels._BaseSPHKernel._validate`.

        Parameters
        ----------
        noraise : bool
            If ``True``, don't raise error if validation fails. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """

        return self._validate(self.sm_lengths, noraise=noraise, quiet=quiet)

    def _validate_error(self, msg, sm_lengths, valid, noraise=False, quiet=False):
        """
        Function managing handling of kernel validation errors.

        Parameters
        ----------
        msg : str
            Error message.

        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths in pixel units.

        valid : ~numpy.typing.ArrayLike
            Boolean array specifying which particles fail validation.

        noraise : bool
            If ``True``, don't raise error if validation fails. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """
        if not quiet:
            print(f"    ---------{self.__class__.__name__} VALIDATION---------")
            print("    Median smoothing length: ", np.median(sm_lengths), "px")
            print("    Minimum smoothing length: ", np.min(sm_lengths), "px")
            print("    Maximum smoothing length: ", np.max(sm_lengths), "px")
            print(
                "    Smoothing length histogram (np.histogram):",
                np.histogram(sm_lengths),
            )
            print(
                "    ",
                np.sum(np.logical_not(valid)),
                "/",
                sm_lengths.size,
                "smoothing lengths fail validation.",
            )
            print(
                "    -----------------------------" + "-" * len(self.__class__.__name__)
            )
        if not noraise:
            raise RuntimeError(msg)
        return

    def eval_kernel(self, r, h):
        """
        Evaluate the kernel, handling array casting and rescaling.

        The kernel parameter is defined:

        .. math::

            q = r / h

        Parameters
        ----------
        r : ~numpy.typing.ArrayLike or ~astropy.units.Quantity
            Distance parameter, same units (if applicable) as ``h``.
        h : ~numpy.typing.ArrayLike or ~astropy.units.Quantity
            Smoothing scale parameter (FWHM), same units (if applicable) as ``r``.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at position(s) ``r / h``.
        """

        q = np.array(r / h / self._rescale)
        if isinstance(q, U.Quantity):
            q = q.to_value(U.dimensionless_unscaled)
        scalar_input = q.ndim == 0
        W = self.kernel(q)
        W /= np.power(self._rescale, 3)
        if scalar_input:
            return W.item()
        else:
            return W

    def _apply_mask(self, mask):
        """
        Apply a mask to maskable attributes.

        Parameters
        ----------
        mask : ~numpy.typing.ArrayLike
            Boolean mask to apply to any maskable attributes.
        """
        self.sm_lengths = self.sm_lengths[mask]
        self.sm_ranges = self.sm_ranges[mask]

        return

    def _init_sm_lengths(self, source=None, datacube=None):
        """
        Determine kernel sizes in pixel units.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source providing the kernel sizes.

        datacube : martini.datacube.DataCube
            The datacube providing the pixel scale.
        """
        self.sm_lengths = np.arctan(
            source.hsm_g
            / source.skycoords.transform_to(datacube.coordinate_frame).distance
        ).to(U.pix, U.pixel_scale(datacube.px_size / U.pix))

        return

    def _init_sm_ranges(self):
        """
        Determine maximum number of pixels reached by kernel.
        """
        # store as float: use case for np.inf in GlobalProfile:
        self.sm_ranges = np.ceil(self.sm_lengths * self.size_in_fwhm)

        return

    @abstractmethod
    def kernel(self, q):
        """
        Abstract method; evaluate the kernel.

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """
        pass

    @abstractmethod
    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Abstract method; calculate the kernel integral over a pixel.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in pixels.
        mask : ~numpy.typing.ArrayLike or slice
            Boolean array, or slice. If the kernel has other internal properties to
            mask, it may use this.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels^-2.
            Integral of smoothing kernel over pixel, per unit pixel area.
        """

        pass

    @abstractmethod
    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Abstract method; check conditions for validity of kernel integral
        calculation.

        Some approximations may only converge if the ratio of the pixel size
        and the smoothing length is sufficiently large, or sufficiently small.
        This method should check these conditions and raise errors when
        appropriate. The smoothing lengths are provided normalized to the pixel
        size. :class:`~martini.sph_kernels._AdaptiveKernel` needs to force errors
        not to raise, other classes should just define ``**kwargs`` in the signature.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths, in units of pixels.

        noraise : bool
            If ``True``, suppress exceptions. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """

        pass


class _WendlandC2Kernel(_BaseSPHKernel):
    """
    Implementation of the Wendland C2 kernel integral.

    The Wendland C2 kernel is used in the EAGLE code and derivatives (not in
    Gadget/Gadget2!). The exact integral is usually too slow to be practical;
    the implementation here approximates the kernel amplitude as constant
    across the pixel, which converges to within 1% of the exact integral
    provided the SPH smoothing lengths are at least 1.51 pixels in size.

    The WendlandC2 kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        \\frac{21}{2\\pi}(1-q)^4(4q+1)
        &{\\rm for}\\;0 \\leq q < 1\\\\
        0 &{\\rm for}\\;q \\geq 1
        \\end{cases}

    """

    min_valid_size = 1.51

    def __init__(self):
        super().__init__()
        _unscaled_fwhm = find_fwhm(lambda r: self.eval_kernel(r, 1))
        self.size_in_fwhm = 1 / _unscaled_fwhm
        self._rescale /= _unscaled_fwhm

        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The WendlandC2 kernel is here defined as:

        .. math::
            W(q) = \\begin{cases}
            \\frac{21}{2\\pi}(1-q)^4(4q+1)
            &{\\rm for}\\;0 \\leq q < 1\\\\
            0 &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """

        W = np.where(q < 1, np.power(1 - q, 4) * (4 * q + 1), np.zeros(q.shape))
        W *= 21 / 2 / np.pi
        return W

    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Calculate the kernel integral over a pixel.

        The formula used approximates the kernel amplitude as constant
        across the pixel area and converges to the true value within 1%
        for smoothing lengths >= 1.51 pixels.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in pixels.
        mask : ~numpy.typing.ArrayLike or slice
            Boolean array, or slice. If the kernel has other internal properties to mask,
            it may use this.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Approximate kernel integral over the pixel area.
        """

        dr2 = np.power(dij, 2).sum(axis=0)
        retval = np.zeros(h.shape)
        R2 = dr2 / (h * h)
        retval[R2 == 0] = 2.0 / 3.0
        use = np.logical_and(R2 < 1, R2 != 0)
        R2 = R2[use]
        A = np.sqrt(1 - R2)
        retval[use] = 5 * R2 * R2 * (0.5 * R2 + 3) * np.log(
            (1 + A) / np.sqrt(R2)
        ) + A * (-27.0 / 2.0 * R2 * R2 - 14.0 / 3.0 * R2 + 2.0 / 3.0)
        norm = 21 / 2 / np.pi
        return retval * norm / np.power(h, 2)

    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in units of pixels.

        noraise : bool
            If ``True``, suppress kernel validation errors. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """
        valid = sm_lengths >= self.min_valid_size * U.pix
        if np.logical_not(valid).any():
            self._validate_error(
                "martini.sph_kernels._WendlandC2Kernel._validate:\n"
                f"SPH smoothing lengths must be >= {self.min_valid_size:f} px in "
                "size for WendlandC2 kernel integral "
                "approximation accuracy within 1%.\nThis check "
                "may be disabled by calling "
                "martini.martini.Martini.insert_source_in_cube with "
                "'skip_validation=True', but use this with "
                "care.\n",
                sm_lengths,
                valid,
                noraise=noraise,
                quiet=quiet,
            )

        return valid


class _WendlandC6Kernel(_BaseSPHKernel):
    """
    Implementation of the Wendland C6 kernel integral.

    The Wendland C6 kernel is used in the Magneticum code (not in
    Gadget/Gadget2!). The exact integral is usually too slow to be practical;
    the implementation here approximates the kernel amplitude as constant
    across the pixel, which converges to within 1% of the exact integral
    provided the SPH smoothing lengths are at least 1.29 pixels in size.

    The WendlandC6 kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        \\frac{1365}{64 \\pi} (1 - q)^8 (1 + 8q + 25q^2 + 32q^3)
        &{\\rm for}\\;0 \\leq q < 1\\\\
        0 &{\\rm for}\\;q \\geq 1
        \\end{cases}

    """

    min_valid_size = 1.29

    def __init__(self):
        super().__init__()
        _unscaled_fwhm = find_fwhm(lambda r: self.eval_kernel(r, 1))
        self.size_in_fwhm = 1 / _unscaled_fwhm
        self._rescale /= _unscaled_fwhm
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The WendlandC6 kernel is here defined as:

        .. math::
            W(q) = \\begin{cases}
            \\frac{1365}{64 \\pi} (1 - q)^8 (1 + 8q + 25q^2 + 32q^3)
            &{\\rm for}\\;0 \\leq q < 1\\\\
            0 &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """

        W = np.where(
            q < 1,
            np.power(1 - q, 8)
            * (1 + 8 * q + 25 * np.power(q, 2) + 32 * np.power(q, 3)),
            np.zeros(q.shape),
        )
        W *= 1365 / 64 / np.pi
        return W

    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Calculate the kernel integral over a pixel.

        The formula used approximates the kernel amplitude as constant
        across the pixel area and converges to the true value within 1%
        for smoothing lengths >= 1.29 pixels.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths, in pixels.
        mask : ~numpy.typing.ArrayLike or slice
            Boolean array, or slice. If the kernel has other internal properties to mask,
            it may use this.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Approximate kernel integral over the pixel area.
        """

        def indef(R, z):
            return (
                -231 * np.power(R, 10) * z
                - 385 * np.power(R, 8) * np.power(z, 3)
                - 1155 * np.power(R, 8) * z
                - 462 * np.power(R, 6) * np.power(z, 5)
                - 1540 * np.power(R, 6) * np.power(z, 3)
                - 462 * np.power(R, 6) * z
                - 330 * np.power(R, 4) * np.power(z, 7)
                - 1386 * np.power(R, 4) * np.power(z, 5)
                - 462 * np.power(R, 4) * np.power(z, 3)
                + 66 * np.power(R, 4) * z
                - (128 + 1 / 3) * np.power(R, 2) * np.power(z, 9)
                - 660 * np.power(R, 2) * np.power(z, 7)
                - 277.2 * np.power(R, 2) * np.power(z, 5)
                + 44 * np.power(R, 2) * np.power(z, 3)
                + 8 / 3 * np.power(z, 11) * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + (16 + 4 / 15)
                * np.power(R, 2)
                * np.power(z, 9)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 70.4 * np.power(z, 9) * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 360.8
                * np.power(R, 2)
                * np.power(z, 7)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 132 * np.power(z, 7) * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 550
                * np.power(R, 2)
                * np.power(z, 5)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                - 11 * np.power(R, 2) * z
                + 7.21875
                * np.power(R, 12)
                * (np.log(np.sqrt(np.power(R, 2) + np.power(z, 2)) + z))
                + 24.7813
                * np.power(R, 10)
                * z
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 173.25
                * np.power(R, 10)
                * (np.log(np.sqrt(np.power(R, 2) + np.power(z, 2)) + z))
                + 530.75
                * np.power(R, 8)
                * z
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 288.75
                * np.power(R, 8)
                * (np.log(np.sqrt(np.power(R, 2) + np.power(z, 2)) + z))
                + 47.4792
                * np.power(R, 8)
                * np.power(z, 3)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 767.25
                * np.power(R, 6)
                * z
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 58.0167
                * np.power(R, 6)
                * np.power(z, 5)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 819.5
                * np.power(R, 6)
                * np.power(z, 3)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 41.7
                * np.power(R, 4)
                * np.power(z, 7)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 752.4
                * np.power(R, 4)
                * np.power(z, 5)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                + 896.5
                * np.power(R, 4)
                * np.power(z, 3)
                * (np.sqrt(np.power(R, 2) + np.power(z, 2)))
                - 21 * np.power(z, 11)
                - (128 + 1 / 3) * np.power(z, 9)
                - 66 * np.power(z, 7)
                + 13.2 * np.power(z, 5)
                - 11 / 3 * np.power(z, 3)
                + z
            )

        dr2 = np.power(dij, 2).sum(axis=0)
        retval = np.zeros(h.shape)
        R = np.sqrt(dr2) / h
        use = np.logical_and(R < 1, R != 0)
        norm = 1365 / 64 / np.pi
        zmax = np.sqrt(1 - np.power(R[use], 2))
        retval[use] = norm * 2 * (indef(R[use], zmax) - indef(R[use], 0))
        retval[R == 0] = norm * 2 * (4 / 15)
        retval = retval / np.power(h, 2)
        return retval

    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths, in units of pixels.

        noraise : bool
            If ``True``, suppress kernel validation errors. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """

        valid = sm_lengths >= self.min_valid_size * U.pix
        if np.logical_not(valid).any():
            self._validate_error(
                "martini.sph_kernels._WendlandC6Kernel._validate:\n"
                f"SPH smoothing lengths must be >= {self.min_valid_size:f} px in "
                "size for WendlandC6 kernel integral "
                "approximation accuracy within 1%.\nThis check "
                "may be disabled by calling "
                "martini.martini.Martini.insert_source_in_cube with "
                "'skip_validation=True', but use this with "
                "care.",
                sm_lengths,
                valid,
                noraise=noraise,
                quiet=quiet,
            )
        return valid


class _CubicSplineKernel(_BaseSPHKernel):
    """
    Implementation of the cubic spline (M4) kernel integral.

    The cubic spline is the 'classic' SPH kernel. The exact integral is usually
    too slow to be practical; the implementation here approximates the kernel
    amplitude as constant across the pixel, which converges to within 1% of
    the exact integral provided the SPH smoothing lengths are at least 1.16
    pixels in size.

    The cubic spline kernel is here defined as:

    .. math ::
        W(q) = \\frac{8}{\\pi}\\begin{cases}
        (1 - 6q^2(1 - q))
        &{\\rm for}\\;0 \\leq q < \\frac{1}{2}\\\\
        2(1 - q)^3
        &{\\rm for}\\;\\frac{1}{2} \\leq q < 1\\\\
        0
        &{\\rm for}\\;q \\geq 1
        \\end{cases}

    """

    min_valid_size = 1.16

    def __init__(self):
        super().__init__()
        _unscaled_fwhm = find_fwhm(lambda r: self.eval_kernel(r, 1))
        self.size_in_fwhm = 1 / _unscaled_fwhm
        self._rescale /= _unscaled_fwhm
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The cubic spline kernel is here defined as:

        .. math ::
            W(q) = \\frac{8}{\\pi}\\begin{cases}
            (1 - 6q^2(1 - q))
            &{\\rm for}\\;0 \\leq q < \\frac{1}{2}\\\\
            2(1 - q)^3
            &{\\rm for}\\;\\frac{1}{2} \\leq q < 1\\\\
            0
            &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """

        W = np.where(
            q < 0.5, 1 - 6 * np.power(q, 2) + 6 * np.power(q, 3), 2 * np.power(1 - q, 3)
        )
        W[q > 1] = 0
        W *= 8 / np.pi
        return W

    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Calculate the kernel integral over a pixel.

        The formula used approximates the kernel amplitude as constant across
        the pixel area and converges to the true value within 1% for smoothing
        lengths >= 1.16 pixels.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels
            Particle smoothing lengths, in pixels.
        mask : ~numpy.typing.ArrayLike or slice
            Boolean array, or slice. If the kernel has other internal properties to mask,
            it may use this.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Approximate kernel integral over the pixel area.
        """

        dij *= 2  # changes interval from [0, 2) to [0, 1)
        dr2 = np.power(dij, 2).sum(axis=0)
        retval = np.zeros(h.shape)
        R2 = dr2 / (h * h)
        retval[R2 == 0] = 11.0 / 16.0 + 0.25 * 0.25
        case1 = np.logical_and(R2 > 0, R2 <= 1)
        case2 = np.logical_and(R2 > 1, R2 <= 4)

        R2_1 = R2[case1]
        R2_2 = R2[case2]
        A_1 = np.sqrt(1 - R2_1)
        B_1 = np.sqrt(4 - R2_1)
        B_2 = np.sqrt(4 - R2_2)
        I1 = (
            A_1
            - 0.5 * np.power(A_1, 3)
            - 1.5 * R2_1 * A_1
            + 3.0 / 32.0 * A_1 * (3 * R2_1 + 2)
            + 9.0 / 32.0 * R2_1 * R2_1 * (np.log(1 + A_1) - np.log(np.sqrt(R2_1)))
        )
        I2 = (
            -B_2 * (3 * R2_2 + 56) / 4.0
            - 3.0 / 8.0 * R2_2 * (R2_2 + 16) * np.log((2 + B_2) / np.sqrt(R2_2))
            + 2 * (3 * R2_2 + 4) * B_2
            + 2 * np.power(B_2, 3)
        )
        I3 = (
            -B_1 * (3 * R2_1 + 56) / 4.0
            + A_1 * (4 * R2_1 + 50) / 8.0
            - 3.0 / 8.0 * R2_1 * (R2_1 + 16) * np.log((2 + B_1) / (1 + A_1))
            + 2 * (3 * R2_1 + 4) * (B_1 - A_1)
            + 2 * (np.power(B_1, 3) - np.power(A_1, 3))
        )
        retval[case1] = I1 + 0.25 * I3
        retval[case2] = 0.25 * I2
        # 1.597 is normalization s.t. kernel integral = 1 for particle mass = 1
        # rescaling from interval [0, 2) to [0, 1) requires mult. by 4
        return retval / 1.59689476201133 / np.power(h, 2) * 4

    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths, in units of pixels.

        noraise : bool
            If ``True``, suppress kernel validation errors. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """

        valid = sm_lengths >= self.min_valid_size * U.pix
        if np.logical_not(valid).any():
            self._validate_error(
                "martini.sph_kernels._CubicSplineKernel._validate:\n"
                f"SPH smoothing lengths must be >= {self.min_valid_size:f} px in "
                "size for CubicSplineKernel kernel integral "
                "approximation accuracy within 1%.\nThis check "
                "may be disabled by calling "
                "martini.martini.Martini.insert_source_in_cube with "
                "'skip_validation=True', but use this with "
                "care.",
                sm_lengths,
                valid,
                noraise=noraise,
                quiet=quiet,
            )
        return valid


class _GaussianKernel(_BaseSPHKernel):
    """
    Implementation of a (truncated) Gaussian kernel integral.

    Calculates the kernel integral over a pixel. The 3 integrals (along dx,
    dy, dz) are evaluated exactly, however the truncation is implemented
    approximately, erring on the side of integrating slightly further than
    the truncation radius.

    The Gaussian kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        (\\sqrt{2\\pi}\\sigma)^{-3}
        \\exp\\left(-\\frac{1}{2}\\left(\\frac{q}{\\sigma}\\right)^2\\right)
        &{\\rm for}\\;0 \\leq q < t\\\\
        0 &{\\rm for}\\;q > t
        \\end{cases}

    with :math:`\\sigma=(2\\sqrt{2\\log(2)})^{-1}`, s.t. FWHM = 1, and
    :math:`t` being the truncation radius.

    Parameters
    ----------
    truncate : float, optional
        Number of standard deviations at which to truncate kernel.
        Truncation radii <2 would lead to large errors and are not permitted.
        (Default: ``3``)
    """

    def __init__(self, truncate=3.0):
        self.truncate = truncate
        if self.truncate < 2:
            raise RuntimeError(
                "GaussianKernel with truncation <2sigma will "
                "cause large errors in total mass."
            )
        elif (self.truncate >= 2) and (self.truncate < 3):
            self.min_valid_size = 3.7
        elif (self.truncate >= 3) and (self.truncate < 4):
            self.min_valid_size = 2.3357
        elif (self.truncate >= 4) and (self.truncate < 5):
            self.min_valid_size = 1.1288
        elif (self.truncate >= 5) and (self.truncate < 6):
            self.min_valid_size = 0.45
        elif self.truncate >= 6:
            self.min_valid_size = 0.336

        self.norm = erf(self.truncate / np.sqrt(2)) - 2 * self.truncate / np.sqrt(
            2 * np.pi
        ) * np.exp(-np.power(self.truncate, 2) / 2)
        super().__init__()
        self.size_in_fwhm = self.truncate / (2 * np.sqrt(2 * np.log(2)))
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The Gaussian kernel is here defined as:

        .. math::
            W(q) = \\begin{cases}
            (\\sqrt{2\\pi}\\sigma)^{-3}
            \\exp\\left(-\\frac{1}{2}\\left(\\frac{q}{\\sigma}\\right)^2\\right)
            &{\\rm for}\\;0 \\leq q < t\\\\
            0 &{\\rm for}\\;q > t
            \\end{cases}

        with :math:`\\sigma=(2\\sqrt{2\\log(2)})^{-1}`, s.t. FWHM = 1, and
        :math:`t` being the truncation radius.

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """

        sig = 1 / (2 * np.sqrt(2 * np.log(2)))  # s.t. FWHM = 1
        return (
            np.where(
                q < self.truncate * sig,
                np.power(sig * np.sqrt(2 * np.pi), -3)
                * np.exp(-np.power(q / sig, 2) / 2),
                np.zeros(q.shape),
            )
            / self.norm
        )

    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Calculate the kernel integral over a pixel.

        The 3 integrals (along :math:`dx`, :math:`dy`, :math:`dz`) are evaluated exactly,
        however the truncation is implemented approximately, erring on the side of
        integrating slightly further than the truncation radius.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in pixels.
        mask : ~numpy.typing.ArrayLike or slice
            Boolean array, or slice. If the kernel has other internal properties to mask,
            it may use this.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel integral over the pixel area.
        """
        sig = 1 / (2 * np.sqrt(2 * np.log(2)))  # s.t. FWHM = 1
        dr = np.sqrt(np.power(dij, 2).sum(axis=0))
        with np.errstate(invalid="ignore"):
            zmax = np.sqrt(np.power(self.truncate, 2) - np.power(dr / h / sig, 2))
        zmax = np.where(self.truncate > dr / h / sig, zmax, 0)
        x0 = (dij[0] - 0.5 * U.pix) / h / np.sqrt(2) / sig
        x1 = (dij[0] + 0.5 * U.pix) / h / np.sqrt(2) / sig
        y0 = (dij[1] - 0.5 * U.pix) / h / np.sqrt(2) / sig
        y1 = (dij[1] + 0.5 * U.pix) / h / np.sqrt(2) / sig

        retval = (
            0.25 * erf(zmax / np.sqrt(2)) * (erf(x1) - erf(x0)) * (erf(y1) - erf(y0))
        )

        # explicit truncation not required as only pixels inside
        # truncation radius should be passed, next line useful for
        # testing, however
        retval[(dr - np.sqrt(0.5) * U.pix) / h / sig > self.truncate] = 0

        retval /= self.norm
        return retval * h.unit**-2

    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in units of pixels.

        noraise : bool
            If ``True``, suppress kernel validation errors. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """
        valid = sm_lengths >= self.min_valid_size * U.pix
        if np.logical_not(valid).any():
            self._validate_error(
                "martini.sph_kernels._GaussianKernel._validate:\n"
                f"SPH smoothing lengths must be >= {self.min_valid_size:f} px in "
                "size for GaussianKernel kernel integral "
                "approximation accuracy within 1%.\nThis check "
                "may be disabled by calling "
                "martini.martini.Martini.insert_source_in_cube with "
                "'skip_validation=True', but use this with "
                "care. Note that the minimum size depends on the kernel truncation.",
                sm_lengths,
                valid,
                noraise=noraise,
                quiet=quiet,
            )
        return valid


class DiracDeltaKernel(_BaseSPHKernel):
    """
    Implementation of a Dirac-delta kernel integral.

    The Dirac-delta kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        \\infty &{\\rm for}\\;q = 0\\\\
        0 &{\\rm for}\\;q > 0
        \\end{cases}

    Parameters
    ----------
    size_in_fwhm : float
        In principle the width of a Dirac-delta kernel is 0, but this would
        lead to no particles contributing to any pixels. Ideally this would
        be set to approximately the size of the pixel, but setting it to
        the smoothing length (1.0) is acceptable given the validation condition.
        (Default: 1.0)

    """

    max_valid_size = 0.5

    def __init__(self, size_in_fwhm=1.0):
        super().__init__()
        self.size_in_fwhm = size_in_fwhm
        self._rescale = 1  # need this to be present
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The Dirac-delta kernel is here defined as:

        .. math::
            W(q) = \\begin{cases}
            \\infty &{\\rm for}\\;q = 0\\\\
            0 &{\\rm for}\\;q > 0
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """

        return np.where(q, np.zeros(q.shape), np.inf * np.ones(q.shape))

    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Calculate the kernel integral over a pixel.

        The particles are approximated as point-like, ignoring any finite-sized
        kernel. This is a reasonable approximation provided the smoothing
        length is < 0.5 pixel in size, ideally << 1 pixel in size.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in pixels.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel integral over the pixel area.
        """
        return np.where((np.abs(dij) < 0.5 * U.pix).all(axis=0), 1, 0) * U.pix**-2

    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Check conditions for validity of kernel integral calculation.

        The Dirac-delta model approaches the exact integral when the smoothing
        length is << 1 pixel in size; at a minimum the smoothing length should
        be less than half the pixel size.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in units of pixels.

        noraise : bool
            If ``True``, suppress kernel validation errors. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """

        valid = sm_lengths <= self.max_valid_size * U.pix
        if np.logical_not(valid).any():
            self._validate_error(
                "martini.sph_kernels.DiracDeltaKernel._validate:\n"
                f"provided smoothing scale (FWHM) must be <= {self.max_valid_size:f} "
                "px in size for DiracDelta kernel to be a "
                "reasonable approximation. Call "
                "martini.martini.Martini.insert_source_in_cube with "
                "'skip_validation=True' to override at the "
                "cost of accuracy, but use this with care.",
                sm_lengths,
                valid,
                noraise=noraise,
                quiet=quiet,
            )
        return valid


class _AdaptiveKernel(_BaseSPHKernel):
    """
    Allows use of multiple kernels to adapt to sph kernel-to-pixel size ratio.

    Other provided kernels generally use approximations which break down if
    the ratio of the pixel size and the sph smoothing length are above or below
    some threshold. This (meta-)kernel accepts a list of other kernels in order
    of decreasing priority. The validity of the approximations used in each
    will be checked in turn and the first usable kernel for a given particle
    will be used to smooth the particle onto the pixel grid. Note that the
    initialized source and datacube instances are required as the smoothing
    lengths and pixel sizes must be known at initialization of the
    :class:`~martini.sph_kernels._AdaptiveKernel` class. Note that if ``skip_validation``
    is used, any particles with no valid kernel will default to the first kernel in the
    list.

    Parameters
    ----------
    kernels : iterable
        An iterable containing classes inheriting from
        :class:`~martini.sph_kernels._BaseSPHKernel`.
        Kernels to use, ordered by decreasing priority.

    """

    def __init__(self, kernels):
        self.kernels = kernels
        super().__init__()
        self.size_in_fwhm = None  # initialized during Martini.__init__
        self._rescale = None  # initialized during Martini.__init__
        return

    def _init_sm_lengths(self, source=None, datacube=None):
        """
        Determine kernel sizes in pixel units.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source providing the kernel sizes.

        datacube : martini.datacube.DataCube
            The datacube providing the pixel scale.
        """

        super()._init_sm_lengths(source=source, datacube=datacube)
        self.kernel_indices = -1 * np.ones(source.hsm_g.shape, dtype=int)
        for ik, K in enumerate(self.kernels):
            # if valid and not already assigned an earlier kernel, assign
            self.kernel_indices[
                np.logical_and(
                    self.kernel_indices == -1,
                    K._validate(self.sm_lengths * K._rescale, noraise=True, quiet=True),
                )
            ] = ik
        _sizes_in_fwhm = np.array([K.size_in_fwhm for K in self.kernels])
        self.size_in_fwhm = _sizes_in_fwhm[self.kernel_indices]
        # ensure default is 0th entry
        self.size_in_fwhm[self.kernel_indices == -1] = _sizes_in_fwhm[0]
        _rescales = np.array([K._rescale for K in self.kernels])
        self._rescale = _rescales[self.kernel_indices]
        # ensure default is 0th entry
        self._rescale[self.kernel_indices == -1] = _rescales[0]

        return

    def _apply_mask(self, mask):
        """
        Apply a mask to maskable attributes.

        Parameters
        ----------
        mask : ~numpy.typing.ArrayLike
            Boolean mask to apply to any maskable attributes.
        """
        self.size_in_fwhm = self.size_in_fwhm[mask]
        self._rescale = self._rescale[mask]
        self.kernel_indices = self.kernel_indices[mask]
        super()._apply_mask(mask)

        return

    def eval_kernel(self, r, h):
        """
        Evaluate the kernel, handling array casting and rescaling.

        Parameters
        ----------
        r : ~numpy.typing.ArrayLike or ~astropy.units.Quantity
            Distance parameter, same units as ``h``.
        h : ~numpy.typing.ArrayLike or ~astropy.units.Quantity
            Smoothing scale parameter (FWHM), same units as ``r``.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at position(s) ``r / h``.
        """

        return self.kernels[0].eval_kernel(r, h)

    def kernel(self, q):
        """
        Evaluate the kernel function of the preferred kernel.

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """
        return self.kernels[0].kernel(q)

    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Calculate the kernel integral over a pixel.

        Adaptively determines which kernel to use.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in pixels.
        mask : ~numpy.typing.ArrayLike or slice
            Boolean array, or slice. If the kernel has other internal properties to mask,
            it may use this.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Approximate kernel integral over the pixel area.
        """

        retval = np.zeros(h.shape) * h.unit**-2
        for ik in np.unique(self.kernel_indices[mask]):
            K = self.kernels[0] if ik == -1 else self.kernels[ik]
            kmask = self.kernel_indices[mask] == ik
            retval[kmask] = K._kernel_integral(dij[:, kmask], h[kmask])
        return retval

    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths (FWHM), in units of pixels.

        noraise : bool
            If ``True``, suppress kernel validation errors. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """

        valid = self.kernel_indices >= 0
        if np.logical_not(valid).any():
            self._validate_error(
                "martini.sph_kernels._AdaptiveKernel._validate:\n"
                "Some particles have no kernel candidate for which "
                "accuracy passes validation.\nThis check "
                "may be disabled by calling "
                "martini.martini.Martini.insert_source_in_cube with "
                "'skip_validation=True', but use this with "
                "care.\n",
                sm_lengths,
                valid,
                noraise=noraise,
                quiet=quiet,
            )

        return


class _QuarticSplineKernel(_BaseSPHKernel):
    """
    Implementation of the quartic spline kernel integral.

    The quartic spline (M5) kernel is used in the SPHENIX scheme (e.g. in Colibre). The
    exact integral is usually too slow to be practical; the implementation here
    approximates the kernel amplitude as constant across the pixel, which converges to
    within 1% of the exact integral provided the SPH smoothing lengths are at least 1.2385
    pixels in size.

    The quartic spline kernel is here defined as:

    .. math ::
        W(q) = \\frac{15625}{512\\pi}\\begin{cases}
        (1 - q)^4 - 5(\\frac{3}{5} - q)^4 + 10(\\frac{1}{5}-q)^4
        &{\\rm for}\\;0 \\leq q < \\frac{1}{5}\\\\
        (1 - q)^4 - 5(\\frac{3}{5} - q)^4
        &{\\rm for}\\;\\frac{1}{5} \\leq q < \\frac{3}{5}\\\\
        (1 - q)^4
        &{\\rm for}\\;\\frac{3}{5} \\leq q < 1\\\\
        0
        &{\\rm for}\\;q \\geq 1
        \\end{cases}

    """

    min_valid_size = 1.2385

    def __init__(self):
        super().__init__()
        _unscaled_fwhm = find_fwhm(lambda r: self.eval_kernel(r, 1))
        self.size_in_fwhm = 1 / _unscaled_fwhm
        self._rescale /= _unscaled_fwhm
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The quartic spline kernel is here defined as:

        .. math ::
            W(q) = \\frac{15625}{512\\pi}\\begin{cases}
            (1 - q)^4 - 5(\\frac{3}{5} - q)^4 + 10(\\frac{1}{5}-q)^4
            &{\\rm for}\\;0 \\leq q < \\frac{1}{5}\\\\
            (1 - q)^4 - 5(\\frac{3}{5} - q)^4
            &{\\rm for}\\;\\frac{1}{5} \\leq q < \\frac{3}{5}\\\\
            (1 - q)^4
            &{\\rm for}\\;\\frac{3}{5} \\leq q < 1\\\\
            0
            &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """

        W = np.zeros(q.shape)
        mask1 = q < 0.2
        W[mask1] = (
            np.power(1 - q[mask1], 4)
            - 5 * np.power(0.6 - q[mask1], 4)
            + 10 * np.power(0.2 - q[mask1], 4)
        )
        mask2 = np.logical_and(q >= 0.2, q < 0.6)
        W[mask2] = np.power(1 - q[mask2], 4) - 5 * np.power(0.6 - q[mask2], 4)
        mask3 = np.logical_and(q >= 0.6, q < 1)
        W[mask3] = np.power(1 - q[mask3], 4)
        W *= 15625 / 512 / np.pi
        return W

    def _kernel_integral(self, dij, h, mask=np.s_[...]):
        """
        Calculate the kernel integral over a pixel.

        The formula used approximates the kernel amplitude as constant across
        the pixel area and converges to the true value within 1% for smoothing
        lengths >= 1.2385 pixels.

        Parameters
        ----------
        dij : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Distances from pixel centre to particle positions, in pixels.
        h : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths, in pixels.
        mask : ~numpy.typing.ArrayLike or slice
            Boolean array, or slice. If the kernel has other internal properties to mask,
            it may use this.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Approximate kernel integral over the pixel area.
        """

        dr = np.sqrt(np.power(dij, 2).sum(axis=0))
        retval = np.zeros(h.shape)
        R = (dr / h).to_value(U.dimensionless_unscaled)

        def IA(R, z, A):
            q = np.sqrt(np.power(z, 2) + np.power(R, 2))
            R2 = np.power(R, 2)
            return (
                np.power(A, 4) * z
                - 2 * np.power(A, 3) * z * q
                + 2 * np.power(A, 2) * z * (3 * R2 + np.power(z, 2))
                - A * R2 * (4 * np.power(A, 2) + 3 * R2) * np.arcsinh(z / R) / 2
                - A * z * q * (5 * R2 + 2 * np.power(z, 2)) / 2
                + R2 * R2 * z
                + 2 * R2 * np.power(z, 3) / 3
                + np.power(z, 5) / 5
            )

        m1 = np.logical_and(R > 0, R < 0.2)
        retval[m1] += 10 * IA(R[m1], np.sqrt(0.2**2 - np.power(R[m1], 2)), 0.2)
        m2 = np.logical_and(R > 0, R < 0.6)
        retval[m2] -= 5 * IA(R[m2], np.sqrt(0.6**2 - np.power(R[m2], 2)), 0.6)
        m3 = np.logical_and(R > 0, R < 1.0)
        retval[m3] += IA(R[m3], np.sqrt(1 - np.power(R[m3], 2)), 1.0)
        retval[R == 0] = 384 / 3125
        # factor of 2 is because all integrals above are half-intervals
        retval *= 2 * 15625 / 512 / np.pi
        return retval / np.power(h, 2)

    def _validate(self, sm_lengths, noraise=False, quiet=False):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels.
            Particle smoothing lengths, in units of pixels.

        noraise : bool
            If ``True``, suppress kernel validation errors. (Default: ``False``)

        quiet : bool
            If ``True``, suppress reports on smoothing lengths. (Default: ``False``)
        """

        valid = sm_lengths >= self.min_valid_size * U.pix
        if np.logical_not(valid).any():
            self._validate_error(
                "martini.sph_kernels._QuarticSplineKernel._validate:\n"
                "SPH smoothing lengths must be >= {self.min_valid_size:f} px in "
                "size for QuarticSplineKernel kernel integral "
                "approximation accuracy within 1%.\nThis check "
                "may be disabled by calling "
                "martini.martini.Martini.insert_source_in_cube with "
                "'skip_validation=True', but use this with "
                "care.",
                sm_lengths,
                valid,
                noraise=noraise,
                quiet=quiet,
            )
        return valid


class WendlandC2Kernel(_AdaptiveKernel):
    """
    Implementation of the Wendland C2 kernel integral.

    The Wendland C2 kernel is used in the EAGLE code and derivatives (not in
    Gadget/Gadget2!). The exact integral is usually too slow to be practical;
    the implementation here approximates the kernel amplitude as constant
    across the pixel, which converges to within 1% of the exact integral
    provided the SPH smoothing lengths are at least 1.51 pixels in size.

    The WendlandC2 kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        \\frac{21}{2\\pi}(1-q)^4(4q+1)
        &{\\rm for}\\;0 \\leq q < 1\\\\
        0 &{\\rm for}\\;q \\geq 1
        \\end{cases}

    This class falls back to the :class:`~martini.sph_kernels.DiracDeltaKernel` and
    :class:`~martini.sph_kernels.GaussianKernel` when the approximation used for the
    surface integral of the WendlandC2 kernel is not accurate to within better than 1%.
    To strictly use the WendlandC2 kernel, use
    :class:`~martini.sph_kernels._WendlandC2Kernel` instead.
    """

    def __init__(self):
        super().__init__(
            (_WendlandC2Kernel(), DiracDeltaKernel(), _GaussianKernel(truncate=6.0))
        )
        self.size_in_fwhm = None  # initialized during Martini.__init__
        self._rescale = None  # initialized during Martini.__init__
        return

    def kernel(self, q):
        """
        Evaluate the kernel function of the WendlandC2 kernel.

        The WendlandC2 kernel is here defined as:

        .. math::
            W(q) = \\begin{cases}
            \\frac{21}{2\\pi}(1-q)^4(4q+1)
            &{\\rm for}\\;0 \\leq q < 1\\\\
            0 &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """
        return super().kernel(q)


class WendlandC6Kernel(_AdaptiveKernel):
    """
    Implementation of the Wendland C6 kernel integral.

    The Wendland C6 kernel is used in the Magneticum code (not in
    Gadget/Gadget2!). The exact integral is usually too slow to be practical;
    the implementation here approximates the kernel amplitude as constant
    across the pixel, which converges to within 1% of the exact integral
    provided the SPH smoothing lengths are at least 1.29 pixels in size.

    The WendlandC6 kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        \\frac{1365}{64 \\pi} (1 - q)^8 (1 + 8q + 25q^2 + 32q^3)
        &{\\rm for}\\;0 \\leq q < 1\\\\
        0 &{\\rm for}\\;q \\geq 1
        \\end{cases}

    This class falls back to the :class:`~martini.sph_kernels.DiracDeltaKernel` and
    :class:`~martini.sph_kernels.GaussianKernel` when the approximation used for the
    surface integral of the WendlandC6 kernel is not accurate to within better than 1%.
    To strictly use the WendlandC6 kernel, use
    :class:`~martini.sph_kernels._WendlandC6Kernel` instead.
    """

    def __init__(self):
        super().__init__(
            (_WendlandC6Kernel(), DiracDeltaKernel(), _GaussianKernel(truncate=6.0))
        )
        self.size_in_fwhm = None  # initialized during Martini.__init__
        self._rescale = None  # initialized during Martini.__init__
        return

    def kernel(self, q):
        """
        Evaluate the kernel function of the WendlandC6 kernel.

        The WendlandC6 kernel is here defined as:

        .. math::
            W(q) = \\begin{cases}
            \\frac{1365}{64 \\pi} (1 - q)^8 (1 + 8q + 25q^2 + 32q^3)
            &{\\rm for}\\;0 \\leq q < 1\\\\
            0 &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """
        return super().kernel(q)


class CubicSplineKernel(_AdaptiveKernel):
    """
    Implementation of the cubic spline (M4) kernel integral.

    The cubic spline is the 'classic' SPH kernel. The exact integral is usually
    too slow to be practical; the implementation here approximates the kernel
    amplitude as constant across the pixel, which converges to within 1% of
    the exact integral provided the SPH smoothing lengths are at least 1.16
    pixels in size.

    The cubic spline kernel is here defined as:

    .. math ::
        W(q) = \\frac{8}{\\pi}\\begin{cases}
        (1 - 6q^2(1 - \\frac{q}{2}))
        &{\\rm for}\\;0 \\leq q < \\frac{1}{2}\\\\
        2(1 - q)^3
        &{\\rm for}\\;\\frac{1}{2} \\leq q < 1\\\\
        0
        &{\\rm for}\\;q \\geq 1
        \\end{cases}

    This class falls back to the :class:`~martini.sph_kernels.DiracDeltaKernel` and
    :class:`~martini.sph_kernels.GaussianKernel` when the approximation used for the
    surface integral of the cubic spline kernel is not accurate to within better than 1%.
    To strictly use the cubic spline kernel, use
    :class:`~martini.sph_kernels._CubicSplineKernel` instead.
    """

    def __init__(self):
        super().__init__(
            (_CubicSplineKernel(), DiracDeltaKernel(), _GaussianKernel(truncate=6.0))
        )
        self.size_in_fwhm = None  # initialized during Martini.__init__
        self._rescale = None  # initialized during Martini.__init__
        return

    def kernel(self, q):
        """
        Evaluate the kernel function of the cubic spline kernel.

        The cubic spline kernel is here defined as:

        .. math ::
            W(q) = \\frac{8}{\\pi}\\begin{cases}
            (1 - 6q^2(1 - \\frac{q}{2}))
            &{\\rm for}\\;0 \\leq q < \\frac{1}{2}\\\\
            2(1 - q)^3
            &{\\rm for}\\;\\frac{1}{2} \\leq q < 1\\\\
            0
            &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """
        return super().kernel(q)


class GaussianKernel(_AdaptiveKernel):
    """
    Implementation of a (truncated) Gaussian kernel integral.

    Calculates the kernel integral over a pixel. The 3 integrals (along :math:`dx`,
    :math:`dy`, :math:`dz`) are evaluated exactly, however the truncation is implemented
    approximately, erring on the side of integrating slightly further than
    the truncation radius.

    The Gaussian kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        (\\sqrt{2\\pi}\\sigma)^{-3}
        \\exp\\left(-\\frac{1}{2}\\left(\\frac{q}{\\sigma}\\right)^2\\right)
        &{\\rm for}\\;0 \\leq q < t\\\\
        0 &{\\rm for}\\;q > t
        \\end{cases}

    with :math:`\\sigma=(2\\sqrt{2\\log(2)})^{-1}`, s.t. FWHM = 1, and
    :math:`t` being the truncation radius.

    This class falls back to the :class:`~martini.sph_kernels.DiracDeltaKernel` and
    :class:`~martini.sph_kernels.GaussianKernel` (with large truncation) when the
    approximation used for the surface integral of the Gaussian kernel is not accurate to
    within better than 1%. To strictly use the Gaussian kernel, use
    :class:`~martini.sph_kernels._GaussianKernel` instead.

    Parameters
    ----------
    truncate : float, optional
        Number of standard deviations at which to truncate kernel.
        Truncation radii <2 would lead to large errors and are not permitted.
        (Default: ``3``)
    """

    def __init__(self, truncate=3.0):
        super().__init__(
            (
                _GaussianKernel(truncate=truncate),
                DiracDeltaKernel(),
                _GaussianKernel(truncate=6.0),
            )
        )
        self.size_in_fwhm = None  # initialized during Martini.__init__
        self._rescale = None  # initialized during Martini.__init__
        return

    def kernel(self, q):
        """
        Evaluate the kernel function of the Gaussian kernel.

        The Gaussian kernel is here defined as:

        .. math::
            W(q) = \\begin{cases}
            (\\sqrt{2\\pi}\\sigma)^{-3}
            \\exp\\left(-\\frac{1}{2}\\left(\\frac{q}{\\sigma}\\right)^2\\right)
            &{\\rm for}\\;0 \\leq q < t\\\\
            0 &{\\rm for}\\;q > t
            \\end{cases}

        with :math:`\\sigma=(2\\sqrt{2\\log(2)})^{-1}`, s.t. FWHM = 1, and
        :math:`t` being the truncation radius.

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """
        return super().kernel(q)


class QuarticSplineKernel(_AdaptiveKernel):
    """
    Implementation of the quartic spline kernel integral.

    The quartic spline (M5) kernel is used in the SPHENIX scheme (e.g. in Colibre). The
    exact integral is usually too slow to be practical; the implementation here
    approximates the kernel amplitude as constant across the pixel, which converges to
    within 1% of the exact integral provided the SPH smoothing lengths are at least 1.2385
    pixels in size.

    The quartic spline kernel is here defined as:

    .. math ::
        W(q) = \\frac{15625}{512\\pi}\\begin{cases}
        (1 - q)^4 - 5(\\frac{3}{5} - q)^4 + 10(\\frac{1}{5}-q)^4
        &{\\rm for}\\;0 \\leq q < \\frac{1}{5}\\\\
        (1 - q)^4 - 5(\\frac{3}{5} - q)^4
        &{\\rm for}\\;\\frac{1}{5} \\leq q < \\frac{3}{5}\\\\
        (1 - q)^4
        &{\\rm for}\\;\\frac{3}{5} \\leq q < 1\\\\
        0
        &{\\rm for}\\;q \\geq 1
        \\end{cases}

    This class falls back to the :class:`~martini.sph_kernels.DiracDeltaKernel` and
    :class:`~martini.sph_kernels.GaussianKernel` when the approximation used for the
    surface integral of the quartic spline kernel is not accurate to within better
    than 1%. To strictly use the quartic spline kernel, use
    :class:`~martini.sph_kernels._QuarticSplineKernel` instead.
    """

    def __init__(self):
        super().__init__(
            (_QuarticSplineKernel(), DiracDeltaKernel(), _GaussianKernel(truncate=6.0))
        )
        self.size_in_fwhm = None  # initialized during Martini.__init__
        self._rescale = None  # initialized during Martini.__init__
        return

    def kernel(self, q):
        """
        Evaluate the kernel function of the quartic spline kernel.

        The quartic spline kernel is here defined as:

        .. math ::
            W(q) = \\frac{15625}{512\\pi}\\begin{cases}
            (1 - q)^4 - 5(\\frac{3}{5} - q)^4 + 10(\\frac{1}{5}-q)^4
            &{\\rm for}\\;0 \\leq q < \\frac{1}{5}\\\\
            (1 - q)^4 - 5(\\frac{3}{5} - q)^4
            &{\\rm for}\\;\\frac{1}{5} \\leq q < \\frac{3}{5}\\\\
            (1 - q)^4
            &{\\rm for}\\;\\frac{3}{5} \\leq q < 1\\\\
            0
            &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : ~numpy.typing.ArrayLike
            Dimensionless distance parameter.

        Returns
        -------
        out : ~numpy.typing.ArrayLike
            Kernel value at positions ``q``.
        """
        return super().kernel(q)


class AdaptiveKernel(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "New pre-configured adaptive kernels have been implemented in v2.0.3, see "
            "the new SPH Kernels page in the documentation for details. You most likely "
            "want to use WendlandC2Kernel(), WendlandC6Kernel(), CubicSplineKernel() or "
            "QuarticSplineKernel() where you previously used AdaptiveKernel(...)."
        )
