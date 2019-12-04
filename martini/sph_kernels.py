from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from scipy.special import erf
from scipy.optimize import fsolve
from warnings import warn


def find_fwhm(f):
    return 2 * fsolve(lambda q: f(q) - f(np.zeros(1)) / 2, .5)


class _BaseSPHKernel(object):
    """
    Abstract base class for classes implementing SPH kernels to inherit from.

    Classes inheriting from _BaseSPHKernel must implement three methods:
    `kernel`, `kernel_integral` and `validate`.

    `kernel` should define the kernel function, normalized such that its volume
    integral is 1.

    `kernel_integral` should define the integral of the kernel over a pixel
    given the distance between the pixel centre and the particle centre, and
    the smoothing length (both in units of pixels). The integral should be
    normalized so that evaluated over the entire kernel it is equal to 1.

    `validate` should check whether any approximations converge to sufficient
    accuracy (for instance, depending on the ratio of the pixel size and
    smoothing length), and raise an error if not.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self._rescale = 1
        warn('The definition of the smoothing length has changed since earlier'
             ' versions. Now all kernel modules expect the smoothing lengths'
             ' provided (via the source module) to be the FWHM of the kernel.'
             ' This is to avoid ambiguity in the definition. Users of'
             ' simulation-specific sources (TNGSource, EAGLESource, ...) do'
             ' not need to worry about this change: the sources have been'
             ' updated accordingly.')
        return

    def px_weight(self, dij, h):
        """
        Calculate kernel integral using scaled smoothing lengths.

        This is the method that should be called by other modules in
        martini, rather than 'kernel_integral'.

        Parameters
        ----------
        dij : Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : Quantity, with dimensions of pixels
            Particle smoothing lengths, defined as the kernel FWHM, in pixels.

        Returns
        -------
        out : Quantity, with dimensions of pixels^-2
            Integral of smoothing kernel over pixel, per unit pixel area.
        """

        return self.kernel_integral(dij, h * self._rescale)

    def confirm_validation(self, sm_lengths):
        """
        Verify kernel accuracy using scaled smoothing lengths.

        This is the method that should be called by other modules in
        martini, rather than 'validate'.

        Parameters
        ----------
        sm_lengths : Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        return self.validate(sm_lengths * self._rescale)

    def eval_kernel(self, r, h):
        """
        Evaluate the kernel, handling array casting and rescaling.

        Parameters
        ----------
        r : array_like or Quantity
            Distance parameter, same units as h.
        h : array_like or Quantity
            Smoothing scale parameter (FWHM), same units as r.

        Returns
        -------
        out : array_like
            Kernel value at position(s) r / h.
        """

        q = np.array(r / h / self._rescale)
        if isinstance(q, U.Quantity):
            q = q.to(U.dimensionless_unscaled).value
        scalar_input = q.ndim == 0
        W = self.kernel(q)
        W /= np.power(self._rescale, 3)
        if scalar_input:
            return W.item()
        else:
            return W

    @abstractmethod
    def kernel(self, q):
        """
        Abstract method; evaluate the kernel.

        Parameters
        ----------
        q : array_like
            Dimensionless distance parameter.

        Returns
        -------
        out : array_like
            Kernel value at positions q.
        """
        pass

    @abstractmethod
    def kernel_integral(self, dij, h):
        """
        Abstract method; calculate the kernel integral over a pixel.

        Parameters
        ----------
        dij : Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : Quantity, with dimensions of pixels
            Particle smoothing lengths (FWHM), in pixels.

        Returns
        -------
        out : Quantity, with dimensions of pixels^-2
            Integral of smoothing kernel over pixel, per unit pixel area.
        """

        pass

    @abstractmethod
    def validate(self, sm_lengths):
        """
        Abstract method; check conditions for validity of kernel integral
        calculation.

        Some approximations may only converge if the ratio of the pixel size
        and the smoothing length is sufficiently large, or sufficiently small.
        This method should check these conditions and raise errors when
        appropriate. The smoothing lengths are provided normalized to the pixel
        size.

        Parameters
        ----------
        sm_lengths : Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        pass


class WendlandC2Kernel(_BaseSPHKernel):
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
        q : array_like
            Dimensionless distance parameter.

        Returns
        -------
        out : array_like
            Kernel value at positions q.
        """

        W = np.where(
            q < 1,
            np.power(1 - q, 4) * (4 * q + 1),
            np.zeros(q.shape)
        )
        W *= (21 / 2 / np.pi)
        return W

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel.

        The formula used approximates the kernel amplitude as constant
        across the pixel area and converges to the true value within 1%
        for smoothing lengths >= 2 pixels.

        Parameters
        ----------
        dij : Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : Quantity, with dimensions of pixels
            Particle smoothing lengths (FWHM), in pixels.

        Returns
        -------
        out : array_like
            Approximate kernel integral over the pixel area.
        """

        dr2 = np.power(dij, 2).sum(axis=0)
        retval = np.zeros(h.shape)
        R2 = dr2 / (h * h)
        retval[R2 == 0] = 2. / 3.
        use = np.logical_and(R2 < 1, R2 != 0)
        R2 = R2[use]
        A = np.sqrt(1 - R2)
        retval[use] = 5 * R2 * R2 * (.5 * R2 + 3) * \
            np.log((1 + A) / np.sqrt(R2)) + \
            A * (-27. / 2. * R2 * R2 - 14. / 3. * R2 + 2. / 3.)
        norm = 21 / 2 / np.pi
        return retval * norm / np.power(h, 2)

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : Quantity, with dimensions of pixels
            Particle smoothing lengths (FWHM), in units of pixels.
        """

        if (sm_lengths < 1.51 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.WendlandC2Kernel.validate:"
                               " SPH smoothing lengths must be >= 1.51 px in "
                               "size for WendlandC2 kernel integral "
                               "approximation accuracy within 1%. This check "
                               "may be disabled by calling "
                               "Martini.Martini.insert_source_in_cube with "
                               "'skip_validation=True', but use this with "
                               "care.")

        return


class WendlandC6Kernel(_BaseSPHKernel):
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
        q : array_like
            Dimensionless distance parameter.

        Returns
        -------
        out : array_like
            Kernel value at positions q.
        """

        W = np.where(
            q < 1,
            np.power(1 - q, 8)
            * (1 + 8 * q + 25 * np.power(q, 2) + 32 * np.power(q, 3)),
            np.zeros(q.shape)
        )
        W *= (1365 / 64 / np.pi)
        return W

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel. Not currently implemented.

        Parameters
        ----------
        dij : Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.

        Returns
        -------
        out : array_like
            Approximate kernel integral over the pixel area.
        """

        def indef(R, z):
            # integral(1 - sqrt(R^2 + z^2))^8 \
            # (1 + 8 sqrt(R^2 + z^2) + 25 (R^2 + z^2) \
            # + 32 (R^2 + z^2)^1.5) dz =
            return -231 * np.power(R, 10) * z \
                - 385 * np.power(R, 8) * np.power(z, 3) \
                - 1155 * np.power(R, 8) * z \
                - 462 * np.power(R, 6) * np.power(z, 5) \
                - 1540 * np.power(R, 6) * np.power(z, 3) \
                - 462 * np.power(R, 6) * z \
                - 330 * np.power(R, 4) * np.power(z, 7) \
                - 1386 * np.power(R, 4) * np.power(z, 5) \
                - 462 * np.power(R, 4) * np.power(z, 3) \
                + 66 * np.power(R, 4) * z \
                - (128 + 1 / 3) * np.power(R, 2) * np.power(z, 9) \
                - 660 * np.power(R, 2) * np.power(z, 7) \
                - 277.2 * np.power(R, 2) * np.power(z, 5) \
                + 44 * np.power(R, 2) * np.power(z, 3) \
                + 8 / 3 * np.power(z, 11) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + (16 + 4 / 15) * np.power(R, 2) * np.power(z, 9) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 70.4 * np.power(z, 9) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 360.8 * np.power(R, 2) * np.power(z, 7) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 132 * np.power(z, 7) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 550 * np.power(R, 2) * np.power(z, 5) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                - 11 * np.power(R, 2) * z \
                + 7.21875 * np.power(R, 12) * (
                    np.log(np.sqrt(np.power(R, 2) + np.power(z, 2)) + z)) \
                + 24.7813 * np.power(R, 10) * z * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 173.25 * np.power(R, 10) * (
                    np.log(np.sqrt(np.power(R, 2) + np.power(z, 2)) + z)) \
                + 530.75 * np.power(R, 8) * z * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 288.75 * np.power(R, 8) * (
                    np.log(np.sqrt(np.power(R, 2) + np.power(z, 2)) + z)) \
                + 47.4792 * np.power(R, 8) * np.power(z, 3) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 767.25 * np.power(R, 6) * z * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 58.0167 * np.power(R, 6) * np.power(z, 5) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 819.5 * np.power(R, 6) * np.power(z, 3) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 41.7 * np.power(R, 4) * np.power(z, 7) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 752.4 * np.power(R, 4) * np.power(z, 5) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                + 896.5 * np.power(R, 4) * np.power(z, 3) * (
                    np.sqrt(np.power(R, 2) + np.power(z, 2))) \
                - 21 * np.power(z, 11) \
                - (128 + 1 / 3) * np.power(z, 9) \
                - 66 * np.power(z, 7) \
                + 13.2 * np.power(z, 5) \
                - 11 / 3 * np.power(z, 3) \
                + z

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

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        if (sm_lengths < 1.29 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.WendlandC6Kernel.validate:"
                               " SPH smoothing lengths must be >= 1.29 px in "
                               "size for WendlandC6 kernel integral "
                               "approximation accuracy within 1%. This check "
                               "may be disabled by calling "
                               "Martini.Martini.insert_source_in_cube with "
                               "'skip_validation=True', but use this with"
                               "care.")
        return


class CubicSplineKernel(_BaseSPHKernel):
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

    """

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
            (1 - 6q^2(1 - \\frac{q}{2}))
            &{\\rm for}\\;0 \\leq q < \\frac{1}{2}\\\\
            2(1 - q)^3
            &{\\rm for}\\;\\frac{1}{2} \\leq q < 1\\\\
            0
            &{\\rm for}\\;q \\geq 1
            \\end{cases}

        Parameters
        ----------
        q : array_like
            Dimensionless distance parameter.

        Returns
        -------
        out : array_like
            Kernel value at positions q.
        """

        W = np.where(
            q < .5,
            1 - 6 * np.power(q, 2) + 6 * np.power(q, 3),
            2 * np.power(1 - q, 3)
        )
        W[q > 1] = 0
        W *= 8 / np.pi
        return W

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel.

        The formula used approximates the kernel amplitude as constant across
        the pixel area and converges to the true value within 1% for smoothing
        lengths >= 2.5 pixels.

        Parameters
        ----------
        dij : Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.

        Returns
        -------
        out : array_like
            Approximate kernel integral over the pixel area.
        """

        dij *= 2  # changes interval from [0, 2) to [0, 1)
        dr2 = np.power(dij, 2).sum(axis=0)
        retval = np.zeros(h.shape)
        R2 = dr2 / (h * h)
        retval[R2 == 0] = 11. / 16. + .25 * .25
        case1 = np.logical_and(R2 > 0, R2 <= 1)
        case2 = np.logical_and(R2 > 1, R2 <= 4)

        R2_1 = R2[case1]
        R2_2 = R2[case2]
        A_1 = np.sqrt(1 - R2_1)
        B_1 = np.sqrt(4 - R2_1)
        B_2 = np.sqrt(4 - R2_2)
        I1 = A_1 - .5 * np.power(A_1, 3) - 1.5 * R2_1 * A_1 + 3. / 32. * A_1 \
            * (3 * R2_1 + 2) + 9. / 32. * R2_1 * R2_1 \
            * (np.log(1 + A_1) - np.log(np.sqrt(R2_1)))
        I2 = -B_2 * (3 * R2_2 + 56) / 4. \
            - 3. / 8. * R2_2 * (R2_2 + 16) \
            * np.log((2 + B_2) / np.sqrt(R2_2)) \
            + 2 * (3 * R2_2 + 4) * B_2 + 2 * np.power(B_2, 3)
        I3 = -B_1 * (3 * R2_1 + 56) / 4. + A_1 * (4 * R2_1 + 50) / 8. \
            - 3. / 8. * R2_1 * (R2_1 + 16) * np.log((2 + B_1) / (1 + A_1)) \
            + 2 * (3 * R2_1 + 4) * (B_1 - A_1) \
            + 2 * (np.power(B_1, 3) - np.power(A_1, 3))
        retval[case1] = I1 + .25 * I3
        retval[case2] = .25 * I2
        # 1.597 is normalization s.t. kernel integral = 1 for particle mass = 1
        # rescaling from interval [0, 2) to [0, 1) requires mult. by 4
        return retval / 1.59689476201133 / np.power(h, 2) * 4

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        if (sm_lengths < 1.16 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.CubicSplineKernel.validate"
                               ": SPH smoothing lengths must be >= 1.16 px in "
                               "size for CubicSplineKernel kernel integral "
                               "approximation accuracy within 1%. This check "
                               "may be disabled by calling "
                               "Martini.Martini.insert_source_in_cube with "
                               "'skip_validation=True', but use this with"
                               "care.")
        return


class GaussianKernel(_BaseSPHKernel):
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
        Number of standard deviations at which to truncate kernel (default=3).
        Truncation radii <2 may lead to large errors and are not recommended.
    """

    def __init__(self, truncate=3):
        self.truncate = truncate
        self.norm = erf(self.truncate / np.sqrt(2)) - 2 * self.truncate \
            / np.sqrt(2 * np.pi) * np.exp(-np.power(self.truncate, 2) / 2)
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
        q : array_like
            Dimensionless distance parameter.

        Returns
        -------
        out : array_like
            Kernel value at positions q.
        """

        sig = 1 / (2 * np.sqrt(2 * np.log(2)))  # s.t. FWHM = 1
        return np.where(
            q < self.truncate * sig,
            np.power(sig * np.sqrt(2 * np.pi), -3)
            * np.exp(-np.power(q / sig, 2) / 2),
            np.zeros(q.shape)
        ) / self.norm

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel.

        The 3 integrals (along dx, dy, dz) are evaluated exactly, however the
        truncation is implemented approximately, erring on the side of
        integrating slightly further than the truncation radius.

        Parameters
        ----------
        dij : Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : Quantity, with dimensions of pixels
            Particle smoothing lengths (FWHM), in pixels.

        Returns
        -------
        out : array_like
            Kernel integral over the pixel area.
        """
        sig = 1 / (2 * np.sqrt(2 * np.log(2)))  # s.t. FWHM = 1
        dr = np.sqrt(np.power(dij, 2).sum(axis=0))
        zmax = np.where(
            self.truncate > dr / h / sig,
            np.sqrt(
                np.power(self.truncate, 2)
                - np.power(dr / h / sig, 2)
            ),
            0
        )
        x0 = (dij[0] - .5 * U.pix) / h / np.sqrt(2) / sig
        x1 = (dij[0] + .5 * U.pix) / h / np.sqrt(2) / sig
        y0 = (dij[1] - .5 * U.pix) / h / np.sqrt(2) / sig
        y1 = (dij[1] + .5 * U.pix) / h / np.sqrt(2) / sig

        retval = .25 \
            * erf(zmax / np.sqrt(2)) \
            * (erf(x1) - erf(x0)) \
            * (erf(y1) - erf(y0))

        # explicit truncation not required as only pixels inside
        # truncation radius should be passed, next line useful for
        # testing, however
        retval[(dr - np.sqrt(.5) * U.pix) / h / sig > self.truncate] = 0

        retval /= self.norm
        return retval * h.unit ** -2

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : Quantity, with dimensions of pixels
            Particle smoothing lengths (FWHM), in units of pixels.
        """

        if self.truncate < 1:
            raise RuntimeError("GaussianKernel with truncation <1sigma will "
                               "cause large errors in total mass.")
        elif (self.truncate >= 1) and (self.truncate < 2):
            lims = (0, 20)
        elif (self.truncate >= 2) and (self.truncate < 3):
            lims = (0, 7)
        elif (self.truncate >= 3) and (self.truncate < 4):
            lims = (0, 3)
        elif (self.truncate >= 4) and (self.truncate < 5):
            lims = (.4, 1)
        elif (self.truncate >= 5) and (self.truncate < 6):
            lims = (.4, .5)
        elif (self.truncate >= 6):
            lims = None
            warn("GaussianKernel with truncation >=6sigma may unnecessarily "
                 "slow down computation.")
        if (lims is not None) and \
           (np.logical_and(
               sm_lengths > lims[0] * U.pix,
               sm_lengths < lims[1] * U.pix
           ).any()):
            raise RuntimeError("Martini.sph_kernels.GaussianKernel.validate: "
                               "SPH smoothing lengths must not be in interval "
                               "[{0:.1f}, {1:.1f}] px ".format(*lims) +
                               "for Gaussian kernel integral approximation "
                               "accuracy within 1%. "
                               "This check may be disabled by calling "
                               "Martini.Martini.insert_source_in_cube with "
                               "'skip_validation=True', but use this with"
                               "care.")
        return


class DiracDeltaKernel(_BaseSPHKernel):
    """
    Implementation of a Dirac-delta kernel integral.

    The Dirac-delta kernel is here defined as:

    .. math::
        W(q) = \\begin{cases}
        \\infty &{\\rm for}\\;q = 0\\\\
        0 &{\\rm for}\\;q > 0
        \\end{cases}

    """

    def __init__(self):
        super().__init__()
        # In principle the size for a DiracDelta kernel is 0, but this would
        # lead to no particles being used. Ideally we would want ~the pixel
        # size here, but the sph smoothing length is acceptable.
        self.size_in_fwhm = 1
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
        q : array_like
            Dimensionless distance parameter.

        Returns
        -------
        out : array_like
            Kernel value at positions q.
        """

        return np.where(q, np.inf * np.ones(q.shape), np.zeros(q.shape))

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel.

        The particles are approximated as point-like, ignoring any finite-sized
        kernel. This is a reasonable approximation provided the smoothing
        length is < 0.5 pixel in size, ideally << 1 pixel in size.

        Parameters
        ----------
        dij : Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.
        h : Quantity, with dimensions of pixels
            Particle smoothing lengths (FWHM), in pixels.

        Returns
        -------
        out : array_like
            Kernel integral over the pixel area.
        """

        return np.where((np.abs(dij) < 0.5 * U.pix).all(axis=0), 1, 0) \
            * U.pix ** -2

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        The Dirac-delta model approaches the exact integral when the smoothing
        length is << 1 pixel in size; at a minimum the smoothing length should
        be less than half the pixel size.

        Parameters
        ----------
        sm_lengths : Quantity, with dimensions of pixels
            Particle smoothing lengths (FWHM), in units of pixels.
        """

        if (sm_lengths > .5 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.DiracDeltaKernel.validate:"
                               " provided smoothing scale (FWHM) must be <= 1 "
                               "px in size for DiracDelta kernel to be a "
                               "reasonable approximation. Call "
                               "Martini.Martini.insert_source_in_cube with "
                               "'skip_validation=True' to override, at the "
                               "cost of accuracy.")
        return
