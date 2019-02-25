from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from scipy.special import erf


class _BaseSPHKernel(object):
    """
    Abstract base class for classes implementing SPH kernels to inherit from.

    Classes inheriting from _BaseSPHKernel must implement three methods:
    'kernel_integral', 'validate' and 'size_in_h'.

    'kernel_integral' should define the integral of the kernel over a pixel
    given the distance between the pixel centre and the particle centre, and
    the smoothing length (both in units of pixels). The integral should be
    normalized so that evaluated over the entire kernel it is equal to 1.

    'validate' should check whether any approximations converge to sufficient
    accuracy (for instance, depending on the ratio of the pixel size and
    smoothing length), and raise an error if not.

    'size_in_h' should return the maximum distance in units of the sph
    smoothing length where the kernel is non-zero.
    """

    __metaclass__ = ABCMeta

    def __init__(self, rescale_sph_h=1):
        self.rescale_sph_h = rescale_sph_h
        return

    def px_weight(self, dij, h):
        return self.kernel_integral(dij, h * self.rescale_sph_h)

    def confirm_validation(self, sm_lengths):
        return self.validate(sm_lengths * self.rescale_sph_h)

    @abstractmethod
    def kernel_integral(self, dij, h):
        """
        Abstract method; calculate the kernel integral over a pixel.

        Parameters
        ----------
        dij : astropy.units.Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.

        h : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.
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
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        pass

    @abstractmethod
    def size_in_h(self):
        """
        Abstract method; return the maximum distance where the kernel is non-
        zero, in units of the SPH smoothing parameter of the particles.
        """
        pass


class WendlandC2Kernel(_BaseSPHKernel):
    """
    Implementation of the Wendland C2 kernel integral.

    The Wendland C2 kernel is used in the EAGLE code and derivatives (not in
    Gadget/Gadget2!). The exact integral is usually too slow to be practical;
    the implementation here approximates the kernel amplitude as constant
    across the pixel, which converges to within 1% of the exact integral
    provided the SPH smoothing lengths are at least 2 pixels in size.

    Parameters
    ----------
    rescale_sph_h : float
        Factor by which to rescale SPH smoothing lengths. This can be used to
        adjust particle smoothing lengths in order to approximate the kernel
        actually used in simulation with a similar kernel with different
        scaling.

    Returns
    -------
    out : WendlandC2Kernel
        An appropriately initialized WendlandC2Kernel object.
    """
    def __init__(self, rescale_sph_h=1):
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel. The formula used
        approximates the kernel amplitude as constant across the pixel area and
        converges to the true value within 1% for smoothing lengths >= 2
        pixels.

        Parameters
        ----------
        dij : astropy.units.Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.

        h : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.

        Returns
        -------
        out : np.array
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
        # .2992 is normalization s.t. kernel integral = 1 for particle mass = 1
        return retval / .2992 / np.power(h, 2)

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        Convergence within 1% of the exact integral is achieved when the
        smoothing lengths are >= 2 pixels.

        Parameters
        ----------
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.x
        """

        if (sm_lengths < 2 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.WendlandC2Kernel.validate:"
                               " SPH smoothing lengths must be >= 2 px in size"
                               " for WendlandC2 kernel integral approximation "
                               "accuracy. This check may be disabled by "
                               "calling Martini.Martini.insert_source_in_cube "
                               "with 'skip_validation=True', but use this with"
                               "care.")
        return

    def size_in_h(self):
        """
        Return the maximum distance where the kernel is non-zero.

        The WendlandC2 kernel is defined such that it reaches 0 at h=1.
        """
        return 1


class CubicSplineKernel(_BaseSPHKernel):
    """
    Implementation of the cubic spline (M4) kernel integral.

    The cubic spline is the 'classic' SPH kernel. The exact integral is usually
    too slow to be practical; the implementation here approximates the kernel
    amplitude as constant across the pixel, which converges to within 1% of
    the exact integral provided the SPH smoothing lengths are at least 2.5
    pixels in size.

    Parameters
    ----------
    rescale_sph_h : float
        Factor by which to rescale SPH smoothing lengths. This can be used to
        adjust particle smoothing lengths in order to approximate the kernel
        actually used in simulation with a similar kernel with different
        scaling.

    Returns
    -------
    out : CubicSplineKernel
        An appropriately initialized CubicSplineKernel object.
    """

    def __init__(self, rescale_sph_h=1):
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel. The formula used
        approximates the kernel amplitude as constant across the pixel area and
        converges to the true value within 1% for smoothing lengths >= 2.5
        pixels.

        Parameters
        ----------
        dij : astropy.units.Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.

        h : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.

        Returns
        -------
        out : np.array
            Approximate kernel integral over the pixel area.
        """

        dr2 = np.power(dij, 2).sum(axis=0)
        retval = np.zeros(h.shape)
        R2 = dr2 / (h * h)
        retval[R2 == 0] = 11. / 16. + .25 * .5
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
        I2 = B_2 - .5 * R2_2 * np.log(2 + B_2) + .5 * R2_2 \
            * np.log(np.sqrt(R2_2))
        I3 = B_1 - .5 * R2_1 * np.log(2 + B_1) - 1.5 * A_1 + .5 * R2_1 \
            * np.log(1 + A_1)
        retval[case1] = I1 + .25 * I3
        retval[case2] = .25 * I2
        # 2.434 is normalization s.t. kernel integral = 1 for particle mass = 1
        return retval / 2.434734306530712 / np.power(h, 2)

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        Convergence within 1% of the exact integral is achieved when the
        smoothing lengths are >= 2.5 pixels.

        Parameters
        ----------
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.x
        """

        if (sm_lengths < 0 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.CubicSplineKernel.validate"
                               ": SPH smoothing lengths must be >= 2.5 px in "
                               "size for cubic spline kernel integral "
                               "approximation accuracy. This check may be "
                               "disabled by calling "
                               "Martini.Martini.insert_source_in_cube with "
                               "'skip_validation=True', but use this with "
                               "care.")
        return

    def size_in_h(self):
        """
        Return the maximum distance where the kernel is non-zero.

        The cubic spline kernel is defined such that it reaches 0 at h=2.
        """
        return 1


class GaussianKernel(_BaseSPHKernel):
    """
    Implementation of a (truncated) Gaussian kernel integral.

    The 3 integrals (along dx, dy, dz) are evaluated exactly, however the
    truncation is implemented approximately in the dx and dy directions. For
    poorly sampled kernels (i.e. large pixels), the normalization is adjusted
    in order to minimize the error.

    Parameters
    ----------
    truncate : float
        Number of standard deviations at which to truncate kernel (default=3).
        Truncation radii <2 may lead to large errors and are not recommended.

    rescale_sph_h : float
        Factor by which to rescale SPH smoothing lengths. This can be used to
        adjust particle smoothing lengths in order to approximate the kernel
        actually used in simulation with a similar kernel with different
        scaling.

    Returns
    -------
    out : GaussianKernel
        An appropriately initialized GaussianKernel object.
    """

    def __init__(self, truncate=3, rescale_sph_h=1):
        self.truncate = truncate
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel. The 3 integrals (along dx,
        dy, dz) are evaluated exactly, however the truncation is implemented
        approximately in the dx and dy directions. For poorly sampled kernels
        (i.e. large pixels), the normalization is adjusted in order to minimize
        the error.

        Parameters
        ----------
        dij : astropy.units.Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.

        h : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.

        Returns
        -------
        out : np.array
            Kernel integral over the pixel area.
        """

        retval = .25 * (
            (erf((dij[0] + .5 * U.pix) / h) - erf((dij[0] - .5 * U.pix) / h)) *
            (erf((dij[1] + .5 * U.pix) / h) - erf((dij[1] - .5 * U.pix) / h))
        )

        # explicit truncation not required as only pixels inside
        # truncation radius should be passed, next 2 lines useful
        # for testing, however
        # dr = np.sqrt(np.power(dij, 2).sum(axis=0))
        # retval[(dr - np.sqrt(.5)) / h > self.truncate] = 0

        # empirically, removing this normalization for poorly sampled kernels
        # leads to increased accuracy
        retval[h > 2.5 * U.pix] = \
            retval[h > 2.5 * U.pix] / np.power(erf(self.truncate), 2)
        return retval * h.unit ** -2

    def validate(self, sm_lengths):

        """
        Check conditions for validity of kernel integral calculation.

        Parameters
        ----------
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """
        if self.truncate < 2:
            raise RuntimeError("Martini.sph_kernels.GaussianKernel.validate: "
                               "truncation radius of <2 may lead to errors in "
                               "mass of >1% per particle (depending on size of"
                               " smoothing kernel relative to pixels). Call "
                               "Martini.Martini.insert_source_in_cube with "
                               "'skip_validation=True' to override, at the "
                               "cost of accuracy.")
        return

    def size_in_h(self):
        """
        Return the maximum distance where the kernel is non-zero.

        The Gaussian kernel is defined such that it reaches 0 at the truncation
        radius.
        """
        return self.truncate


class DiracDeltaKernel(_BaseSPHKernel):
    """
    Implementation of a Dirac-delta kernel integral.

    Parameters
    ----------
    rescale_sph_h : float
        Factor by which to rescale SPH smoothing lengths. This can be used to
        adjust particle smoothing lengths in order to approximate the kernel
        actually used in simulation with a similar kernel with different
        scaling.

    Returns
    -------
    out : DiracDeltaKernel
        An appropriately initialized DiracDeltaKernel object.
    """

    def __init__(self, rescale_sph_h=1):
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel_integral(self, dij, h):
        """
        Calculate the kernel integral over a pixel. The particles are
        approximated as point-like, ignoring any finite-sized kernel.
        This is a reasonable approximation provided the smoothing
        length is < 0.5 pixel in size, ideally << 1 pixel in size.

        Parameters
        ----------
        dij : astropy.units.Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.

        h : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.

        Returns
        -------
        out : np.array
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
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        if (sm_lengths > .5 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.DiracDeltaKernel.validate:"
                               " SPH smoothing lengths must be <= 1 px in size"
                               " for DiracDelta kernel to be a reasonable "
                               "approximation. Call Martini.Martini.insert_"
                               "source_in_cube with 'skip_validation=True' to "
                               "override, at the cost of accuracy.")
        return

    def size_in_h(self):
        """
        Return the maximum distance where the kernel is non-zero.

        In principle the size for a DiracDelta kernel is 0, but this would lead
        to no particles being used. Ideally we would want ~the pixel size here,
        but the sph smoothing length is acceptable.
        """
        return 1
