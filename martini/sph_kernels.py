from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from scipy.special import erf


class _BaseSPHKernel(object):
    """
    Abstract base class for classes implementing SPH kernels to inherit from.

    Classes inheriting from _BaseSPHKernel must implement two methods:
    'px_weight' and 'validate'.

    'px_weight' should define the integral of the kernel over a pixel given the
    distance between the pixel centre and the particle centre, and the
    smoothing length (both in units of pixels). The integral should be
    normalized so that evaluated over the entire kernel it is equal to 1.

    'validate' should check whether any approximations converge to sufficient
    accuracy (for instance, depending on the ratio of the pixel size and
    smoothing length), and raise an error if not.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def px_weight(self, dij, h):
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


class WendlandC2Kernel(_BaseSPHKernel):
    """
    Implementation of the Wendland C2 kernel integral.

    The Wendland C2 kernel is used in the EAGLE code and derivatives (not in
    Gadget/Gadget2!). The exact integral is usually too slow to be practical;
    the implementation here approximates the kernel amplitude as constant
    across the pixel, which converges to within 1% of the exact integral
    provided the SPH smoothing lengths are at least 2 pixels in size.

    Returns
    -------
    out : WendlandC2Kernel
        An appropriately initialized WendlandC2Kernel object.
    """
    def __init__(self):
        super().__init__()
        return

    def px_weight(self, dij, h):
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

    Returns
    -------
    out : GaussianKernel
        An appropriately initialized GaussianKernel object.
    """

    def __init__(self, truncate=3):
        self.truncate = truncate
        super().__init__()

    def px_weight(self, dij, h):
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

        retval = .25 \
            * (erf((dij[0] + .5) / h) - erf((dij[0] - .5) / h)) \
            * (erf((dij[1] + .5) / h) - erf((dij[1] - .5) / h))

        # explicit truncation not required as only pixels inside
        # truncation radius should be passed, next 2 lines useful
        # for testing, however
        # dr = np.sqrt(np.power(dij, 2).sum(axis=0))
        # retval[(dr - np.sqrt(.5)) / h > self.truncate] = 0

        # empirically, removing this normalization for poorly sampled kernels
        # leads to increased accuracy
        retval[h > 2.5] = retval[h > 2.5] / np.power(erf(self.truncate), 2)
        return retval

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


class DiracDeltaKernel(_BaseSPHKernel):
    """
    Implementation of a Dirac-delta kernel integral.

    Returns
    -------
    out : DiracDeltaKernel
        An appropriately initialized DiracDeltaKernel object.
    """

    def __init__(self):
        super().__init__()
        return

    def px_weight(self, dij, h):
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
