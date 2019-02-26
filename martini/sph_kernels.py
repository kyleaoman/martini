from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from scipy.special import erf


class _BaseSPHKernel(object):
    """
    Abstract base class for classes implementing SPH kernels to inherit from.

    Classes inheriting from _BaseSPHKernel must implement four methods:
    'kernel', 'kernel_integral', 'validate' and 'size_in_h'.

    'kernel' should define the kernel function, normalized such that its volume
    integral is 1.

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

    @classmethod
    def mimic(cls, other_kernel, **kwargs):
        """
        Approximate a different kernel using this kernel.

        Note that the only class at present which has no restrictions on the
        smoothing lengths relative to the pixel size is the GaussianKernel. In
        cases where other kernels cannot be used due to such restrictions, it
        may be useful to approximate them using the GaussianKernel.

        Parameters
        ----------
        other_kernel : a class or instance inheriting from _BaseSPHKernel
            The kernel to be imitated. If the class is passed, its default
            parameters are used.

        Returns
        -------
        out : an instance of a class inheriting from _BaseSPHKernel
            An appropriately initialized object of the same type as this
            kernel.

        Examples
        --------
        The WendlandC2Kernel implementation loses accuracy when the smoothing
        lengths are small relative to the pixels. A GaussianKernel of similar
        shape can be used instead::

            kernel = GaussianKernel.mimic(WendlandC2Kernel)

        If the kernel to be mimicked must itself be rescaled for use with the
        particle SPH smoothing lengths, it can be rescaled before being
        mimicked. For instance, the CubicSplineKernel implementation here is
        non-zero in the interval [0, 2). There is an equivalent formulation of
        the same kernel which is non-zero in the interval [0, 1). To mimic this
        kernel using a GaussianKernel::

            kernel = GaussianKernel.mimic(CubicSplineKernel(rescale_sph_h=0.5))
        """

        try:
            other_kernel = other_kernel()
        except TypeError:
            pass
        if other_kernel.hscale_to_gaussian is None:
            raise ValueError(
                "{:s} cannot be rescaled, and therefore cannot be mimicked."
                .format(type(other_kernel).__name__)
            )
        if cls.hscale_to_gaussian is None:
            raise ValueError(
                "{:s} cannot be rescaled, and therefore cannot mimic other "
                "kernels.".format(cls.__name__)
            )
        return cls(
            rescale_sph_h=cls.hscale_to_gaussian
            / other_kernel.hscale_to_gaussian
            * other_kernel.rescale_sph_h,
            **kwargs
        )

    def px_weight(self, dij, h):
        """
        Calculate kernel integral using scaled smoothing lengths.

        This is the method that should be called by other modules in
        martini, rather than 'kernel_integral'.

        Parameters
        ----------
        dij : astropy.units.Quantity, with dimensions of pixels
            Distances from pixel centre to particle positions, in pixels.

        h : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in pixels.

        Returns
        -------
        out : astropy.units.Quantity, with dimensions of pixels^-2
            Integral of smoothing kernel over pixel, per unit pixel area.
        """

        return self.kernel_integral(dij, h * self.rescale_sph_h)

    def confirm_validation(self, sm_lengths):
        """
        Verify kernel accuracy using scaled smoothing lengths.

        This is the method that should be called by other modules in
        martini, rather than 'validate'.

        Parameters
        ----------
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        return self.validate(sm_lengths * self.rescale_sph_h)

    def eval_kernel(self, r, h):
        """
        Evaluate the kernel, handling array casting and rescaling. Note that
        if the object was initialized with rescale_sph_h, this will be applied.

        This is the method that should be called by other modules in
        martini, rather than 'kernel'.

        Parameters
        ----------
        r : float or np.array or astropy.units.Quantity
            Distance parameter, same units as h.

        h : float or np.array or astropy.units.Quantity
            Smoothing scale parameter, same units as r.

        Returns
        -------
        out : float or np.array
            Kernel value at position(s) r / h.
        """

        q = np.array(r / h / self.rescale_sph_h)
        if isinstance(q, U.Quantity):
            q = q.to(U.dimensionless_unscaled).value
        scalar_input = q.ndim == 0
        W = self.kernel(q)
        W /= np.power(self.rescale_sph_h, 3)
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
        q : np.array
            Dimensionless distance parameter.

        Returns
        -------
        out : np.array
            Kernel value at positions q.
        """
        pass

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

        Returns
        -------
        out : astropy.units.Quantity, with dimensions of pixels^-2
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
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.
        """

        pass

    @abstractmethod
    def size_in_h(self):
        """
        Abstract method; return the maximum distance where the kernel is non-
        zero, in units of the SPH smoothing parameter of the particles.

        Returns
        -------
        out : float
            Maximum distance where the kernel is non-zero, in units of the SPH
            smoothing parameter.
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

    To obtain a Wendland C2 kernel which resembles a Gaussian kernel with
    sigma = 1, use rescale_sph_h = 2 * sqrt(2 * log(2)). Note that this
    scaling can be used to approximate this kernel using other kernels, or
    vice-versa.

    The WendlandC2 kernel is here defined as (q = r / h):
        W(q) = (21 / pi) * (1 - q)^4 * (4 * q + 1)
        for 0 <= q < 1
        W(q) = 0
        for q >= 1

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

    hscale_to_gaussian = 2 * np.sqrt(2 * np.log(2))

    def __init__(self, rescale_sph_h=1):
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The WendlandC2 kernel is here defined as (q = r / h):
        W(q) = (21 / pi) * (1 - q)^4 * (4 * q + 1)
        for 0 <= q < 1
        W(q) = 0
        for q >= 1

        Parameters
        ----------
        q : np.array
            Dimensionless distance parameter.

        Returns
        -------
        out : np.array
            Kernel value at positions q.
        """

        W = np.where(
            q < 1,
            np.power(1 - q, 4) * (4 * q + 1),
            np.zeros(q.shape)
        )
        W *= (21 / np.pi)
        return W

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

    To obtain a cubic spline kernel which resembles a Gaussian kernel with
    sigma = 1, use rescale_sph_h = sqrt(2 * log(2)). Note that this scaling
    can be used to approximate this kernel using other kernels, or vice-versa.

    The cubic spline kernel is here defined as (q = r / h):
        W(q) = (2 / pi) * (1 - 1.5 * q^2 * (1 - 0.5 * q))
            for 0 <= q < 1
        W(q) = (2 / pi) * (2 - q)^3
            for 1 <= q < 2
        W(q) = 0
            for q >= 2

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

    hscale_to_gaussian = np.sqrt(2 * np.log(2))

    def __init__(self, rescale_sph_h=1):
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The cubic spline kernel is here defined as (q = r / h):
        W(q) = (2 / pi) * (1 - 1.5 * q^2 * (1 - 0.5 * q))
        for 0 <= q < 1
        W(q) = (2 / pi) * (2 - q)^3
        for 1 <= q < 2
        W(q) = 0
        for q >= 2

        Parameters
        ----------
        q : np.array
            Dimensionless distance parameter.

        Returns
        -------
        out : np.array
            Kernel value at positions q.
        """

        W = np.where(
            q < 1,
            1 - 1.5 * np.power(q, 2) + .75 * np.power(q, 3),
            .25 * np.power(2 - q, 3)
        )
        W[q > 2] = 0
        W *= 2 / np.pi
        return W

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
        return retval / 1.59689476201133 / np.power(h, 2)

    def validate(self, sm_lengths):
        """
        Check conditions for validity of kernel integral calculation.

        Convergence within 1% of the exact integral is achieved when the
        smoothing lengths are >= 1.0 pixels.

        Parameters
        ----------
        sm_lengths : astropy.units.Quantity, with dimensions of pixels
            Particle smoothing lengths, in units of pixels.x
        """

        if (sm_lengths < 1.0 * U.pix).any():
            raise RuntimeError("Martini.sph_kernels.CubicSplineKernel.validate"
                               ": SPH smoothing lengths must be >= 1.0 px in "
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

    The Gaussian kernel is here defined as (q = r / h):
    W(q) = (2 / pi^1.5) * np.exp(-q^2)
    for 0 <= q < truncate
    W(q) = 0
    for q >= truncate

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

    hscale_to_gaussian = 1

    def __init__(self, truncate=3, rescale_sph_h=1):
        self.truncate = truncate
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The Gaussian kernel is here defined as (q = r / h):
        W(q) = (2 / pi^1.5) * np.exp(-q^2)
        for 0 <= q < truncate
        W(q) = 0
        for q >= truncate

        Parameters
        ----------
        q : np.array
            Dimensionless distance parameter.

        Returns
        -------
        out : np.array
            Kernel value at positions q.
        """

        return np.where(
            q < self.truncate,
            2 * np.power(np.pi, -1.5) * np.exp(-np.power(q, 2)),
            np.zeros(q.shape)
        )

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

    The Dirac-delta kernel is here defined as (q = r / h)
    W(q) = inf
    for q == 0
    W(q) = 0
    for q != 0

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

    hscale_to_gaussian = None

    def __init__(self, rescale_sph_h=1):
        super().__init__(rescale_sph_h=rescale_sph_h)
        return

    def kernel(self, q):
        """
        Evaluate the kernel function.

        The Dirac-delta kernel is here defined as (q = r / h):
        W(q) = inf
        for q == 0
        W(q) = 0
        for q != 0

        Parameters
        ----------
        q : np.array
            Dimensionless distance parameter.

        Returns
        -------
        out : np.array
            Kernel value at positions q.
        """

        return np.where(q, np.inf * np.ones(q.shape), np.zeros(q.shape))

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
