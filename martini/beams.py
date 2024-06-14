from abc import ABCMeta, abstractmethod
import scipy.interpolate
import numpy as np
import astropy.units as U


class _BaseBeam(object):
    """
    Abstract base class for classes implementing a radio telescope beam model.

    Classes inheriting from _BaseBeam must implement three methods:
    :meth:`~martini.beams._BaseBeam.f_kernel`,
    :meth:`~martini.beams._BaseBeam.kernel_size_px` and
    :meth:`~martini.beams._BaseBeam.init_beam_header`.

    :meth:`~martini.beams._BaseBeam.f_kernel` should return a function accepting
    two arguments, the RA and Dec offsets from the beam centroid (provided with
    units of arcsec), and returning the beam amplitude at that location.

    :meth:`~martini.beams._BaseBeam.kernel_px_size` should return a 2-tuple
    containing the half-size (x, y) of the beam image that will be initialized,
    in pixels.

    :meth:`~martini.beams._BaseBeam.init_beam_header` should be defined if the
    major/minor axis FWHM of the beam and its position angle are not defined when
    the beam object is initialized, for instance if modelling a particular telescope
    this function can be used to set the (constant) parameters of the beam of that
    particular facility.

    Parameters
    ----------
    bmaj : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Beam major axis (FWHM) angular size.

    bmin : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Beam minor axis (FWHM) angular size.

    bpa : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimesions of angle.
        Beam position angle (East from North).

    See Also
    --------
    ~martini.beams.GaussianBeam
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        bmaj=15.0 * U.arcsec,
        bmin=15.0 * U.arcsec,
        bpa=0.0 * U.deg,
    ):
        # some beams need information from the datacube; in this case make
        # their call to _BaseBeam.__init__ with bmaj == bmin == bpa == None
        # and define a init_beam_header, to be called after the ra, dec,
        # vel, etc. of the datacube are known
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.px_size = None
        self.kernel = None

        # since bmaj, bmin are FWHM, need to include conversion to
        # gaussian-equivalent width (2sqrt(2log2)sigma = FWHM), and then
        # A = 2pi * sigma_maj * sigma_min = pi * b_maj * b_min / 4 / log2
        self.area = (np.pi * self.bmaj * self.bmin) / 4 / np.log(2)

        return

    def needs_pad(self):
        """
        Determine the padding of the datacube required by the beam to prevent
        edge effects during convolution.

        Returns
        -------
        out : tuple
            2-tuple, each element an integer, containing pad dimensions (x, y).
        """

        if self.kernel is None:
            raise RuntimeError("Beam kernel not initialized.")
        return self.kernel.shape[0] // 2, self.kernel.shape[1] // 2

    def init_kernel(self, datacube):
        """
        Calculate the required size of the beam image.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            Data cube to use, cube size is required for pixel size, position &
            velocity centroids.
        """

        self.px_size = datacube.px_size
        self.vel = datacube.spectral_centre
        self.ra = datacube.ra
        self.dec = datacube.dec
        if (self.bmaj is None) or (self.bmin is None) or (self.bpa is None):
            self.init_beam_header()
        npx_x, npx_y = self.kernel_size_px()
        if self.px_size is not None:
            px_size_unit = self.px_size.unit
        else:
            raise RuntimeError("Beam pixel size not initialized.")
        px_edges_x = np.arange(-npx_x - 0.5, npx_x + 0.50001, 1) * self.px_size
        px_edges_y = np.arange(-npx_y - 0.5, npx_y + 0.50001, 1) * self.px_size
        # Elliptical Gaussian has no analytic surface integral in cartesian coordinates
        # and other beam shapes presumably much worse, so let's make a spline interpolator
        # based on a fine sampling of the beam function and integrate that.
        fine_sample_x = np.arange(-npx_x - 0.5, npx_x + 0.501, 0.1) * self.px_size
        fine_sample_y = np.arange(-npx_y - 0.5, npx_y + 0.501, 0.1) * self.px_size
        # set meshgrid indexing to avoid unintentional transpose
        rbs = scipy.interpolate.RectBivariateSpline(
            fine_sample_x.to_value(px_size_unit),
            fine_sample_y.to_value(px_size_unit),
            self.f_kernel()(
                *np.meshgrid(fine_sample_x, fine_sample_y, indexing="ij")
            ).to_value(px_size_unit**-2),
            kx=3,
            ky=3,
        )
        # rbs.integral only evaluates a point at a time, resort to np.vectorize
        xgrid, ygrid = np.meshgrid(
            px_edges_x.to_value(px_size_unit), px_edges_y.to_value(px_size_unit)
        )
        # f_kernel is in px_size_unit ** -2, xgrid and ygrid are in px_size_unit so 2D
        # integral is dimensionless:
        self.kernel = (
            np.vectorize(rbs.integral)(
                xgrid[1:, :-1], xgrid[1:, 1:], ygrid[:-1, 1:], ygrid[1:, 1:]
            )
            * U.dimensionless_unscaled
        )

        # can turn 2D beam into a 3D beam here; use above for central channel
        # then shift in frequency up and down for other channels
        # then probably need to adjust convolution step to do the 2D
        # convolution on a stack

        return

    @abstractmethod
    def f_kernel(self):
        """
        Abstract method; returns a function defining the beam amplitude as a
        function of position.

        The function returned by this method should accept two parameters, the
        RA and Dec offset from the beam centroid, and return the beam amplitude
        at that position. The offsets are provided as :class:`~astropy.units.Quantity`
        objects with dimensions of angle (arcsec).
        """

        pass

    @abstractmethod
    def kernel_size_px(self):
        """
        Abstract method; returns a 2-tuple specifying the half-size of the beam
        image to be initialized, in pixels.
        """

        pass

    @abstractmethod
    def init_beam_header(self):
        """
        Abstract method; sets beam major/minor axis lengths and position angle.

        This method is optional, and only needs to be defined if these
        parameters are not specified in the call to the ``__init__`` method of the
        derived class.
        """

        pass


class GaussianBeam(_BaseBeam):
    """
    Class implementing a Gaussian beam model.

    Parameters
    ----------
    bmaj : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Beam major axis (FWHM) angular size.

    bmin : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Beam minor axis (FWHM) angular size.

    bpa : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Beam position angle (East of North).

    truncate : float
        Number of FWHM at which to truncate the beam image.
    """

    def __init__(
        self,
        bmaj=15.0 * U.arcsec,
        bmin=15.0 * U.arcsec,
        bpa=0.0 * U.deg,
        truncate=4.0,
    ):
        self.truncate = truncate
        super().__init__(bmaj=bmaj, bmin=bmin, bpa=bpa)
        return

    def f_kernel(
        self,
    ):
        """
        Returns a function defining the beam amplitude as a function of
        position.

        The model implemented is a 2D Gaussian with FWHM's specified by ``bmaj``
        and ``bmin`` and orientation by ``bpa``.

        Returns
        -------
        out : callable
            Accepts 2 arguments (both ``ArrayLike``) and return an
            ``ArrayLike`` of corresponding size.
        """

        def fwhm_to_sigma(fwhm):
            return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        sigmamaj = fwhm_to_sigma(self.bmaj)  # arcsec
        sigmamin = fwhm_to_sigma(self.bmin)  # arcsec

        a = np.power(np.cos(self.bpa), 2) / (2.0 * np.power(sigmamin, 2)) + np.power(
            np.sin(self.bpa), 2
        ) / (2.0 * np.power(sigmamaj, 2))
        # signs set for CCW rotation (PA)
        b = -np.sin(2.0 * self.bpa) / (4 * np.power(sigmamin, 2)) + np.sin(
            2.0 * self.bpa
        ) / (4 * np.power(sigmamaj, 2))
        c = np.power(np.sin(self.bpa), 2) / (2.0 * np.power(sigmamin, 2)) + np.power(
            np.cos(self.bpa), 2
        ) / (2.0 * np.power(sigmamaj, 2))
        A = np.power(2.0 * np.pi * sigmamin * sigmamaj, -1)  # arcsec^-2

        return lambda x, y: A * np.exp(
            -a * np.power(x, 2) - 2.0 * b * x * y - c * np.power(y, 2)
        )

    def kernel_size_px(self):
        """
        Returns a 2-tuple specifying the half-size of the beam image to be
        initialized, in pixels.

        Returns
        -------
        out : tuple
            2-tuple, each element an integer, specifying the kernel size (x, y) in pixels.
        """
        size = np.ceil(
            (self.bmaj * self.truncate).to_value(
                U.pix, U.pixel_scale(self.px_size / U.pix)
            )
            + 1
        )

        return size, size
