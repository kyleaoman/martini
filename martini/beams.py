from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as U
from astropy.io import fits
from astropy import wcs
from scipy.interpolate import RectBivariateSpline
import warnings
import os.path

f_HI = 1.420405751 * U.GHz


class _BaseBeam(object):
    """
    Abstract base class for classes implementing a radio telescope beam model.

    Classes inheriting from _BaseBeam must implement three methods: 'f_kernel',
    'kernel_size_px' and 'init_beam_header'.

    'f_kernel' should return a function accepting two arguments, the RA and Dec
    offsets from the beam centroid (provided with units of arcsec), and
    returning the beam amplitude at that location.

    'kernel_px_size' should return a 2-tuple containing the half-size (x, y) of
    the beam image that will be initialized, in pixels.

    'init_beam_header' should be defined if the major/minor axis FWHM of the
    beam and its position angle are not defined when the beam object is
    initialized, for instance if modelling a particular telescope this function
    can be used to set the (constant) parameters of the beam of that particular
    facility.

    Parameters
    ----------
    bmaj : astropy.units.Quantity, with dimensions of angle
        Beam major axis (FWHM) angular size.

    bmin : astropy.units.Quantity, with dimensions of angle
        Beam minor axis (FWHM) angular size.

    bpa : astropy.units.Quantity, with dimesions of angle
        Beam position angle (East from North).

    See Also
    --------
    GaussianBeam
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 bmaj=15. * U.arcsec,
                 bmin=15. * U.arcsec,
                 bpa=0. * U.deg):
        # some beams need information from the datacube; in this case make
        # their call to _BaseBeam.__init__ with bmaj == bmin == bpa == None
        # and define a init_beam_header, to be called after the ra, dec,
        # vel, etc. of the datacube are known
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.px_size = None
        self.kernel = None

        return

    def needs_pad(self):
        """
        Determine the padding of the datacube required by the beam to prevent
        edge effects during convolution.

        Returns
        -------
        2-tuple, each element an integer
        """

        return self.kernel.shape[0] // 2, self.kernel.shape[1] // 2

    def init_kernel(self, datacube):
        """
        Calculate the required size of the beam image

        Parameters
        ----------
        datacube : DataCube instance
            Datacube to use, cube size is required for pixel size, position &
            velocity centroids.
        """

        self.px_size = datacube.px_size
        self.vel = datacube.velocity_centre
        self.ra = datacube.ra
        self.dec = datacube.dec
        if (self.bmaj is None) or (self.bmin is None) or (self.bpa is None):
            self.init_beam_header()
        npx_x, npx_y = self.kernel_size_px()
        px_centres_x = (np.arange(-npx_x, npx_x + 1)) * self.px_size
        px_centres_y = (np.arange(-npx_y, npx_y + 1)) * self.px_size
        self.kernel = self.f_kernel()(*np.meshgrid(px_centres_x, px_centres_y))
        self.arcsec_to_beam = (U.Jy * U.arcsec**-2, U.Jy * U.beam**-1,
                               lambda x: x * (np.pi * self.bmaj * self.bmin),
                               lambda x: x / (np.pi * self.bmaj * self.bmin))

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
        at that position. The offsets are provided as astropy.units.Quantity
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
        parameters are not specified in the call to the __init__ method of the
        derived class.
        """

        pass


class GaussianBeam(_BaseBeam):
    """
    Class implementing a Gaussiam beam model.

    Paramters
    ---------
    bmaj : astropy.units.Quantity, with dimensions of angle
        Beam major axis (FWHM) angular size.

    bmin : astropy.units.Quantity, with dimensions of angle
        Beam minor axis (FWHM) angular size.

    bpa : astropy.units.Quantity, with dimensions of angle
        Beam position angle (East of North).

    truncate : float
        Number of FWHM at which to truncate the beam image.
    """

    def __init__(self,
                 bmaj=15. * U.arcsec,
                 bmin=15. * U.arcsec,
                 bpa=0. * U.deg,
                 truncate=4.):
        self.truncate = truncate
        super().__init__(bmaj=bmaj, bmin=bmin, bpa=bpa)
        return

    def f_kernel(self):
        """
        Returns a function defining the beam amplitude as a function of
        position.

        The model implemented is a 2D Gaussian with FWHM's specified by bmaj
        and bmin and orientation by bpa.

        Returns
        -------
        Callable accepting 2 arguments (both float or array) and returning a
        float or array of corresponding size.
        """

        def fwhm_to_sigma(fwhm):
            return fwhm / (2. * np.sqrt(2. * np.log(2.)))

        sigmamaj = fwhm_to_sigma(self.bmaj)
        sigmamin = fwhm_to_sigma(self.bmin)

        a = np.power(np.cos(self.bpa), 2) / (2. * np.power(sigmamin, 2)) \
            + np.power(np.sin(self.bpa), 2) / (2. * np.power(sigmamaj, 2))
        # signs set for CCW rotation (PA)
        b = -np.sin(2. * self.bpa) / (4 * np.power(sigmamin, 2)) \
            + np.sin(2. * self.bpa) / (4 * np.power(sigmamaj, 2))
        c = np.power(np.sin(self.bpa), 2) / (2. * np.power(sigmamin, 2)) \
            + np.power(np.cos(self.bpa), 2) / (2. * np.power(sigmamaj, 2))
        A = np.power(2. * np.pi * sigmamin * sigmamaj, -1)
        A *= np.power(
            self.px_size, 2
        ).value
        # above causes an extra factor of pixel area, need to track this down
        # properly an see whether correction should apply to all beams, or
        # somewhere else?

        return lambda x, y: A * np.exp(
            -a * np.power(x, 2) - 2. * b * x * y - c * np.power(y, 2)
        )

    def kernel_size_px(self):
        """
        Returns a 2-tuple specifying the half-size of the beam image to be
        initialized, in pixels.

        Returns
        -------
        2-tuple, each element an integer.
        """

        size = np.ceil(
            (self.bmaj * self.truncate)
            .to(U.pix, U.pixel_scale(self.px_size / U.pix))
        ).value + 1
        return size, size


class WSRTBeam(_BaseBeam):
    """
    Class implementing a beam model for the Westerbork Synthesis Radio
    Telescope.

    The beam is adapted from an actual image from the telescope. The image is
    scaled for frequency and declination then interpolated to the required
    resolution. Note that the image is for the "classic" WSRT, not APERTIF.

    The input image is located at:
    '<martini directory>/data/beam00_freq02.fits'
    """

    beamfile = os.path.join(
        os.path.dirname(__file__), 'data/beam00_freq02.fits')

    def __init__(self):
        super().__init__(bmaj=None, bmin=None, bpa=None)

    def _load_beamfile(self):
        """
        Open the input image.

        Returns
        -------
        2-tuple containing an astropy.io.fits.Header and an
        astropy.units.Quantity array.
        """
        with fits.open(self.beamfile) as f:
            return f[0].header, np.transpose(
                f[0].data, axes=(1, 2, 0)) * U.Jy / U.beam

    def _centroid(self):
        """
        Determine the centre in RA, Dec and frequency of the input image.

        Returns
        -------
        3-tuple containing: three astropy.units.Quantity objects with
        dimensions of angle, angle and frequency, respectively, corresponding
        to the RA, Dec and frequency centres.
        """

        bheader, bdata = self._load_beamfile()
        c = wcs.WCS(header=bheader)
        RAgrid, Decgrid, freqgrid = c.wcs_pix2world(
            *np.meshgrid(*tuple([np.arange(N) for N in bdata.shape])), 0)
        RAgrid *= U.deg
        Decgrid *= U.deg
        freqgrid *= U.Hz
        return tuple([
            A[tuple(np.array(bdata.shape) // 2)]
            for A in (RAgrid, Decgrid, freqgrid)
        ])

    def init_beam_header(self):
        """
        Set beam major/minor axis lengths and position angle to WSRT values.
        """

        self.bmaj = 15. * U.arcsec / np.sin(self.dec)
        self.bmin = 15. * U.arcsec
        self.bpa = 90. * U.deg
        return

    def f_kernel(self):
        """
        Returns a function defining the beam amplitude as a function of
        position.

        The model implemented is based on an interpolation of an image of the
        WSRT beam, re-scaled according to declination and frequency.

        Returns
        -------
        Callable accepting 2 arguments (both float or array) and returning a
        float or array of corresponding size.
        """

        if self.dec <= 0. * U.deg:
            raise ValueError('WSRT beam requires positive declination.')
        freq = self.vel.to(U.GHz, equivalencies=U.doppler_radio(f_HI))
        bheader, bdata = self._load_beamfile()
        centroid = self._centroid()
        bpx_ra = np.abs(bheader['CDELT1'] * U.deg).to(U.arcsec)
        bpx_dec = np.abs(bheader['CDELT2'] * U.deg).to(U.arcsec)
        dRAs = np.arange(-(bdata.shape[0] // 2), bdata.shape[0] // 2 + 1) \
            * bpx_ra * (centroid[2] / freq).to(U.dimensionless_unscaled)
        dDecs = np.arange(-(bdata.shape[1] // 2), bdata.shape[1] // 2 + 1) \
            * bpx_dec * np.sin(self.dec) / np.sin(centroid[1]) \
            * (centroid[2] / freq)\
            .to(U.dimensionless_unscaled)
        interpolator = RectBivariateSpline(
            dRAs, dDecs, bdata[..., 0], kx=1, ky=1, s=0)
        # RectBivariateSpline is a wrapper around Fortran code and causes a
        # transpose
        return lambda x, y: interpolator(y, x, grid=False).T

    def kernel_size_px(self):
        """
        Returns a 2-tuple specifying the half-size of the beam image to be
        initialized, in pixels.

        The interpolation may become unstable if the image is severely
        undersampled, pixel sizes of more than 8 arcsec are not recommended.

        Returns
        -------
        2-tuple, each element an integer.
        """

        if self.px_size > 12. * U.arcsec:
            warnings.warn(
                "Using WSRT beam with datacube pixel size >> 8 arcsec,"
                " beam interpolation may fail.")
        freq = self.vel.to(U.GHz, equivalencies=U.doppler_radio(f_HI))
        bheader, bdata = self._load_beamfile()
        centroid = self._centroid()
        aspect_x, aspect_y = np.floor(
            bdata.shape[0] // 2 * np.sin(self.dec)), bdata.shape[1] // 2
        aspect_x = aspect_x * np.abs((bheader['CDELT1'] * U.deg)).to(U.arcsec)\
            * (centroid[2] / freq).to(U.dimensionless_unscaled)
        aspect_y = aspect_y * (bheader['CDELT2'] * U.deg).to(U.arcsec) \
            * (centroid[2] / freq).to(U.dimensionless_unscaled)
        return tuple([
            (a.to(U.pix, U.pixel_scale(self.px_size / U.pix))).value + 1
            for a in (aspect_x, aspect_y)
        ])
