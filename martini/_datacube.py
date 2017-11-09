import numpy as np
import astropy.units as U
from astropy import wcs

class DataCube():

    def __init__(
            self, 
            n_px_x = 256, 
            n_px_y = 256, 
            n_channels = 64, 
            px_size = 15. * U.arcsec, 
            channel_width = 4. * U.km * U.s ** -1,
            velocity_centre = 0. * U.km * U.s ** -1
    ):

        datacube_unit = U.Jy * U.pix ** -2
        self._array = np.zeros((n_px_x, n_px_y, n_channels, 1)) * datacube_unit
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size = px_size
        self.channel_width = channel_width
        self.velocity_centre = velocity_centre
        self.pad = 0
        self.wcs = wcs.WCS(naxis=3)
        self.wcs.wcs.crpix = [
            self.n_px_x / 2. + .5, 
            self.n_px_x / 2. + .5, 
            self.n_channels // 2
        ]
        self.units = [U.deg, U.deg, U.m * U.s ** -1]
        self.wcs.wcs.cunit = [str(unit).replace(' ', '') for unit in self.units]
        self.wcs.wcs.cdelt = [
            -self.px_size.to(self.units[0]).value, 
            self.px_size.to(self.units[1]).value, 
            self.channel_width.to(self.units[2]).value
        ]
        self.wcs.wcs.crval = [0, 0, self.velocity_centre.to(self.units[2]).value]
        self.wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'VELO-OBS']
        self.wcs = wcs.utils.add_stokes_axis_to_wcs(self.wcs, self.wcs.wcs.naxis)
        self._channel_mids()
        self._channel_edges()
            
        return

    def _channel_mids(self):
        self.channel_mids = self.wcs.wcs_pix2world(
            np.zeros(self.n_channels), 
            np.zeros(self.n_channels), 
            np.arange(self.n_channels), 
            np.zeros(self.n_channels),
            0
        )[2] * self.units[2]
        return

    def _channel_edges(self):
        self.channel_edges = self.wcs.wcs_pix2world(
            np.zeros(self.n_channels + 1), 
            np.zeros(self.n_channels + 1), 
            np.arange(self.n_channels + 1) - .5, 
            np.zeros(self.n_channels + 1),
            0
        )[2] * self.units[2]
        return

    def spatial_slices(self):
        return iter(self._array[..., 0].transpose((2, 0, 1)))

    def spectra(self):
        return iter(self._array[..., 0].reshape(self.n_px_x * self.n_px_y, self.n_channels))

    def freq_channels(self):
        velocity_widths = np.diff(self.channel_edges)
        HIfreq = 1.420405751E9 * U.Hz
        convert_to_Hz = lambda q: q.to(U.Hz, equivalencies=U.doppler_radio(HIfreq))
        self.wcs.wcs.cdelt[2] = np.abs(convert_to_Hz(self.wcs.wcs.cdelt[2] * self.units[2]) \
                                 - convert_to_Hz(0. * self.units[2])).value
        self.wcs.wcs.crval[2] = convert_to_Hz(self.wcs.wcs.crval[2] * self.units[2]).value
        self.wcs.wcs.ctype[2] = 'FREQ-OBS'
        self.wcs.wcs.cunit[2] = 'Hz'
        self.units[2] = U.Hz
        self._channel_mids()
        self._channel_edges()
        return

    def add_pad(self, pad):
        tmp = self._array
        self._array = np.zeros((self.n_px_x + pad * 2, self.n_px_y + pad * 2, self.n_channels, 1))
        self._array = self._array * tmp.unit
        self._array[pad:-pad, pad:-pad, ...] = tmp
        self.wcs.wcs.crpix += np.array([pad, pad, 0, 0])
        self.pad = pad
        return
        
    def drop_pad(self):
        if self.pad == 0:
            return
        self._array = self._array[self.pad:-self.pad, self.pad:-self.pad, ...]
        self.wcs.wcs.crpix -= np.array([self.pad, self.pad, 0, 0])
        self.pad = 0
        return

    def copy(self):
        copy = DataCube(
            self.n_px_x, 
            self.n_px_y, 
            self.n_channels, 
            self.px_size, 
            self.channel_width,
            self.velocity_centre
        )
        copy.pad = self.pad
        copy.wcs = self.wcs
        copy._array = self._array.copy()
        return copy

    def __repr__(self):
        return self._array.__repr__()
