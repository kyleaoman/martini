import numpy as np
import astropy.units as U
from astropy import wcs

HIfreq = 1.420405751e9 * U.Hz


class DataCube(object):
    """
    Handles creation and management of the data cube itself.

    Basic usage simply involves initializing with the parameters listed below.
    More advanced usage might arise if designing custom classes for other sub-
    modules, especially beams. To initialize a DataCube from a saved state, see
    DataCube.load_state.

    Parameters
    ----------
    n_px_x : int, optional
        Pixel count along the x (RA) axis. Even integers strongly preferred.
        (Default: 256.)

    n_px_y : int, optional
        Pixel count along the y (Dec) axis. Even integers strongly preferred.
        (Default: 256.)

    n_channels : int, optional
        Number of channels along the spectral axis. (Default: 64.)

    px_size : Quantity, with dimensions of angle, optional
        Angular scale of one pixel. (Default: 15 arcsec.)

    channel_width : Quantity, with dimensions of velocity or frequency, optional
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency. (Default: 4 km/s.)

    velocity_centre : Quantity, with dimensions of velocity or frequency, optional
        Velocity (or frequency) of the centre along the spectral axis.
        (Default: 0 km/s.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension of the cube centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination of the cube centroid. (Default: 0 deg.)

    stokes_axis : bool, optional
        Whether the datacube should be initialized with a Stokes' axis. (Default: False.)

    See Also
    --------
    load_state
    """

    def __init__(
        self,
        n_px_x=256,
        n_px_y=256,
        n_channels=64,
        px_size=15.0 * U.arcsec,
        channel_width=4.0 * U.km * U.s**-1,
        velocity_centre=0.0 * U.km * U.s**-1,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        stokes_axis=False,
    ):
        self.stokes_axis = stokes_axis
        datacube_unit = U.Jy * U.pix**-2
        self._array = np.zeros((n_px_x, n_px_y, n_channels)) * datacube_unit
        if self.stokes_axis:
            self._array = self._array[..., np.newaxis]
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size = px_size
        self.arcsec2_to_pix = (
            U.Jy * U.pix**-2,
            U.Jy * U.arcsec**-2,
            lambda x: x / self.px_size**2,
            lambda x: x * self.px_size**2,
        )
        self.velocity_centre = velocity_centre.to(
            U.m / U.s, equivalencies=U.doppler_radio(HIfreq)
        )
        self.channel_width = np.abs(
            (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(HIfreq)
                )
                + 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq))
            - (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(HIfreq)
                )
                - 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq))
        )
        self.ra = ra
        self.dec = dec
        self.padx = 0
        self.pady = 0
        self._freq_channel_mode = False
        self._init_wcs()
        self._channel_mids()
        self._channel_edges()

        return

    def _init_wcs(self):
        self.wcs = wcs.WCS(naxis=3)
        self.wcs.wcs.crpix = [
            self.n_px_x / 2.0 + 0.5,
            self.n_px_y / 2.0 + 0.5,
            self.n_channels / 2.0 + 0.5,
        ]
        self.units = [U.deg, U.deg, U.m / U.s]
        self.wcs.wcs.cunit = [unit.to_string("fits") for unit in self.units]
        self.wcs.wcs.cdelt = [
            -self.px_size.to_value(self.units[0]),
            self.px_size.to_value(self.units[1]),
            self.channel_width.to_value(
                self.units[2], equivalencies=U.doppler_radio(HIfreq)
            ),
        ]
        self.wcs.wcs.crval = [
            self.ra.to_value(self.units[0]),
            self.dec.to_value(self.units[1]),
            self.velocity_centre.to_value(
                self.units[2], equivalencies=U.doppler_radio(HIfreq)
            ),
        ]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
        self.wcs.wcs.specsys = "GALACTOC"
        if self.stokes_axis:
            self.wcs = wcs.utils.add_stokes_axis_to_wcs(self.wcs, self.wcs.wcs.naxis)
        return

    def _channel_mids(self):
        """
        Calculate the centres of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels),
            np.zeros(self.n_channels),
            np.arange(self.n_channels) - 0.5,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels),)
        self.channel_mids = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def _channel_edges(self):
        """
        Calculate the edges of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels + 1),
            np.zeros(self.n_channels + 1),
            np.arange(self.n_channels + 1) - 1,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels + 1),)
        self.channel_edges = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def spatial_slices(self):
        """
        Return an iterator over the spatial 'slices' of the cube.

        Returns
        -------
        out : iterator
            Iterator over the spatial 'slices' of the cube.
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].transpose((2, 0, 1)))

    def spectra(self):
        """
        Return an iterator over the spectra (one in each spatial pixel).

        Returns
        -------
        out : iterator
            Iterator over the spectra (one in each spatial pixel).
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].reshape(self.n_px_x * self.n_px_y, self.n_channels))

    def freq_channels(self):
        """
        Convert spectral axis to frequency units.
        """
        if self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = -np.abs(
            (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(HIfreq))
            - (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.Hz, equivalencies=U.doppler_radio(HIfreq)
        )
        self.wcs.wcs.ctype[2] = "FREQ"
        self.units[2] = U.Hz
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = True
        self._channel_mids()
        self._channel_edges()
        return

    def velocity_channels(self):
        """
        Convert spectral axis to velocity units.
        """
        if not self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = np.abs(
            (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(HIfreq))
            - (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.m / U.s, equivalencies=U.doppler_radio(HIfreq)
        )
        self.wcs.wcs.ctype[2] = "VRAD"
        self.units[2] = U.m * U.s**-1
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = False
        self._channel_mids()
        self._channel_edges()
        return

    def add_pad(self, pad):
        """
        Resize the cube to add a padding region in the spatial direction.

        Accurate convolution with a beam requires a cube padded according to
        the size of the beam kernel (its representation sampled on a grid with
        the same spacing). The beam class is required to handle defining the
        size of pad required.

        Parameters
        ----------
        pad : 2-tuple (or other sequence)
            Number of pixels to add in the x (RA) and y (Dec) directions.

        See Also
        ----------
        drop_pad
        """

        if self.padx > 0 or self.pady > 0:
            raise RuntimeError("Tried to add padding to already padded datacube array.")
        tmp = self._array
        shape = (self.n_px_x + pad[0] * 2, self.n_px_y + pad[1] * 2, self.n_channels)
        if self.stokes_axis:
            shape = shape + (1,)
        self._array = np.zeros(shape)
        self._array = self._array * tmp.unit
        xregion = np.s_[pad[0] : -pad[0]] if pad[0] > 0 else np.s_[:]
        yregion = np.s_[pad[1] : -pad[1]] if pad[1] > 0 else np.s_[:]
        self._array[xregion, yregion, ...] = tmp
        extend_crpix = [pad[0], pad[1], 0]
        if self.stokes_axis:
            extend_crpix.append(0)
        self.wcs.wcs.crpix += np.array(extend_crpix)
        self.padx, self.pady = pad
        return

    def drop_pad(self):
        """
        Remove the padding added using add_pad.

        After convolution, the pad region contains meaningless information and
        can be discarded.

        See Also
        --------
        add_pad
        """

        if (self.padx == 0) and (self.pady == 0):
            return
        self._array = self._array[self.padx : -self.padx, self.pady : -self.pady, ...]
        retract_crpix = [self.padx, self.pady, 0]
        if self.stokes_axis:
            retract_crpix.append(0)
        self.wcs.wcs.crpix -= np.array(retract_crpix)
        self.padx, self.pady = 0, 0
        return

    def copy(self):
        """
        Produce a copy of the DataCube.

        May be especially useful to create multiple datacubes with differing
        intermediate steps.

        Returns
        -------
        out : DataCube
            Copy of the DataCube object.
        """
        in_freq_channel_mode = self._freq_channel_mode
        if in_freq_channel_mode:
            self.velocity_channels()
        copy = DataCube(
            self.n_px_x,
            self.n_px_y,
            self.n_channels,
            self.px_size,
            self.channel_width,
            self.velocity_centre,
            self.ra,
            self.dec,
        )
        copy.padx, copy.pady = self.padx, self.pady
        copy.wcs = self.wcs
        copy._freq_channel_mode = self._freq_channel_mode
        copy.channel_edges = self.channel_edges
        copy.channel_mids = self.channel_mids
        copy._array = self._array.copy()
        return copy

    def save_state(self, filename, overwrite=False):
        """
        Write a file from which the current DataCube state can be
        re-initialized (see DataCube.load_state). Note that h5py must be
        installed for use. NOT for outputting mock observations, for this
        see Martini.write_fits and Martini.write_hdf5.

        Parameters
        ----------
        filename : str
            File to write.

        overwrite : bool
            Whether to allow overwriting existing files (default: False).

        See Also
        --------
        load_state
        """
        import h5py

        mode = "w" if overwrite else "w-"
        with h5py.File(filename, mode=mode) as f:
            array_unit = self._array.unit
            f["_array"] = self._array.to_value(array_unit)
            f["_array"].attrs["datacube_unit"] = str(array_unit)
            f["_array"].attrs["n_px_x"] = self.n_px_x
            f["_array"].attrs["n_px_y"] = self.n_px_y
            f["_array"].attrs["n_channels"] = self.n_channels
            px_size_unit = self.px_size.unit
            f["_array"].attrs["px_size"] = self.px_size.to_value(px_size_unit)
            f["_array"].attrs["px_size_unit"] = str(px_size_unit)
            channel_width_unit = self.channel_width.unit
            f["_array"].attrs["channel_width"] = self.channel_width.to_value(
                channel_width_unit
            )
            f["_array"].attrs["channel_width_unit"] = str(channel_width_unit)
            velocity_centre_unit = self.velocity_centre.unit
            f["_array"].attrs["velocity_centre"] = self.velocity_centre.to_value(
                velocity_centre_unit
            )
            f["_array"].attrs["velocity_centre_unit"] = str(velocity_centre_unit)
            ra_unit = self.ra.unit
            f["_array"].attrs["ra"] = self.ra.to_value(ra_unit)
            f["_array"].attrs["ra_unit"] = str(ra_unit)
            dec_unit = self.dec.unit
            f["_array"].attrs["dec"] = self.dec.to_value(dec_unit)
            f["_array"].attrs["dec_unit"] = str(self.dec.unit)
            f["_array"].attrs["padx"] = self.padx
            f["_array"].attrs["pady"] = self.pady
            f["_array"].attrs["_freq_channel_mode"] = int(self._freq_channel_mode)
            f["_array"].attrs["stokes_axis"] = self.stokes_axis
        return

    @classmethod
    def load_state(cls, filename):
        """
        Initialize a DataCube from a state saved using DataCube.save_state.
        Note that h5py must be installed for use. Note that ONLY the DataCube
        state is restored, other modules and their configurations are not
        affected.

        Parameters
        ----------
        filename : str
            File to open.

        Returns
        -------
        out : martini.DataCube
            A suitably initialized DataCube object.

        See Also
        --------
        save_state
        """
        import h5py

        with h5py.File(filename, mode="r") as f:
            n_px_x = f["_array"].attrs["n_px_x"]
            n_px_y = f["_array"].attrs["n_px_y"]
            n_channels = f["_array"].attrs["n_channels"]
            px_size = f["_array"].attrs["px_size"] * U.Unit(
                f["_array"].attrs["px_size_unit"]
            )
            channel_width = f["_array"].attrs["channel_width"] * U.Unit(
                f["_array"].attrs["channel_width_unit"]
            )
            velocity_centre = f["_array"].attrs["velocity_centre"] * U.Unit(
                f["_array"].attrs["velocity_centre_unit"]
            )
            ra = f["_array"].attrs["ra"] * U.Unit(f["_array"].attrs["ra_unit"])
            dec = f["_array"].attrs["dec"] * U.Unit(f["_array"].attrs["dec_unit"])
            stokes_axis = bool(f["_array"].attrs["stokes_axis"])
            D = cls(
                n_px_x=n_px_x,
                n_px_y=n_px_y,
                n_channels=n_channels,
                px_size=px_size,
                channel_width=channel_width,
                velocity_centre=velocity_centre,
                ra=ra,
                dec=dec,
                stokes_axis=stokes_axis,
            )
            D._init_wcs()
            D.add_pad((f["_array"].attrs["padx"], f["_array"].attrs["pady"]))
            if bool(f["_array"].attrs["_freq_channel_mode"]):
                D.freq_channels()
            D._array = f["_array"] * U.Unit(f["_array"].attrs["datacube_unit"])
        return D

    def __repr__(self):
        """
        Print the contents of the data cube array itself.

        Returns
        -------
        out : str
            Text representation of the DataCube._array contents.
        """
        return self._array.__repr__()
