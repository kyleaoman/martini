import numpy as np
import astropy.units as U
from astropy import wcs
from astropy.coordinates import ICRS
import warnings

HIfreq = 1.420405751e9 * U.Hz


class DataCube(object):
    """
    Handles creation and management of the data cube itself.

    Basic usage simply involves initializing with the parameters listed below.
    More advanced usage might arise if designing custom classes for other sub-
    modules, especially beams. To initialize a :class:`~martini.datacube.DataCube`
    from a saved state, see :meth:`~martini.datacube.DataCube.load_state`.

    Parameters
    ----------
    n_px_x : int, optional
        Pixel count along the x (RA) axis. Even integers strongly preferred.
        (Default: ``256``)

    n_px_y : int, optional
        Pixel count along the y (Dec) axis. Even integers strongly preferred.
        (Default: ``256``)

    n_channels : int, optional
        Number of channels along the spectral axis. (Default: ``64``)

    px_size : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Angular scale of one pixel. (Default: ``15 * U.arcsec``)

    channel_width : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity or frequency.
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency. (Default: ``4 * U.km * U.s**-1``)

    spectral_centre : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity` with dimensions of velocity or frequency.
        Velocity (or frequency) of the centre along the spectral axis.
        (Default: ``0 * U.km * U.s**-1``)

    ra : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Right ascension of the cube centroid. (Default: ``0 * U.deg``)

    dec : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity` with dimensions of angle.
        Declination of the cube centroid. (Default: ``0 * U.deg``)

    stokes_axis : bool, optional
        Whether the datacube should be initialized with a Stokes' axis.
        (Default: ``False``)

    See Also
    --------
    ~martini.datacube.DataCube.load_state
    """

    def __init__(
        self,
        n_px_x=256,
        n_px_y=256,
        n_channels=64,
        px_size=15.0 * U.arcsec,
        channel_width=4.0 * U.km * U.s**-1,
        spectral_centre=0.0 * U.km * U.s**-1,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        stokes_axis=False,
        coordinate_frame=ICRS(),
        specsys="GALACTOC",
    ):
        self.stokes_axis = stokes_axis
        self.coordinate_frame = coordinate_frame
        self.specsys = specsys
        datacube_unit = U.Jy * U.pix**-2
        self._array = np.zeros((n_px_x, n_px_y, n_channels)) * datacube_unit
        if self.stokes_axis:
            self._array = self._array[..., np.newaxis]
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size = px_size
        self.arcsec2_to_pix = (
            U.Jy * U.pix**-2,
            U.Jy * U.arcsec**-2,
            lambda x: x / self.px_size.to_value(U.arcsec) ** 2,
            lambda x: x * self.px_size.to_value(U.arcsec) ** 2,
        )
        if U.get_physical_type(channel_width) == "frequency":
            self._freq_channel_mode = True
        elif U.get_physical_type(channel_width) == "velocity":
            self._freq_channel_mode = False
        else:
            raise ValueError("Channel width must have frequency or velocity units.")
        self.channel_width = np.abs(channel_width)
        self.spectral_centre = spectral_centre.to(
            channel_width.unit, equivalencies=U.doppler_radio(HIfreq)
        )
        self.ra = ra
        self.dec = dec
        self.padx = 0
        self.pady = 0
        self._channel_edges = None
        self._channel_mids = None
        self._wcs = None

        return

    @classmethod
    def from_wcs(cls, input_wcs, specsys=None):

        init_args = dict(
            n_px_x=None,
            n_px_y=None,
            n_channels=None,
            px_size=None,
            channel_width=None,
            spectral_centre=None,
            ra=None,
            dec=None,
            stokes_axis=None,
            coordinate_frame=None,
            specsys=None,
        )
        for axis_type in input_wcs.get_axis_types():
            if axis_type["coordinate_type"] == "stokes":
                init_args["stokes_axis"] = True
        if init_args["stokes_axis"] is None:
            init_args["stokes_axis"] = False
        centre_coords = input_wcs.all_pix2world(
            [[n_px // 2 + (1 + n_px % 2) / 2 for n_px in input_wcs.pixel_shape]],
            1,  # origin, i.e. index pixels from 1
        ).squeeze()
        ax_ra, ax_dec, ax_spec = (
            input_wcs.wcs.lng,
            input_wcs.wcs.lat,
            input_wcs.wcs.spec,
        )
        unit_ra = U.Unit(input_wcs.world_axis_units[ax_ra], format="fits")
        unit_dec = U.Unit(input_wcs.world_axis_units[ax_dec], format="fits")
        unit_spec = U.Unit(input_wcs.world_axis_units[ax_spec], format="fits")
        ra_px_size = np.abs(input_wcs.wcs.cdelt[ax_ra]) * unit_ra
        init_args["n_px_x"] = input_wcs.pixel_shape[ax_ra]
        init_args["ra"] = centre_coords[ax_ra] * unit_ra
        dec_px_size = input_wcs.wcs.cdelt[ax_dec] * unit_dec
        init_args["n_px_y"] = input_wcs.pixel_shape[ax_dec]
        init_args["dec"] = centre_coords[ax_dec] * unit_dec
        init_args["channel_width"] = np.abs(input_wcs.wcs.cdelt[ax_spec]) * unit_spec
        init_args["n_channels"] = input_wcs.pixel_shape[ax_spec]
        init_args["spectral_centre"] = centre_coords[ax_spec] * unit_spec
        if specsys is not None:
            init_args["specsys"] = specsys
            input_wcs.wcs.specsys = specsys
        elif input_wcs.wcs.specsys == "":
            warnings.warn(UserWarning("Input WCS did not specify 'SPECSYS'."))
        else:
            init_args["specsys"] = input_wcs.wcs.specsys

        if ra_px_size != dec_px_size:
            raise ValueError(
                "Martini requires square pixels but input wcs has non-square pixels"
                " (|CDELT| for RA and Dec axes do not match)."
            )
        else:
            init_args["px_size"] = ra_px_size  # == dec_px_size
        datacube = cls(**init_args)
        # order celestial, then spectral, then other (stokes):
        datacube_wcs = input_wcs.reorient_celestial_first()
        if datacube_wcs.wcs.lat == 0:  # RA & Dec swapped
            datacube_wcs = datacube_wcs.swapaxes(0, 1)
        datacube._wcs = datacube_wcs
        datacube._freq_channel_mode = (
            U.get_physical_type(
                U.Unit(
                    datacube_wcs.world_axis_units[datacube_wcs.wcs.spec], format="fits"
                )
            )
            == "frequency"
        )
        return datacube

    @property
    def units(self):
        return [U.Unit(unit, format="fits") for unit in self.wcs.wcs.cunit]

    @property
    def wcs(self):
        """
        The DataCube's World Coordinate System (WCS).
        """
        if self._wcs is None:
            hdr = wcs.utils.celestial_frame_to_wcs(self.coordinate_frame).to_header()
            hdr.update(dict(WCSAXES=3))  # add spectral axis
            hdr.update(
                dict(NAXIS1=self.n_px_x, NAXIS2=self.n_px_y, NAXIS3=self.n_channels)
            )
            self._wcs = wcs.WCS(hdr)
            self._wcs.wcs.ctype = [
                self._wcs.wcs.ctype[0],
                self._wcs.wcs.ctype[1],
                "FREQ" if self._freq_channel_mode else "VRAD",
            ]
            self._wcs.wcs.specsys = self.specsys
            self._wcs.wcs.cunit = [
                self._wcs.wcs.cunit[0],
                self._wcs.wcs.cunit[1],
                (
                    U.Hz.to_string(format="fits")
                    if self._freq_channel_mode
                    else (U.m / U.s).to_string(format="fits")
                ),
            ]
            self._wcs.wcs.crpix = [
                self.n_px_x / 2.0 + 0.5 + self.padx,
                self.n_px_y / 2.0 + 0.5 + self.pady,
                self.n_channels / 2.0 + 0.5,
            ]
            spec_step_sign = 1 if self._freq_channel_mode else -1
            self._wcs.wcs.cdelt = [
                -self.px_size.to_value(self._wcs.wcs.cunit[0]),
                self.px_size.to_value(self._wcs.wcs.cunit[1]),
                spec_step_sign
                * np.abs(self.channel_width.to_value(self._wcs.wcs.cunit[2])),
            ]
            self._wcs.wcs.crval = [
                self.ra.to_value(self._wcs.wcs.cunit[0]),
                self.dec.to_value(self._wcs.wcs.cunit[1]),
                self.spectral_centre.to_value(self._wcs.wcs.cunit[2]),
            ]
            if self.stokes_axis:
                self._wcs = wcs.utils.add_stokes_axis_to_wcs(
                    self._wcs, self._wcs.wcs.naxis
                )
                self._wcs.pixel_shape = (self.n_px_x, self.n_px_y, self.n_channels, 1)
        return self._wcs

    @property
    def channel_mids(self):
        """
        The centres of the channels from the coordinate system.
        """
        if self._channel_mids is None:
            self._channel_mids = (
                self.wcs.sub(("spectral",)).all_pix2world(
                    np.arange(self.n_channels),
                    0,
                )
                * self.units[2]
            ).squeeze()
        return self._channel_mids

    @property
    def channel_edges(self):
        """
        The edges of the channels from the coordinate system.
        """
        if self._channel_edges is None:
            self._channel_edges = (
                self.wcs.sub(("spectral",)).all_pix2world(
                    np.arange(self.n_channels + 1) - 0.5,
                    0,
                )
                * self.units[2]
            ).squeeze()
        return self._channel_edges

    @property
    def velocity_channel_mids(self):
        return self.channel_mids.to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq))

    @property
    def velocity_channel_edges(self):
        return self.channel_edges.to(U.m / U.s, equivalencies=U.doppler_radio(HIfreq))

    @property
    def frequency_channel_mids(self):
        return self.channel_mids.to(U.Hz, equivalencies=U.doppler_radio(HIfreq))

    @property
    def frequency_channel_edges(self):
        return self.channel_edges.to(U.Hz, equivalencies=U.doppler_radio(HIfreq))

    @property
    def _stokes_index(self):
        """
        The position of the Stokes' axis in the WCS axis order.

        Unlike RA (wcs.WCS().wcs.lng), Dec (wcs.WCS().wcs.lat) and spectral axis
        (wcs.WCS().wcs.spec), the Stokes' axis isn't exposed in a convenient way, so
        implement a helper.
        """
        for index, axis_type in enumerate(self.wcs.get_axis_types()):
            if axis_type == "stokes":
                return index
        return None  # not found

    @property
    def spatial_slices(self):
        """
        An iterator over the spatial 'slices' of the cube.
        """
        if self.stokes_axis:
            return iter(
                self._array.squeeze(self._stokes_index).transpose(
                    (self.wcs.wcs.spec, self.wcs.wcs.lng, self.wcs.wcs.lat)
                )
            )
        else:
            return iter(
                self._array.transpose(
                    (self.wcs.wcs.spec, self.wcs.wcs.lng, self.wcs.wcs.lat)
                )
            )

    @property
    def spectra(self):
        """
        An iterator over the spectra (one in each spatial pixel).
        """
        if self.stokes_axis:
            return iter(
                self._array.squeeze(self._stokes_index)
                .transpose((self.wcs.wcs.lng, self.wcs.wcs.lat, self.wcs.wcs.spec))
                .reshape(self.n_px_x * self.n_px_y, self.n_channels)
            )
        else:
            return iter(
                self._array.transpose(
                    (self.wcs.wcs.lng, self.wcs.wcs.lat, self.wcs.wcs.spec)
                ).reshape(self.n_px_x * self.n_px_y, self.n_channels)
            )

    def add_pad(self, pad):
        """
        Resize the cube to add a padding region in the spatial direction.

        Accurate convolution with a beam requires a cube padded according to
        the size of the beam kernel (its representation sampled on a grid with
        the same spacing). The beam class is required to handle defining the
        size of pad required.

        Parameters
        ----------
        pad : tuple
            2-tuple (or other sequence) containing the number of pixels to add in the
            x (RA) and y (Dec) directions.

        See Also
        --------
        ~martini.datacube.DataCube.drop_pad
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
        self._wcs.wcs.crpix = self.wcs.wcs.crpix + np.array(extend_crpix)
        self.padx, self.pady = pad
        return

    def drop_pad(self):
        """
        Remove the padding added using :meth:`~martini.datacube.DataCube.add_pad`.

        After convolution, the pad region contains meaningless information and can be
        discarded.

        See Also
        --------
        ~martini.datacube.DataCube.add_pad
        """

        if (self.padx == 0) and (self.pady == 0):
            return
        self._array = self._array[self.padx : -self.padx, self.pady : -self.pady, ...]
        retract_crpix = [self.padx, self.pady, 0]
        if self.stokes_axis:
            retract_crpix.append(0)
        self._wcs.wcs.crpix = self.wcs.wcs.crpix - np.array(retract_crpix)
        self.padx, self.pady = 0, 0
        return

    def copy(self):
        """
        Produce a copy of the :class:`~martini.datacube.DataCube`.

        May be especially useful to create multiple datacubes with differing intermediate
        steps.

        Returns
        -------
        out : martini.datacube.DataCube
            Copy of the :class:`~martini.datacube.DataCube` object.
        """
        copy = DataCube(
            self.n_px_x,
            self.n_px_y,
            self.n_channels,
            self.px_size,
            self.channel_width,
            self.spectral_centre,
            self.ra,
            self.dec,
        )
        copy.padx, copy.pady = self.padx, self.pady
        copy._wcs = self.wcs.copy()
        copy._freq_channel_mode = self._freq_channel_mode
        copy._channel_edges = self._channel_edges
        copy._channel_mids = self._channel_mids
        copy._array = self._array.copy()
        return copy

    def save_state(self, filename, overwrite=False):
        """
        Write a file from which the current :class:`~martini.datacube.DataCube`
        state can be re-initialized (see :meth:`~martini.datacube.DataCube.load_state`).
        Note that :mod:`h5py` must be installed for use. NOT for outputting mock
        observations, for this see :meth:`~martini.martini.Martini.write_fits` and
        :meth:`~martini.martini.Martini.write_hdf5`.

        Parameters
        ----------
        filename : str
            File to write.

        overwrite : bool
            Whether to allow overwriting existing files. (default: ``False``)

        See Also
        --------
        ~martini.datacube.DataCube.load_state
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
            spectral_centre_unit = self.spectral_centre.unit
            f["_array"].attrs["spectral_centre"] = self.spectral_centre.to_value(
                spectral_centre_unit
            )
            f["_array"].attrs["spectral_centre_unit"] = str(spectral_centre_unit)
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
            f["_array"].attrs["wcs_hdr"] = self.wcs.to_header_string()
        return

    @classmethod
    def load_state(cls, filename):
        """
        Initialize a :class:`~martini.datacube.DataCube` from a state saved using
        :meth:`~martini.datacube.DataCube.save_state`. Note that :mod:`h5py` must be
        installed for use. Note that ONLY the :class:`~martini.datacube.DataCube`
        state is restored, other modules and their configurations are not affected.

        Parameters
        ----------
        filename : str
            File to open.

        Returns
        -------
        out : martini.datacube.DataCube
            A suitably initialized :class:`~martini.datacube.DataCube` object.

        See Also
        --------
        ~martini.datacube.DataCube.save_state
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
            spectral_centre = f["_array"].attrs["spectral_centre"] * U.Unit(
                f["_array"].attrs["spectral_centre_unit"]
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
                spectral_centre=spectral_centre,
                ra=ra,
                dec=dec,
                stokes_axis=stokes_axis,
            )
            D.add_pad((f["_array"].attrs["padx"], f["_array"].attrs["pady"]))
            D._array = f["_array"] * U.Unit(f["_array"].attrs["datacube_unit"])
            # must be after add_pad:
            D._wcs = wcs.WCS(f["_array"].attrs["wcs_hdr"])
        return D

    def __repr__(self):
        """
        Print the contents of the data cube array itself.

        Returns
        -------
        out : str
            Text representation of the :attr:`~martini.datacube.DataCube._array` contents.
        """
        return self._array.__repr__()


class _GlobalProfileDataCube(DataCube):
    """
    Helper class that configures a data cube with a single pixel to hold a spectrum.

    Parameters
    ----------
    n_channels : int, optional
        Number of channels along the spectral axis. (Default: ``64``)

    channel_width : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity` with dimensions of velocity or frequency.
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency. (Default: ``4 U.km * U.s**-1``)

    spectral_centre : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity` with dimensions of velocity or frequency.
        Velocity (or frequency) of the centre along the spectral axis.
        (Default: ``0 * U.km * U.s**-1``)
    """

    def __init__(
        self,
        n_channels=64,
        channel_width=4.0 * U.km * U.s**-1,
        spectral_centre=0.0 * U.km * U.s**-1,
        specsys="GALACTOC",
    ):
        super().__init__(
            n_px_x=1,
            n_px_y=1,
            n_channels=n_channels,
            px_size=1 * U.deg,  # must be >0, ignored for insertion, needed for units
            channel_width=channel_width,
            spectral_centre=spectral_centre,
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            stokes_axis=False,
            coordinate_frame=ICRS(),
            specsys=specsys,
        )

        return
