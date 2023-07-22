import subprocess
import os
import tqdm
from scipy.signal import fftconvolve
import numpy as np
import astropy.units as U
from astropy.io import fits
from astropy.time import Time
from astropy import __version__ as astropy_version
from itertools import product
from .__version__ import __version__ as martini_version
from warnings import warn
from martini.datacube import DataCube

try:
    gc = subprocess.check_output(
        ["git", "describe", "--always"],
        stderr=open(os.devnull, "w"),
        cwd=os.path.dirname(os.path.realpath(__file__)),
    )
except (subprocess.CalledProcessError, FileNotFoundError):
    gc = b""
else:
    martini_version = martini_version + "_commit_" + gc.strip().decode()


class Martini:
    """
    Creates synthetic HI data cubes from simulation data.

    Usual use of martini involves first creating instances of classes from each
    of the required and optional sub-modules, then creating a Martini with
    these instances as arguments. The object can then be used to create
    synthetic observations, usually by calling `insert_source_in_cube`,
    (optionally) `add_noise`, (optionally) `convolve_beam` and `write_fits` in
    order.

    Parameters
    ----------
    source : an instance of a class derived from martini.source._BaseSource
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). Sources leveraging the simobj package for reading
        simulation data (github.com/kyleaoman/simobj) and a few test sources
        (e.g. single particle) are provided, creation of customized sources,
        for instance to leverage other interfaces to simulation data, is
        straightforward. See sub-module documentation.

    datacube : martini.DataCube instance
        A description of the datacube to create, including pixels, channels,
        sky position. See sub-module documentation.

    beam : an instance of a class derived from beams._BaseBeam, optional
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See sub-module documentation.

    noise : an instance of a class derived from noise._BaseNoise, optional
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        sub-module documentation.

    sph_kernel : an instance of a class derived from sph_kernels._BaseSPHKernel
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        SPH kernel submodule documentation for guidance.

    spectral_model : an instance of a class derived from \
    spectral_models._BaseSpectrum
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See sub-module documentation.

    quiet : bool
        If True, suppress output to stdout. (Default: False)

    See Also
    --------
    martini.sources
    martini.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models

    Examples
    --------
    More detailed examples can be found in the examples directory in the github
    distribution of the package.

    The following example illustrates basic use of martini, using a (very!)
    crude model of a gas disk. This example can be run by doing
    'from martini import demo; demo()'::

        # ------make a toy galaxy----------
        N = 500
        phi = np.random.rand(N) * 2 * np.pi
        r = []
        for L in np.random.rand(N):

            def f(r):
                return L - 0.5 * (2 - np.exp(-r) * (np.power(r, 2) + 2 * r + 2))

            r.append(fsolve(f, 1.0)[0])
        r = np.array(r)
        # exponential disk
        r *= 3 / np.sort(r)[N // 2]
        z = -np.log(np.random.rand(N))
        # exponential scale height
        z *= 0.5 / np.sort(z)[N // 2] * np.sign(np.random.rand(N) - 0.5)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz_g = np.vstack((x, y, z)) * U.kpc
        # linear rotation curve
        vphi = 100 * r / 6.0
        vx = -vphi * np.sin(phi)
        vy = vphi * np.cos(phi)
        # small pure random z velocities
        vz = (np.random.rand(N) * 2.0 - 1.0) * 5
        vxyz_g = np.vstack((vx, vy, vz)) * U.km * U.s**-1
        T_g = np.ones(N) * 8e3 * U.K
        mHI_g = np.ones(N) / N * 5.0e9 * U.Msun
        # ~mean interparticle spacing smoothing
        hsm_g = np.ones(N) * 4 / np.sqrt(N) * U.kpc
        # ---------------------------------

        source = SPHSource(
            distance=3.0 * U.Mpc,
            rotation={"L_coords": (60.0 * U.deg, 0.0 * U.deg)},
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            h=0.7,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )

        datacube = DataCube(
            n_px_x=128,
            n_px_y=128,
            n_channels=32,
            px_size=10.0 * U.arcsec,
            channel_width=10.0 * U.km * U.s**-1,
            velocity_centre=source.vsys,
        )

        beam = GaussianBeam(
            bmaj=30.0 * U.arcsec, bmin=30.0 * U.arcsec, bpa=0.0 * U.deg, truncate=4.0
        )

        noise = GaussianNoise(rms=3.0e-5 * U.Jy * U.beam**-1)

        spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

        sph_kernel = CubicSplineKernel()

        M = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
        )

        M.insert_source_in_cube()
        M.add_noise()
        M.convolve_beam()
        M.write_beam_fits(beamfile, channels="velocity")
        M.write_fits(cubefile, channels="velocity")
        print(f"Wrote demo fits output to {cubefile}, and beam image to {beamfile}.")
        try:
            M.write_hdf5(hdf5file, channels="velocity")
        except ModuleNotFoundError:
            print("h5py package not present, skipping hdf5 output demo.")
        else:
            print(f"Wrote demo hdf5 output to {hdf5file}.")
    """

    def __init__(
        self,
        source=None,
        datacube=None,
        beam=None,
        noise=None,
        sph_kernel=None,
        spectral_model=None,
        quiet=False,
    ):
        self.quiet = quiet
        if source is not None:
            self.source = source
        else:
            raise ValueError("A source instance is required.")
        if datacube is not None:
            self.datacube = datacube
        else:
            raise ValueError("A datacube instance is required.")
        self.beam = beam
        self.noise = noise
        if sph_kernel is not None:
            self.sph_kernel = sph_kernel
        else:
            raise ValueError("An SPH kernel instance is required.")
        if spectral_model is not None:
            self.spectral_model = spectral_model
        else:
            raise ValueError("A spectral model instance is required.")

        if self.beam is not None:
            self.beam.init_kernel(self.datacube)
            self.datacube.add_pad(self.beam.needs_pad())

        self.source._init_skycoords()
        self.source._init_pixcoords(self.datacube)  # after datacube is padded

        self.sph_kernel._init_sm_lengths(source=self.source, datacube=self.datacube)
        self.sph_kernel._init_sm_ranges()
        self._prune_particles()  # prunes both source, and kernel if applicable

        self.spectral_model.init_spectra(self.source, self.datacube)

        return

    def convolve_beam(self):
        """
        Convolve the beam and DataCube.
        """

        if self.beam is None:
            warn("Skipping beam convolution, no beam object provided to " "Martini.")
            return

        unit = self.datacube._array.unit
        for spatial_slice in self.datacube.spatial_slices():
            # use a view [...] to force in-place modification
            spatial_slice[...] = (
                fftconvolve(spatial_slice, self.beam.kernel, mode="same") * unit
            )
        self.datacube.drop_pad()
        self.datacube._array = self.datacube._array.to(
            U.Jy * U.beam**-1,
            equivalencies=U.beam_angular_area(self.beam.area),
        )
        if not self.quiet:
            print(
                "Beam convolved.",
                "  Data cube RMS after beam convolution:"
                f" {np.std(self.datacube._array):.2e}",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def add_noise(self):
        """
        Insert noise into the DataCube.
        """

        if self.noise is None:
            warn("Skipping noise, no noise object provided to Martini.")
            return

        # this unit conversion means noise can be added before or after source insertion:
        noise_cube = (
            self.noise.generate(self.datacube, self.beam)
            .to(
                U.Jy * U.arcsec**-2,
                equivalencies=U.beam_angular_area(self.beam.area),
            )
            .to(self.datacube._array.unit, equivalencies=[self.datacube.arcsec2_to_pix])
        )
        self.datacube._array = self.datacube._array + noise_cube
        if not self.quiet:
            print(
                "Noise added.",
                f"  Noise cube RMS: {np.std(noise_cube):.2e} (before beam convolution).",
                "  Data cube RMS after noise addition (before beam convolution): "
                f"{np.std(self.datacube._array):.2e}",
                sep="\n",
            )
        return

    def _prune_particles(self):
        """
        Determines which particles cannot contribute to the DataCube and
        removes them to speed up calculation. Assumes the kernel is 0 at
        distances greater than the kernel size (which may differ from the
        SPH smoothing length).
        """

        if not self.quiet:
            print(
                f"Source module contained {self.source.npart} particles with total HI"
                f" mass of {self.source.mHI_g.sum():.2e}."
            )
        spectrum_half_width = (
            self.spectral_model.half_width(self.source) / self.datacube.channel_width
        )
        reject_conditions = (
            (
                self.source.pixcoords[:2] + self.sph_kernel.sm_ranges[np.newaxis]
                < 0 * U.pix
            ).any(axis=0),
            self.source.pixcoords[0] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            self.source.pixcoords[1] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            self.source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            self.source.pixcoords[2] - 4 * spectrum_half_width * U.pix
            > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(self.source.pixcoords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        self.source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        self.sph_kernel.apply_mask(np.logical_not(reject_mask))
        if not self.quiet:
            print(
                f"Pruned particles that will not contribute to data cube, "
                f"{self.source.npart} particles remaining with total HI mass of "
                f"{self.source.mHI_g.sum():.2e}."
            )
        return

    def insert_source_in_cube(self, skip_validation=False, progressbar=True):
        """
        Populates the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the DataCube is approximated for
            increased speed. For some combinations of pixel size, distance
            and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            RuntimeError if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter True. (Default: False.)

        progressbar : bool, optional
            If True, a progress bar will be shown. (Default: True.)
        """

        assert self.spectral_model.spectra is not None

        self.sph_kernel.confirm_validation(noraise=skip_validation, quiet=self.quiet)

        # pixel iteration
        ij_pxs = list(
            product(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
            )
        )
        if not self.quiet:
            print("Inserting source in cube.")
        if progressbar:
            if self.quiet:
                print(
                    "To silence progress bar, set"
                    " insert_source_in_cube(progressbar=False)"
                )
            ij_pxs = tqdm.tqdm(ij_pxs)
        for ij_px in ij_pxs:
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            mask = (
                np.abs(ij - self.source.pixcoords[:2]) <= self.sph_kernel.sm_ranges
            ).all(axis=0)
            weights = self.sph_kernel.px_weight(
                self.source.pixcoords[:2, mask] - ij, mask=mask
            )
            insertion_slice = (
                np.s_[ij_px[0], ij_px[1], :, 0]
                if self.datacube.stokes_axis
                else np.s_[ij_px[0], ij_px[1], :]
            )
            self.datacube._array[insertion_slice] = (
                self.spectral_model.spectra[mask] * weights[..., np.newaxis]
            ).sum(axis=-2)

        self.datacube._array = self.datacube._array.to(
            U.Jy / U.arcsec**2, equivalencies=[self.datacube.arcsec2_to_pix]
        )
        pad_mask = (
            np.s_[
                self.datacube.padx : -self.datacube.padx,
                self.datacube.pady : -self.datacube.pady,
                ...,
            ]
            if self.datacube.padx > 0 and self.datacube.pady > 0
            else np.s_[...]
        )
        inserted_flux = (
            self.datacube._array[pad_mask].sum() * self.datacube.px_size**2
        )
        inserted_mass = (
            2.36e5
            * U.Msun
            * self.source.distance.to_value(U.Mpc) ** 2
            * inserted_flux.to_value(U.Jy)
            * self.datacube.channel_width.to_value(U.km / U.s)
        )
        if not self.quiet:
            print(
                "Source inserted.",
                f"  Flux in cube: {inserted_flux:.2e}",
                f"  Mass in cube (assuming distance {self.source.distance:.2f}):"
                f" {inserted_mass:.2e}",
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def write_fits(
        self,
        filename,
        channels="frequency",
        overwrite=True,
    ):
        """
        Output the DataCube to a FITS-format file.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)
        """

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()
        wcs_header.rename_keyword("WCSAXES", "NAXIS")

        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", self.datacube.n_px_x))
        header.append(("NAXIS2", self.datacube.n_px_y))
        header.append(("NAXIS3", self.datacube.n_channels))
        if self.datacube.stokes_axis:
            header.append(("NAXIS4", 1))
        header.append(("EXTEND", "T"))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRPIX1", wcs_header["CRPIX1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRPIX2", wcs_header["CRPIX2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CDELT3", wcs_header["CDELT3"]))
        header.append(("CRPIX3", wcs_header["CRPIX3"]))
        header.append(("CRVAL3", wcs_header["CRVAL3"]))
        header.append(("CTYPE3", wcs_header["CTYPE3"]))
        header.append(("CUNIT3", wcs_header["CUNIT3"]))
        if self.datacube.stokes_axis:
            header.append(("CDELT4", wcs_header["CDELT4"]))
            header.append(("CRPIX4", wcs_header["CRPIX4"]))
            header.append(("CRVAL4", wcs_header["CRVAL4"]))
            header.append(("CTYPE4", wcs_header["CTYPE4"]))
            header.append(("CUNIT4", "PAR"))
        header.append(("EPOCH", 2000))
        header.append(("INSTRUME", "MARTINI", martini_version))
        # header.append(('BLANK', -32768)) #only for integer data
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        datacube_array_units = self.datacube._array.unit
        header.append(
            ("DATAMAX", np.max(self.datacube._array.to_value(datacube_array_units)))
        )
        header.append(
            ("DATAMIN", np.min(self.datacube._array.to_value(datacube_array_units)))
        )
        header.append(("ORIGIN", "astropy v" + astropy_version))
        # long names break fits format, don't let the user set this
        header.append(("OBJECT", "MOCK"))
        if self.beam is not None:
            header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("OBSERVER", "K. Oman"))
        # header.append(('NITERS', ???))
        # header.append(('RMS', ???))
        # header.append(('LWIDTH', ???))
        # header.append(('LSTEP', ???))
        header.append(("BUNIT", datacube_array_units.to_string("fits")))
        # header.append(('PCDEC', ???))
        # header.append(('LSTART', ???))
        header.append(("DATE-OBS", Time.now().to_value("fits")))
        # header.append(('LTYPE', ???))
        # header.append(('PCRA', ???))
        # header.append(('CELLSCAL', ???))
        if self.beam is not None:
            header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
            header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header, data=self.datacube._array.to_value(datacube_array_units).T
        )
        hdu.writeto(filename, overwrite=overwrite)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_beam_fits(self, filename, channels="frequency", overwrite=True):
        """
        Output the beam to a FITS-format file.

        The beam is written to file, with pixel sizes, coordinate system, etc.
        similar to those used for the DataCube.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        Raises
        ------
        ValueError
            If Martini was initialized without a beam.
        """

        if self.beam is None:
            raise ValueError(
                "Martini.write_beam_fits: Called with beam set " "to 'None'."
            )
        assert self.beam.kernel is not None
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_beam_fits: Unknown 'channels' "
                "value (use 'frequency' or 'velocity'."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()

        beam_kernel_units = self.beam.kernel.unit
        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        # header.append(('NAXIS', self.beam.kernel.ndim))
        header.append(("NAXIS", 3))
        header.append(("NAXIS1", self.beam.kernel.shape[0]))
        header.append(("NAXIS2", self.beam.kernel.shape[1]))
        header.append(("NAXIS3", 1))
        header.append(("EXTEND", "T"))
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        # this is 1/arcsec^2, is this right?
        header.append(("BUNIT", beam_kernel_units.to_string("fits")))
        header.append(("CRPIX1", self.beam.kernel.shape[0] // 2 + 1))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CRPIX2", self.beam.kernel.shape[1] // 2 + 1))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CRPIX3", 1))
        header.append(("CDELT3", wcs_header["CDELT3"]))
        header.append(("CRVAL3", wcs_header["CRVAL3"]))
        header.append(("CTYPE3", wcs_header["CTYPE3"]))
        header.append(("CUNIT3", wcs_header["CUNIT3"]))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))
        header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
        header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("BTYPE", "beam    "))
        header.append(("EPOCH", 2000))
        header.append(("OBSERVER", "K. Oman"))
        # long names break fits format
        header.append(("OBJECT", "MOCKBEAM"))
        header.append(("INSTRUME", "MARTINI", martini_version))
        header.append(("DATAMAX", np.max(self.beam.kernel.to_value(beam_kernel_units))))
        header.append(("DATAMIN", np.min(self.beam.kernel.to_value(beam_kernel_units))))
        header.append(("ORIGIN", "astropy v" + astropy_version))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header,
            data=self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis].T,
        )
        hdu.writeto(filename, overwrite=True)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_hdf5(
        self,
        filename,
        channels="frequency",
        overwrite=True,
        memmap=False,
        compact=False,
    ):
        """
        Output the DataCube and Beam to a HDF5-format file. Requires the h5py
        package.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.hdf5' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        memmap: bool, optional
            If True, create a file-like object in memory and return it instead
            of writing file to disk. (Default: False.)

        compact: bool, optional
            If True, omit pixel coordinate arrays to save disk space. In this
            case pixel coordinates can still be reconstructed from FITS-style
            keywords stored in the FluxCube attributes. (Default: False.)
        """

        import h5py

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            pass
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".hdf5" else filename + ".hdf5"

        wcs_header = self.datacube.wcs.to_header()

        mode = "w" if overwrite else "x"
        driver = "core" if memmap else None
        h5_kwargs = {"backing_store": False} if memmap else dict()
        f = h5py.File(filename, mode, driver=driver, **h5_kwargs)
        datacube_array_units = self.datacube._array.unit
        s = np.s_[..., 0] if self.datacube.stokes_axis else np.s_[...]
        f["FluxCube"] = self.datacube._array.to_value(datacube_array_units)[s]
        c = f["FluxCube"]
        origin = 0  # index from 0 like numpy, not from 1
        if not compact:
            xgrid, ygrid, vgrid = np.meshgrid(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
                np.arange(self.datacube._array.shape[2]),
            )
            cgrid = (
                np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                        np.zeros(vgrid.shape).flatten(),
                    )
                ).T
                if self.datacube.stokes_axis
                else np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                    )
                ).T
            )
            wgrid = self.datacube.wcs.all_pix2world(cgrid, origin)
            ragrid = wgrid[:, 0].reshape(self.datacube._array.shape)[s]
            decgrid = wgrid[:, 1].reshape(self.datacube._array.shape)[s]
            chgrid = wgrid[:, 2].reshape(self.datacube._array.shape)[s]
            f["RA"] = ragrid
            f["RA"].attrs["Unit"] = wcs_header["CUNIT1"]
            f["Dec"] = decgrid
            f["Dec"].attrs["Unit"] = wcs_header["CUNIT2"]
            f["channel_mids"] = chgrid
            f["channel_mids"].attrs["Unit"] = wcs_header["CUNIT3"]
        c.attrs["AxisOrder"] = "(RA,Dec,Channels)"
        c.attrs["FluxCubeUnit"] = str(self.datacube._array.unit)
        c.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
        c.attrs["RA0_in_px"] = wcs_header["CRPIX1"] - 1
        c.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
        c.attrs["RAUnit"] = wcs_header["CUNIT1"]
        c.attrs["RAProjType"] = wcs_header["CTYPE1"]
        c.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
        c.attrs["Dec0_in_px"] = wcs_header["CRPIX2"] - 1
        c.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
        c.attrs["DecUnit"] = wcs_header["CUNIT2"]
        c.attrs["DecProjType"] = wcs_header["CTYPE2"]
        c.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
        c.attrs["V0_in_px"] = wcs_header["CRPIX3"] - 1
        c.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
        c.attrs["VUnit"] = wcs_header["CUNIT3"]
        c.attrs["VProjType"] = wcs_header["CTYPE3"]
        if self.beam is not None:
            c.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            c.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            c.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
        c.attrs["DateCreated"] = str(Time.now())
        c.attrs["MartiniVersion"] = martini_version
        c.attrs["AstropyVersion"] = astropy_version
        if self.beam is not None:
            if self.beam.kernel is None:
                raise ValueError(
                    "Martini.write_hdf5: Called with beam present but beam kernel"
                    " uninitialized."
                )
            beam_kernel_units = self.beam.kernel.unit
            f["Beam"] = self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis]
            b = f["Beam"]
            b.attrs["BeamUnit"] = self.beam.kernel.unit.to_string("fits")
            b.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
            b.attrs["RA0_in_px"] = self.beam.kernel.shape[0] // 2
            b.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
            b.attrs["RAUnit"] = wcs_header["CUNIT1"]
            b.attrs["RAProjType"] = wcs_header["CTYPE1"]
            b.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
            b.attrs["Dec0_in_px"] = self.beam.kernel.shape[1] // 2
            b.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
            b.attrs["DecUnit"] = wcs_header["CUNIT2"]
            b.attrs["DecProjType"] = wcs_header["CTYPE2"]
            b.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
            b.attrs["V0_in_px"] = 0
            b.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
            b.attrs["VUnit"] = wcs_header["CUNIT3"]
            b.attrs["VProjType"] = wcs_header["CTYPE3"]
            b.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            b.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            b.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
            b.attrs["DateCreated"] = str(Time.now())
            b.attrs["MartiniVersion"] = martini_version
            b.attrs["AstropyVersion"] = astropy_version

        if channels == "frequency":
            self.datacube.velocity_channels()
        if memmap:
            return f
        else:
            f.close()
            return

    def reset(self):
        """
        Re-initializes the DataCube with zero-values.
        """
        init_kwargs = dict(
            n_px_x=self.datacube.n_px_x,
            n_px_y=self.datacube.n_px_y,
            n_channels=self.datacube.n_channels,
            px_size=self.datacube.px_size,
            channel_width=self.datacube.channel_width,
            velocity_centre=self.datacube.velocity_centre,
            ra=self.datacube.ra,
            dec=self.datacube.dec,
        )
        self.datacube = DataCube(**init_kwargs)
        if self.beam is not None:
            self.datacube.add_pad(self.beam.needs_pad())
        return
