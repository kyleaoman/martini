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
from martini.datacube import DataCube, _GlobalProfileDataCube
from martini.sph_kernels import DiracDeltaKernel

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


class _BaseMartini:
    """
    Common methods for the core Martini class and related classes.

    Parameters
    ----------
    source : SPHSource
        An instance of a class derived from :class:`martini.sources.SPHSource`.
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). See :doc:`sub-module documentation </sources/index>`.

    datacube : DataCube
        A :class:`~martini.datacube.DataCube` instance.
        A description of the datacube to create, including pixels, channels,
        sky position. See :doc:`sub-module documentation </datacube/index>`.

    beam : _BaseBeam, optional
        An instance of a class derived from `~martini.beams._BaseBeam`.
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See
        :doc:`sub-module documentation </beams/index>`.

    noise : _BaseNoise, optional
        An instance of a class derived from :class:`~martini.noise._BaseNoise`.
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        :doc:`sub-module documentation </noise/index>`.

    sph_kernel : _BaseSPHKernel
        An instance of a class derived from :class:`~martini.sph_kernels._BaseSPHKernel`.
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        :doc:`SPH kernel sub-module documentation </sph_kernels/index>` for guidance.

    spectral_model : _BaseSpectrum
        An instance of a class derived from
        :class:`~martini.spectral_models._BaseSpectrum`.
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See
        :doc:`sub-module documentation </spectral_models/index>`.

    quiet : bool, optional
        If ``True``, suppress output to stdout. (Default: ``False``)

    See Also
    --------
    ~martini.sources.sph_source.SPHSource
    ~martini.datacube.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models
    ~martini.martini.GlobalProfile
    """

    def __init__(
        self,
        source=None,
        datacube=None,
        beam=None,
        noise=None,
        sph_kernel=None,
        spectral_model=None,
        _prune_kwargs=dict(),
        quiet=False,
    ):
        self.quiet = quiet
        if source is not None:
            self.source = source
        else:
            raise ValueError("A source instance is required.")
        if datacube is not None:
            self._datacube = datacube
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
            self.beam.init_kernel(self._datacube)
            self._datacube.add_pad(self.beam.needs_pad())

        self.source._init_skycoords()
        self.source._init_pixcoords(self._datacube)  # after datacube is padded

        self.sph_kernel._init_sm_lengths(source=self.source, datacube=self._datacube)
        self.sph_kernel._init_sm_ranges()
        self._prune_particles(
            **_prune_kwargs
        )  # prunes both source, and kernel if applicable

        self.spectral_model.init_spectra(self.source, self._datacube)

        return

    def _prune_particles(self, spatial=True, spectral=True, obj_type_str="data cube"):
        """
        Determines which particles cannot contribute to the DataCube and
        removes them to speed up calculation. Assumes the kernel is 0 at
        distances greater than the kernel size (which may differ from the
        SPH smoothing length).

        Parameters
        ----------
        spatial : bool
            If ``True``, prune particles that fall outside the spatial aperture.
            (Default: ``True``)
        spectral : bool
            If ``True``, prune particles that fall outside the spectral bandwidth.
            (Default: ``True``)
        obj_type_str : str
            String describing the object to be pruned for messages.
            (Default: ``"data cube"``)
        """

        if not self.quiet:
            print(
                f"Source module contained {self.source.npart} particles with total HI"
                f" mass of {self.source.mHI_g.sum():.2e}."
            )
        spectrum_half_width = self.spectral_model.half_width(self.source) / np.max(
            np.abs(np.diff(self._datacube.velocity_channel_edges))
        )
        spatial_reject_conditions = (
            (
                (
                    self.source.pixcoords[:2] + self.sph_kernel.sm_ranges[np.newaxis]
                    < 0 * U.pix
                ).any(axis=0),
                self.source.pixcoords[0] - self.sph_kernel.sm_ranges
                > (self._datacube.n_px_x + self._datacube.padx * 2) * U.pix,
                self.source.pixcoords[1] - self.sph_kernel.sm_ranges
                > (self._datacube.n_px_y + self._datacube.pady * 2) * U.pix,
            )
            if spatial
            else tuple()
        )
        spectral_reject_conditions = (
            (
                self.source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
                self.source.pixcoords[2] - 4 * spectrum_half_width * U.pix
                > self._datacube.n_channels * U.pix,
            )
            if spectral
            else tuple()
        )
        reject_mask = np.zeros(self.source.pixcoords[0].shape)
        # this could be a logical_or.reduce?:
        for condition in spatial_reject_conditions + spectral_reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        self.source.apply_mask(np.logical_not(reject_mask))
        self.sph_kernel._apply_mask(np.logical_not(reject_mask))
        if not self.quiet:
            print(
                f"Pruned particles that will not contribute to {obj_type_str}, "
                f"{self.source.npart} particles remaining with total HI mass of "
                f"{self.source.mHI_g.sum():.2e}."
            )
        return

    def _evaluate_pixel_spectrum(self, ranks_and_ij_pxs, progressbar=True):
        """
        Add up contributions of particles to the spectrum in a pixel.

        This is the core loop of MARTINI. It is embarrassingly parallel. To support
        parallel excecution we accept storing up to a copy of the entire (future) datacube
        in one-pixel pieces. This avoids the need for concurrent access to the datacube
        by parallel processes, which would in the simplest case duplicate a copy of the
        datacube array per parallel process! In realistic use cases the memory overhead
        from the equivalent of a second datacube array should be minimal - memory-
        limited applications should be limited by the memory consumed by particle data,
        which is not duplicated in parallel execution.

        The arguments that differ between parallel ranks must be bundled into one for
        compatibility with `multiprocess`.

        Parameters
        ----------
        rank_and_ij_pxs : tuple
            A 2-tuple containing an integer (cpu "rank" in the case of parallel execution)
            and a list of 2-tuples specifying the indices (i, j) of pixels in the grid.

        progressbar : bool, optional
            Whether to display a :mod:`tqdm` progressbar. (Default: ``True``)

        Returns
        -------
        out : list
            A list containing 2-tuples. Each 2-tuple contains and "insertion slice" that
            is an index into the datacube._array instance held by this martini instance
            where the pixel spectrum is to be placed, and a 1D array containing the
            spectrum, whose length must match the length of the spectral axis of the
            datacube.
        """
        result = list()
        rank, ij_pxs = ranks_and_ij_pxs
        if progressbar:
            ij_pxs = tqdm.tqdm(ij_pxs, position=rank)
        for ij_px in ij_pxs:
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            mask = (
                np.abs(ij - self.source.pixcoords[:2]) <= self.sph_kernel.sm_ranges
            ).all(axis=0)
            weights = self.sph_kernel._px_weight(
                self.source.pixcoords[:2, mask] - ij, mask=mask
            )
            insertion_slice = (
                np.s_[ij_px[0], ij_px[1], :, 0]
                if self._datacube.stokes_axis
                else np.s_[ij_px[0], ij_px[1], :]
            )
            result.append(
                (
                    insertion_slice,
                    (self.spectral_model.spectra[mask] * weights[..., np.newaxis]).sum(
                        axis=-2
                    ),
                )
            )
        return result

    def _insert_pixel(self, insertion_slice, insertion_data):
        """
        Insert the spectrum for a single pixel into the datacube array.

        Parameters
        ----------
        insertion_slice : integer, tuple or slice
            Index into the datacube's _array specifying the insertion location.

        insertion data : array-like
            1D array containing the spectrum at the location specified by insertion_slice.
        """
        self._datacube._array[insertion_slice] = insertion_data
        return

    def _insert_source_in_cube(
        self, skip_validation=False, progressbar=None, ncpu=1, quiet=None
    ):
        """
        Populates the :class:`~martini.datacube.DataCube` with flux from the
        particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the DataCube is approximated for
            increased speed. For some combinations of pixel size, distance
            and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            RuntimeError if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter True. (Default: ``False``)

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If martini was initialised with
            `quiet` set to `True`, progress bars are switched off unless explicitly
            turned on. (Default: ``None``)

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the `multiprocess` module (n.b. not the same as
            `multiprocessing`). (Default: ``1``)

        quiet : bool, optional
            If ``True``, suppress output to stdout. If specified, takes precedence over
            quiet parameter of class. (Default: ``None``)
        """

        assert self.spectral_model.spectra is not None

        if progressbar is None:
            progressbar = not self.quiet

        self.sph_kernel._confirm_validation(noraise=skip_validation, quiet=self.quiet)

        ij_pxs = list(
            product(
                np.arange(self._datacube._array.shape[0]),
                np.arange(self._datacube._array.shape[1]),
            )
        )

        if ncpu == 1:
            for insertion_slice, insertion_data in self._evaluate_pixel_spectrum(
                (0, ij_pxs), progressbar=progressbar
            ):
                self._insert_pixel(insertion_slice, insertion_data)
        else:
            # not multiprocessing, need serialization from dill not pickle
            from multiprocess import Pool

            with Pool(processes=ncpu) as pool:
                for result in pool.imap_unordered(
                    lambda x: self._evaluate_pixel_spectrum(x, progressbar=progressbar),
                    [(icpu, ij_pxs[icpu::ncpu]) for icpu in range(ncpu)],
                ):
                    for insertion_slice, insertion_data in result:
                        self._insert_pixel(insertion_slice, insertion_data)

        self._datacube._array = self._datacube._array.to(
            U.Jy / U.arcsec**2, equivalencies=[self._datacube.arcsec2_to_pix]
        )
        pad_mask = (
            np.s_[
                self._datacube.padx : -self._datacube.padx,
                self._datacube.pady : -self._datacube.pady,
                ...,
            ]
            if self._datacube.padx > 0 and self._datacube.pady > 0
            else np.s_[...]
        )
        inserted_flux_density = np.sum(
            self._datacube._array[pad_mask] * self._datacube.px_size**2
        ).to(U.Jy)
        inserted_mass = (
            2.36e5
            * U.Msun
            * self.source.distance.to_value(U.Mpc) ** 2
            * np.sum(
                (self._datacube._array[pad_mask] * self._datacube.px_size**2)
                .sum((0, 1))
                .squeeze()
                .to_value(U.Jy)
                * np.abs(np.diff(self._datacube.velocity_channel_edges)).to_value(
                    U.km / U.s
                )
            )
        )
        if (quiet is None and not self.quiet) or (quiet is not None and not quiet):
            print(
                "Source inserted.",
                f"  Flux density in cube: {inserted_flux_density:.2e}",
                f"  Mass in cube (assuming distance {self.source.distance:.2f} and a"
                f" spatially resolved source):"
                f" {inserted_mass:.2e}",
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]",
                f"  Maximum pixel: {self._datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self._datacube._array[self._datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def reset(self):
        """
        Re-initializes the :class:`~martini.datacube.DataCube` with zero-values.
        """
        init_kwargs = dict(
            n_px_x=self._datacube.n_px_x,
            n_px_y=self._datacube.n_px_y,
            n_channels=self._datacube.n_channels,
            px_size=self._datacube.px_size,
            channel_width=self._datacube.channel_width,
            spectral_centre=self._datacube.spectral_centre,
            ra=self._datacube.ra,
            dec=self._datacube.dec,
            stokes_axis=self._datacube.stokes_axis,
        )
        self._datacube = DataCube(**init_kwargs)
        if self.beam is not None:
            self._datacube.add_pad(self.beam.needs_pad())
        return

    def preview(
        self,
        max_points=5000,
        fig=1,
        lim=None,
        vlim=None,
        point_scaling="auto",
        title="",
        save=None,
    ):
        """
        Produce a figure showing the source particle coordinates and velocities, and the
        data cube region.

        Makes a 3-panel figure showing the projection of the source as it will appear in
        the mock observation. The first panel shows the particles in the y-z plane,
        coloured by the x-component of velocity (MARTINI projects the source along the
        x-axis). The second and third panels are position-velocity diagrams showing the
        x-component of velocity against the y and z coordinates, respectively. The red
        boxes drawn on the panels show the data cube extent in position and velocity at
        the distance and systemic velocity of the source.

        Parameters
        ----------
        max_points : int, optional
            Maximum number of points to draw per panel, the particles will be randomly
            subsampled if the source has more. (Default: ``1000``)

        fig : int, optional
            Number of the figure in matplotlib, it will be created as ``plt.figure(fig)``.
            (Default: ``1``)

        lim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity` with dimensions of length.
            The coordinate axes extend from ``-lim`` to ``lim``. If unspecified, the
            maximum absolute coordinate of particles in the source is used. Whether
            specified or not, the axes are expanded if necessary to contain the extent of
            the data cube. Alternatively, the string ``"datacube"`` can be passed. In this
            case the axis limits are set by the extent of the data cube.
            (Default: ``None``)

        vlim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity` with dimensions of speed.
            The velocity axes and colour bar extend from ``-vlim`` to ``vlim``. If
            unspecified, the maximum absolute velocity of particles in the source is
            used. Whether specified or not, the axes are expanded if necessary to contain
            the extent of the data cube. Alternatively, ``"datacube"`` can be
            passed. In this case the axis limits are set by the extent of the data cube.
            (Default: ``None``)

        point_scaling : str, optional
            By default points are scaled in size and transparency according to their HI
            mass and the smoothing length (loosely proportional to their surface
            densities, but with different scaling to achieve a visually useful plot). For
            some sources the automatic scaling may not give useful results, using
            ``point_scaling="fixed"`` will plot points of constant size without opacity.
            (Default: ``"auto"``)

        title : str, optional
            A title for the figure can be provided. (Default: ``""``)

        save : str, optional
            If provided, the figure is saved using ``plt.savefig(save)``. A `.png` or
            `.pdf` suffix is recommended. (Default: ``None``)

        Returns
        -------
        out : Figure
            The preview :class:`~matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt

        y_cube = (
            (self._datacube.ra - self.source.ra)
            / np.cos(self.source.dec)
            * self.source.distance
        ).to(U.kpc, U.dimensionless_angles())
        z_cube = ((self._datacube.dec - self.source.dec) * self.source.distance).to(
            U.kpc, U.dimensionless_angles()
        )
        v_cube = (self._datacube.spectral_centre - self.source.vsys).to(U.km / U.s)
        dy_cube = 0.5 * (
            self._datacube.px_size * self._datacube.n_px_x * self.source.distance
        ).to(
            U.kpc, U.dimensionless_angles()
        )  # half-length
        dz_cube = 0.5 * (
            self._datacube.px_size * self._datacube.n_px_y * self.source.distance
        ).to(
            U.kpc, U.dimensionless_angles()
        )  # half-length
        dv_cube = 0.5 * (self._datacube.channel_width * self._datacube.n_channels).to(
            U.km / U.s
        )
        # should issue some warnings if dRA or dDec > 5deg, or if dec within 5 deg of pole
        clip_lim = lim == "datacube"
        if clip_lim:
            lim = max(
                np.abs(
                    U.Quantity(
                        [
                            y_cube - dy_cube,
                            y_cube + dy_cube,
                            z_cube - dz_cube,
                            z_cube + dz_cube,
                        ]
                    )
                )
            )
        clip_vlim = vlim == "datacube"
        if clip_vlim:
            clip_vlim = True
            vlim = max(np.abs(U.Quantity([v_cube - dv_cube, v_cube + dv_cube])))
        else:
            clip_vlim = False
        # pass through arguments, except save (which we will do later if desired)
        fig = self.source.preview(
            max_points=max_points,
            fig=fig,
            lim=lim,
            vlim=vlim,
            point_scaling=point_scaling,
            title=title,
            save=None,
        )
        sp1, cb, sp2, sp3 = fig.get_axes()
        sp1.add_patch(
            plt.Rectangle(
                (
                    (y_cube - dy_cube).to_value(U.kpc),
                    (z_cube - dz_cube).to_value(U.kpc),
                ),
                2 * dy_cube.to_value(U.kpc),
                2 * dz_cube.to_value(U.kpc),
                linewidth=5,
                edgecolor="red",
                facecolor="None",
            )
        )
        sp2.add_patch(
            plt.Rectangle(
                (
                    (y_cube - dy_cube).to_value(U.kpc),
                    (v_cube - dv_cube).to_value(U.km / U.s),
                ),
                2 * dy_cube.to_value(U.kpc),
                2 * dv_cube.to_value(U.km / U.s),
                linewidth=5,
                edgecolor="red",
                facecolor="None",
            )
        )
        sp3.add_patch(
            plt.Rectangle(
                (
                    (z_cube - dz_cube).to_value(U.kpc),
                    (v_cube - dv_cube).to_value(U.km / U.s),
                ),
                2 * dz_cube.to_value(U.kpc),
                2 * dv_cube.to_value(U.km / U.s),
                linewidth=5,
                edgecolor="red",
                facecolor="None",
            )
        )
        if clip_lim:
            sp1.set_xlim(
                (y_cube + dy_cube).to_value(U.kpc), (y_cube - dy_cube).to_value(U.kpc)
            )
            sp1.set_ylim(
                (z_cube - dz_cube).to_value(U.kpc), (z_cube + dz_cube).to_value(U.kpc)
            )
            sp2.set_xlim(
                (y_cube + dy_cube).to_value(U.kpc), (y_cube - dy_cube).to_value(U.kpc)
            )
            sp3.set_xlim(
                (z_cube - dz_cube).to_value(U.kpc), (z_cube + dz_cube).to_value(U.kpc)
            )
        else:
            if lim is None:
                lim = sp1.get_xlim()[1] * U.kpc
            sp1.set_xlim(
                (
                    max(1.1 * (y_cube + dy_cube), lim).to_value(U.kpc),
                    min(1.1 * (y_cube - dy_cube), -lim).to_value(U.kpc),
                )
            )
            sp2.set_xlim(
                (
                    max(1.1 * (y_cube + dy_cube), lim).to_value(U.kpc),
                    min(1.1 * (y_cube - dy_cube), -lim).to_value(U.kpc),
                )
            )
            sp1.set_ylim(
                (
                    min(1.1 * (z_cube - dz_cube), -lim).to_value(U.kpc),
                    max(1.1 * (z_cube + dz_cube), lim).to_value(U.kpc),
                )
            )
            sp3.set_xlim(
                (
                    min(1.1 * (z_cube - dz_cube), -lim).to_value(U.kpc),
                    max(1.1 * (z_cube + dz_cube), lim).to_value(U.kpc),
                )
            )
        if clip_vlim:
            sp2.set_ylim(
                (v_cube - dv_cube).to_value(U.km / U.s),
                (v_cube + dv_cube).to_value(U.km / U.s),
            )
            sp3.set_ylim(
                (v_cube - dv_cube).to_value(U.km / U.s),
                (v_cube + dv_cube).to_value(U.km / U.s),
            )
        else:
            if vlim is None:
                vlim = sp2.get_ylim()[1] * U.km / U.s
            sp2.set_ylim(
                (
                    min(1.1 * (v_cube - dv_cube), -vlim).to_value(U.km / U.s),
                    max(1.1 * (v_cube + dv_cube), vlim).to_value(U.km / U.s),
                )
            )
            sp3.set_ylim(
                (
                    min(1.1 * (v_cube - dv_cube), -vlim).to_value(U.km / U.s),
                    max(1.1 * (v_cube + dv_cube), vlim).to_value(U.km / U.s),
                )
            )

        if save is not None:
            plt.savefig(save)
        return fig


class Martini(_BaseMartini):
    """
    Creates synthetic HI data cubes from simulation data.

    Usual use of martini involves first creating instances of classes from each
    of the required and optional sub-modules, then creating a
    :class:`~martini.martini.Martini` with these instances as arguments. The object can
    then be used to create synthetic observations, usually by calling
    :meth:`~martini.martini.Martini.insert_source_in_cube`,
    (optionally) :meth:`~martini.martini.Martini.add_noise`, (optionally)
    :meth:`~martini.martini.Martini.convolve_beam` and
    :meth:`~martini.martini.Martini.write_fits` in order.

    Parameters
    ----------
    source : SPHSource
        An instance of a class derived from
        :class:`~martini.sources.sph_source.SPHSource`.
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). See :doc:`sub-module documentation </sources/index>`.

    datacube : DataCube
        A :class:`~martini.datacube.DataCube` instance.
        A description of the datacube to create, including pixels, channels,
        sky position. See :doc:`sub-module documentation </datacube/index>`.

    beam : _BaseBeam, optional
        An instance of a class derived from `~martini.beams._BaseBeam`.
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See
        :doc:`sub-module documentation </beams/index>`.

    noise : _BaseNoise, optional
        An instance of a class derived from :class:`~martini.noise._BaseNoise`.
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        :doc:`sub-module documentation </noise/index>`.

    sph_kernel : _BaseSPHKernel
        An instance of a class derived from :class:`~martini.sph_kernels._BaseSPHKernel`.
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        :doc:`SPH kernel sub-module documentation </sph_kernels/index>` for guidance.

    spectral_model : _BaseSpectrum
        An instance of a class derived from
        :class:`~martini.spectral_models._BaseSpectrum`.
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See
        :doc:`sub-module documentation </spectral_models/index>`.

    quiet : bool, optional
        If ``True``, suppress output to stdout. (Default: ``False``)

    See Also
    --------
    ~martini.sources.sph_source.SPHSource
    ~martini.datacube.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models
    ~martini.martini.GlobalProfile

    Examples
    --------
    More detailed examples can be found in the examples directory in the github
    distribution of the package.

    The following example illustrates basic use of MARTINI, using a (very!)
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
            spectral_centre=source.vsys,
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
        M.write_beam_fits(beamfile)
        M.write_fits(cubefile)
        print(f"Wrote demo fits output to {cubefile}, and beam image to {beamfile}.")
        try:
            M.write_hdf5(hdf5file)
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
        super().__init__(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
            quiet=quiet,
        )

        return

    @property
    def datacube(self):
        """
        The :class:`~martini.datacube.DataCube` object for this mock observation.

        Returns
        -------
        out : ~martini.datacube.DataCube
            The :class:`~martini.datacube.DataCube` contained by this
            :class:`~martini.martini.Martini` instance.
        """
        return self._datacube

    def insert_source_in_cube(self, skip_validation=False, progressbar=None, ncpu=1):
        """
        Populates the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the :class:`~martini.datacube.DataCube`
            is approximated for increased speed. For some combinations of pixel size,
            distance and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            ``RuntimeError`` if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter ``True``. (Default: ``False``)

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If :class:`~martini.martini.Martini` was
            initialised with ``quiet`` set to ``True``, progress bars are switched off
            unless explicitly turned on. (Default: ``None``)

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the :mod:`multiprocess` module (n.b. not the same as
            ``multiprocessing``). (Default: ``1``)

        """

        super()._insert_source_in_cube(
            skip_validation=skip_validation, progressbar=progressbar, ncpu=ncpu
        )

        return

    def convolve_beam(self):
        """
        Convolve the beam and data cube.
        """

        if self.beam is None:
            warn("Skipping beam convolution, no beam object provided to Martini.")
            return

        minimum_padding = self.beam.needs_pad()
        if (self._datacube.padx < minimum_padding[0]) or (
            self._datacube.pady < minimum_padding[1]
        ):
            raise ValueError(
                "datacube padding insufficient for beam convolution (perhaps you loaded a"
                " datacube state with datacube.load_state that was previously initialized"
                " by martini with a smaller beam?)"
            )

        unit = self._datacube._array.unit
        for spatial_slice in self._datacube.spatial_slices:
            # use a view [...] to force in-place modification
            spatial_slice[...] = (
                fftconvolve(spatial_slice, self.beam.kernel, mode="same") * unit
            )
        self._datacube.drop_pad()
        self._datacube._array = self._datacube._array.to(
            U.Jy * U.beam**-1,
            equivalencies=U.beam_angular_area(self.beam.area),
        )
        if not self.quiet:
            print(
                "Beam convolved.",
                "  Data cube RMS after beam convolution:"
                f" {np.std(self._datacube._array):.2e}",
                f"  Maximum pixel: {self._datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self._datacube._array[self._datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def add_noise(self):
        """
        Insert noise into the data cube.
        """

        if self.noise is None:
            warn("Skipping noise, no noise object provided to Martini.")
            return

        # this unit conversion means noise can be added before or after source insertion:
        noise_cube = (
            self.noise.generate(self._datacube, self.beam)
            .to(
                U.Jy * U.arcsec**-2,
                equivalencies=U.beam_angular_area(self.beam.area),
            )
            .to(
                self._datacube._array.unit,
                equivalencies=[self._datacube.arcsec2_to_pix],
            )
        )
        self._datacube._array = self._datacube._array + noise_cube
        if not self.quiet:
            print(
                "Noise added.",
                f"  Noise cube RMS: {np.std(noise_cube):.2e} (before beam convolution).",
                "  Data cube RMS after noise addition (before beam convolution): "
                f"{np.std(self._datacube._array):.2e}",
                sep="\n",
            )
        return

    def write_fits(
        self,
        filename,
        overwrite=True,
        channels=None,  # deprecated
    ):
        """
        Output the data cube to a FITS-format file.

        Parameters
        ----------
        filename : str
            Name of the file to write. ``'.fits'`` will be appended if not already
            present.

        overwrite : bool, optional
            Whether to allow overwriting existing files. (Default: ``True``)

        channels : str, deprecated
            Deprecated, channels and their units now fixed at
            :class:`~martini.datacube.DataCube` initialization.
        """

        if channels is not None:
            DeprecationWarning(
                "`channels` argument to `write_fits` ignored, channels and their units"
                " now fixed at DataCube initialization."
            )
        self._datacube.drop_pad()

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self._datacube.wcs.to_header()
        wcs_header.rename_keyword("WCSAXES", "NAXIS")

        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", self._datacube.n_px_x))
        header.append(("NAXIS2", self._datacube.n_px_y))
        header.append(("NAXIS3", self._datacube.n_channels))
        if self._datacube.stokes_axis:
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
        if self._datacube.stokes_axis:
            header.append(("CDELT4", wcs_header["CDELT4"]))
            header.append(("CRPIX4", wcs_header["CRPIX4"]))
            header.append(("CRVAL4", wcs_header["CRVAL4"]))
            header.append(("CTYPE4", wcs_header["CTYPE4"]))
            header.append(("CUNIT4", "PAR"))
        header.append(("EPOCH", 2000))
        header.append(("INSTRUME", "MARTINI", martini_version))
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        datacube_array_units = self._datacube._array.unit
        header.append(
            ("DATAMAX", np.max(self._datacube._array.to_value(datacube_array_units)))
        )
        header.append(
            ("DATAMIN", np.min(self._datacube._array.to_value(datacube_array_units)))
        )
        header.append(("ORIGIN", "astropy v" + astropy_version))
        # long names break fits format, don't let the user set this
        header.append(("OBJECT", "MOCK"))
        if self.beam is not None:
            header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("OBSERVER", "K. Oman"))
        header.append(("BUNIT", datacube_array_units.to_string("fits")))
        header.append(("DATE-OBS", Time.now().to_value("fits")))
        header.append(("MJD-OBS", Time.now().to_value("mjd")))
        if self.beam is not None:
            header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
            header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))
        header.append(("RESTFRQ", wcs_header["RESTFRQ"]))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header, data=self._datacube._array.to_value(datacube_array_units).T
        )
        hdu.writeto(filename, overwrite=overwrite)

        return

    def write_beam_fits(
        self,
        filename,
        overwrite=True,
        channels=None,  # deprecated
    ):
        """
        Output the beam to a FITS-format file.

        The beam is written to file, with pixel sizes, coordinate system, etc.
        similar to those used for the :class:`~martini.datacube.DataCube`.

        Parameters
        ----------
        filename : str
            Name of the file to write. ``".fits"`` will be appended if not already
            present.

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: ``True``)

        channels : str, deprecated
            Deprecated, channels and their units now fixed at
            :class:`~martini.datacube.DataCube` initialization.

        Raises
        ------
        ValueError
            If :class:`~martini.martini.Martini` was initialized without a ``beam``.
        """

        if channels is not None:
            DeprecationWarning(
                "`channels` argument to `write_fits` ignored, channels and their units"
                " now fixed at DataCube initialization."
            )

        if self.beam is None:
            raise ValueError(
                "Martini.write_beam_fits: Called with beam set " "to 'None'."
            )
        assert self.beam.kernel is not None

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self._datacube.wcs.to_header()

        beam_kernel_units = self.beam.kernel.unit
        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
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

        return

    def write_hdf5(
        self,
        filename,
        overwrite=True,
        memmap=False,
        compact=False,
        channels=None,  # deprecated
    ):
        """
        Output the data cube and beam to a HDF5-format file. Requires the :mod:`h5py`
        package.

        Parameters
        ----------
        filename : str
            Name of the file to write. ``'.hdf5'`` will be appended if not already
            present.

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: ``True``)

        memmap: bool, optional
            If ``True``, create a file-like object in memory and return it instead
            of writing file to disk. (Default: ``False``)

        compact: bool, optional
            If ``True``, omit pixel coordinate arrays to save disk space. In this
            case pixel coordinates can still be reconstructed from FITS-style
            keywords stored in the FluxCube attributes. (Default: ``False``)

        channels : str, deprecated
            Deprecated, channels and their units now fixed at
            :class:`~martini.datacube.DataCube` initialization.

        """

        if channels is not None:
            DeprecationWarning(
                "`channels` argument to `write_fits` ignored, channels and their units"
                " now fixed at DataCube initialization."
            )

        import h5py

        self._datacube.drop_pad()

        filename = filename if filename[-5:] == ".hdf5" else filename + ".hdf5"

        wcs_header = self._datacube.wcs.to_header()

        mode = "w" if overwrite else "x"
        driver = "core" if memmap else None
        h5_kwargs = {"backing_store": False} if memmap else dict()
        f = h5py.File(filename, mode, driver=driver, **h5_kwargs)
        datacube_array_units = self._datacube._array.unit
        s = np.s_[..., 0] if self._datacube.stokes_axis else np.s_[...]
        f["FluxCube"] = self._datacube._array.to_value(datacube_array_units)[s]
        c = f["FluxCube"]
        origin = 0  # index from 0 like numpy, not from 1
        if not compact:
            xgrid, ygrid, vgrid = np.meshgrid(
                np.arange(self._datacube._array.shape[0]),
                np.arange(self._datacube._array.shape[1]),
                np.arange(self._datacube._array.shape[2]),
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
                if self._datacube.stokes_axis
                else np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                    )
                ).T
            )
            wgrid = self._datacube.wcs.all_pix2world(cgrid, origin)
            ragrid = wgrid[:, 0].reshape(self._datacube._array.shape)[s]
            decgrid = wgrid[:, 1].reshape(self._datacube._array.shape)[s]
            chgrid = wgrid[:, 2].reshape(self._datacube._array.shape)[s]
            f["RA"] = ragrid
            f["RA"].attrs["Unit"] = wcs_header["CUNIT1"]
            f["Dec"] = decgrid
            f["Dec"].attrs["Unit"] = wcs_header["CUNIT2"]
            f["channel_mids"] = chgrid
            f["channel_mids"].attrs["Unit"] = wcs_header["CUNIT3"]
            for dataset_name in (
                "velocity_channel_mids",
                "velocity_channel_edges",
                "frequency_channel_mids",
                "frequency_channel_edges",
            ):
                f[dataset_name] = getattr(self._datacube, dataset_name)
                f[dataset_name].attrs["Unit"] = str(
                    getattr(self._datacube, dataset_name).unit
                )
        c.attrs["AxisOrder"] = "(RA,Dec,Channels)"
        c.attrs["FluxCubeUnit"] = str(self._datacube._array.unit)
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
        c.attrs["SpecSys"] = wcs_header["SPECSYS"]
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

        if memmap:
            return f
        else:
            f.close()
            return


class GlobalProfile(_BaseMartini):
    """
    A simplified version of the main :class:`~martini.martini.Martini` class to just
    produce a spectrum.

    Sometimes only the spatially integrated spectrum of a source is wanted, which makes
    many parts of the standard MARTINI process unnecessary. This class streamlines the
    process of producing a spatially-integrated spectrum, or "global profile", with some
    specific assumptions:

     - The :class:`~martini.martini.GlobalProfile` class does not assume any spatial
       aperture, instead every particle in the source contributes to the spectrum (unless
       it falls outside of the spectral bandwidth).
     - The positions of particles are still used to calculate the line-of-sight vector
       and the velocity along this direction.

    There is therefore no need or way to specify a beam or SPH kernel as with the main
    :class:`~martini.martini.Martini` class. It is also not possible to use MARTINI's
    noise modules with this class. If these restrictions are found to be too limiting
    (for example, if the spectrum within a spatial mask defined by a signal-to-noise or
    other cut is desired), the best course of action is to produce a spatially-resolved
    mock observation and derive the spectrum from those data as would be done with "real"
    observations. This class is mainly intended to efficiently provide a "quick look" at
    the spectrum, or a reference "ideal" spectrum.

    Parameters
    ----------
    source : ~martini.sources.sph_source.SPHSource
        An instance of a class derived from
        :class:`~martini.sources.sph_source.SPHSource`.
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). See :doc:`sub-module documentation </sources/index>`.

    spectral_model : ~martini.spectral_models._BaseSpectrum
        An instance of a class derived from
        :class:`~martini.spectral_models._BaseSpectrum`.
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See
        :doc:`sub-module documentation </spectral_models/index>`.

    n_channels : int, optional
        Number of channels along the spectral axis. (Default: ``64``)

    channel_width : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity or frequency.
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency. (Default: ``4 * U.km / U.s``)

    spectral_centre : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity` with dimensions of velocity or frequency.
        Velocity (or frequency) of the centre along the spectral axis.
        (Default: ``0 * U.km / U.s``)

    quiet : bool, optional
        If ``True``, suppress output to stdout. (Default: ``False``)

    channels : str, deprecated
        Deprecated, channels and their units now fixed at
        :class:`~martini.datacube.DataCube` initialization.

    Examples
    --------
    ::

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

        spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

        M = GlobalProfile(
            source=source,
            spectral_model=spectral_model,
            n_channels=32,
            channel_width=10 * U.km * U.s**-1,
            spectral_centre=source.vsys,
        )

        # spectrum and channel edges and centres can be accessed as:
        M.spectrum  # spectrum is evaluated at first access if not already done
        M.channel_edges
        M.channel_mids

        # the spectrum can be explicitly evaluated with:
        M.insert_source_in_spectrum()

    See Also
    --------
    ~martini.sources.sph_source.SPHSource
    martini.spectral_models
    ~martini.martini.Martini
    """

    def __init__(
        self,
        source=None,
        spectral_model=None,
        n_channels=64,
        channel_width=4 * U.km * U.s**-1,
        spectral_centre=0 * U.km * U.s**-1,
        quiet=False,
        channels=None,  # deprecated
    ):
        if channels is not None:
            DeprecationWarning(
                "The `channels` argument to `GlobalProfile.__init__` is deprecated"
                " and has been ignored. If `channel_width` has velocity units channels"
                " are evenly spaced in velocity, and if it has frequency units they are"
                " evenly spaced in frequency."
            )
        super().__init__(
            source=source,
            datacube=_GlobalProfileDataCube(
                n_channels=n_channels,
                channel_width=channel_width,
                spectral_centre=spectral_centre,
            ),
            beam=None,
            noise=None,
            sph_kernel=DiracDeltaKernel(size_in_fwhm=np.inf),
            spectral_model=spectral_model,
            _prune_kwargs=dict(
                spatial=False,
                spectral=True,
                obj_type_str="spectrum",
            ),
            quiet=quiet,
        )
        self.source.pixcoords[:2] = 0

        return

    def insert_source_in_spectrum(self):
        """
        Populates the :class:`~martini.datacube.DataCube` with flux from the particles
        in the source.

        For the :class:`~martini.martini.GlobalProfile` class we assume that every
        particle in the source contributes (if it falls in the spectral bandwidth)
        regardless of  position on the sky. The line-of-sight vector still depends on
        the particle positions, so the direction to the individual particles is still
        taken into account.
        """
        # skip_validation=True: all particles can contribute their kernel to the pixel;
        # ncpu=1 since we have 1 pixel and source insertion is parallel over pixels;
        # no progressbar since there's only 1 pixel of progress;
        # quiet=True because messages assume a resolved source, replace with new ones
        super()._insert_source_in_cube(
            skip_validation=True, progressbar=False, ncpu=1, quiet=True
        )
        # The datacube in Jy/arcsec^2 is a bit misleading because the source is
        # (presumably) completely unresolved so extrapolating its surface brightness
        # across the entire pixel is incorrect. Correctly integrate out spatial
        # information and convert to Jy:
        self._spectrum = (
            (self._datacube._array.squeeze()).to(
                U.Jy / U.pix**2, equivalencies=[self._datacube.arcsec2_to_pix]
            )
            * U.pix**2
        ).to(U.Jy)
        if not self.quiet:
            # Need a slightly different calculation for a completely unresolved source.
            inserted_flux_density = self.spectrum.sum()
            inserted_mass = (
                2.36e5
                * U.Msun
                * self.source.distance.to_value(U.Mpc) ** 2
                * inserted_flux_density.to_value(U.Jy)
                * self._datacube.channel_width.to_value(U.km / U.s)
            )
            print(
                "Source inserted.",
                f"  Flux density in spectrum: {inserted_flux_density:.2e}",
                f"  Mass in spectrum (assuming distance {self.source.distance:.2f}):"
                f" {inserted_mass:.2e}",
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]",
                sep="\n",
            )

    @property
    def spectrum(self):
        """
        The spectrum of the source with spatial information integrated out.

        The spectrum is calculated when this property is accessed if this hasn't already
        been done.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of flux density.
            Spatially-integrated spectrum of the source.
        """
        if not hasattr(self, "_spectrum"):
            self.insert_source_in_spectrum()
        return self._spectrum

    @property
    def channel_edges(self):
        """
        The edges of the channels for the spectrum.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency or velocity.
            Edges of the channels with units depending on
            :class:`~martini.martini.GlobalProfile`'s native channel spacing.

        See Also
        --------
        ~martini.martini.GlobalProfile.channel_mids
        ~martini.martini.GlobalProfile.velocity_channel_mids
        ~martini.martini.GlobalProfile.velocity_channel_edges
        ~martini.martini.GlobalProfile.frequency_channel_mids
        ~martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.channel_edges

    @property
    def channel_mids(self):
        """
        The centres of the channels for the spectrum.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency or velocity.
            Edges of the channels with units depending on
            :class:`~martini.martini.GlobalProfile`'s native channel spacing.

        See Also
        --------
        ~martini.martini.GlobalProfile.channel_edges
        ~martini.martini.GlobalProfile.velocity_channel_mids
        ~martini.martini.GlobalProfile.velocity_channel_edges
        ~martini.martini.GlobalProfile.frequency_channel_mids
        ~martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.channel_mids

    @property
    def frequency_channel_edges(self):
        """
        The edges of the frequency channels for the spectrum.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency.
            Edges of the channels with frequency units.

        See Also
        --------
        ~martini.martini.GlobalProfile.velocity_channel_mids
        ~martini.martini.GlobalProfile.velocity_channel_edges
        ~martini.martini.GlobalProfile.frequency_channel_mids
        """
        return self._datacube.channel_edges

    @property
    def frequency_channel_mids(self):
        """
        The centres of the frequency channels for the spectrum.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency.
            Edges of the channels with frequency units.

        See Also
        --------
        ~martini.martini.GlobalProfile.velocity_channel_mids
        ~martini.martini.GlobalProfile.velocity_channel_edges
        ~martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.channel_mids

    @property
    def velocity_channel_edges(self):
        """
        The edges of the channels for the spectrum in velocity units.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of velocity.
            Edges of the channels with velocity units.

        See Also
        --------
        ~martini.martini.GlobalProfile.velocity_channel_mids
        ~martini.martini.GlobalProfile.frequency_channel_mids
        ~martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.channel_edges

    @property
    def velocity_channel_mids(self):
        """
        The centres of the channels for the spectrum.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of velocity.
            Edges of the channels with velocity units.

        See Also
        --------
        ~martini.martini.GlobalProfile.velocity_channel_edges
        ~martini.martini.GlobalProfile.frequency_channel_mids
        ~martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.channel_mids

    @property
    def channel_width(self):
        """
        The width of the channels for the spectrum.

        Returns
        -------
        out : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimentions of frequency or velocity.
            Width of the channels with units depending on
            :class:`~martini.martini.GlobalProfile`'s ``channels`` argument.
        """
        return self._datacube.channel_width

    def reset(self):
        """
        Re-initializes the spectrum with zero-values.
        """
        super().reset()
        if hasattr(self, "_spectrum"):
            del self._spectrum
        return

    def plot_spectrum(
        self,
        fig=1,
        title="",
        channels="velocity",
        show_vsys=True,
        save=None,
    ):
        """
        Produce a figure showing the spectrum.

        Parameters
        ----------
        fig : int, optional
            Number of the figure in matplotlib, it will be created as ``plt.figure(fig)``.
            (Default: ``1``)

        title : str, optional
            A title for the figure can be provided. (Default: ``""``)

        channels : str, optional
            The type of spectral axis for the plot, either ``"velocity"`` or
            ``"frequency"``. (Default: ``"velocity"``)

        show_vsys : bool, optional
            If ``True``, draw a vertical line at the source systemic velocity.
            (Default: ``True``)

        save : str, optional
            If provided, the figure is saved using ``plt.savefig(save)``. A `.png` or
            `.pdf` suffix is recommended. (Default: ``None``)

        Returns
        -------
        out : Figure
            The spectrum :class:`~matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt

        if channels == "velocity":
            channel_mids = self.velocity_channel_mids
        elif channels == "frequency":
            channel_mids = self.frequency_channel_mids
        else:
            raise ValueError(
                "`plot_spectrum` argument `channels` must be 'velocity' or 'frequency'."
            )

        fig = plt.figure(fig, figsize=(4, 3))
        fig.clf()
        fig.suptitle(title)

        xunit = dict(velocity=U.km * U.s**-1, frequency=U.GHz)[channels]

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            channel_mids.to_value(xunit),
            self.spectrum.to_value(U.Jy),
            ls="solid",
            color="black",
            lw=2,
        )
        if show_vsys:
            ax.axvline(
                self.source.vsys.to_value(xunit),
                linestyle="dotted",
                lw=1.5,
                color="black",
            )
        ax.set_ylabel(r"Flux density $[\mathrm{Jy}]$")
        if channels == "velocity":
            ax.set_xlabel(r"Velocity $[\mathrm{km}\,\mathrm{s}^{-1}]$")
        elif channels == "frequency":
            ax.set_xlabel(r"Frequency $[\mathrm{GHz}]$")

        if save is not None:
            plt.savefig(save)
        return fig
