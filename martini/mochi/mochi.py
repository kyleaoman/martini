"""
Create mock observations of HI sources from cosmological hydrodynamical simulations.

Provides :class:`~martini.martini.Mochi`, the main class of the package.
"""

import warnings
import numpy as np
from astropy import units as U
from martini.martini import Martini
from martini.datacube import DataCube
from martini.beams import _BaseBeam
from martini.sources import SPHSource
from martini.sph_kernels import _BaseSPHKernel, DiracDeltaKernel
from martini.spectral_models import _BaseSpectrum
from martini.noise import _BaseNoise
from typing import Callable
from martini.mochi.cell_grid import CellGrid, AdaptiveCellGrid
from martini.mochi.refinement import refine_grid_to_half_particle_scale
from martini.mochi.radiative_transfer import optically_thin
from martini.mochi import interpolants
from martini._unit_conversion import MHI_to_Jy_inplace


def _coarse_shape(
    shape: tuple[int, int],
    max_extra_fraction: float = 0.1,
) -> tuple[tuple[int, int], int, tuple[int, int]]:
    """
    Find a coarse grid of power-of-two sized cells aligned with the pixel grid.

    The initial (coarse) cells have a side length of ``2**k`` pixels, so that cells
    obtained by repeated bisection have edges that coincide with pixel edges (or nest
    within a single pixel once smaller than a pixel). The coarse grid covers the input
    shape, which is padded symmetrically to match the coarse grid. The input shape must
    have even side lengths, such that the coarse grid extends an even number of pixels
    beyond it along each axis. The largest ``k`` such that the coarse grid covers at
    most ``1 + max_extra_fraction`` times as many pixels as the input shape is chosen.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the cube (including any existing padding), with even side lengths.

    max_extra_fraction : float, optional
        Maximum fractional increase in the number of pixels covered by the coarse grid,
        relative to ``shape``.

    Returns
    -------
    coarse_shape : tuple[int, int]
        Number of coarse cells along each axis.

    cell_size : int
        Side length of the coarse cells in pixels.

    pad : tuple[int, int]
        Padding (per side) to add to ``shape`` along each axis.
    """
    n_px = np.array(shape)
    for k in range(int(np.ceil(np.log2(n_px.max()))), -1, -1):
        cell_size = 2**k
        coarse_shape = -(-n_px // cell_size)  # ceiling division
        grid_px = coarse_shape * cell_size
        if np.prod(grid_px) <= (1 + max_extra_fraction) * np.prod(n_px):
            break
    pad = (grid_px - n_px) // 2
    return (
        (int(coarse_shape[0]), int(coarse_shape[1])),
        cell_size,
        (int(pad[0]), int(pad[1])),
    )


class Mochi(Martini):
    """
    Creates synthetic HI data cubes from simulation data.

    ??.

    Parameters
    ----------
    source : SPHSource
        An instance of a class derived from
        :class:`~martini.sources.sph_source.SPHSource`.
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). See :doc:`sub-module documentation </sources/index>`.

    datacube : ~martini.datacube.DataCube
        A :class:`~martini.datacube.DataCube` instance.
        A description of the datacube to create, including pixels, channels,
        sky position. See :doc:`sub-module documentation </datacube/index>`. The number
        of pixels along each spatial axis must be even.

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

    interpolant : Callable
        Function that interpolates the gas particle (or gas cell) properties onto the
        grid. E.g. :func:`~martini.mochi.interpolants.sph` or
        :func:`~martini.mochi.interpolants.mfm` found in the
        :mod:`~martini.mochi.interpolants` module. With a
        :class:`~martini.sph_kernels.DiracDeltaKernel` use
        :func:`~martini.mochi.interpolants.particle_in_cell` (or
        :func:`~martini.mochi.interpolants.voronoi_mesh`).

    radiative_transfer : Callable, optional
        Function that evaluates a mock spectral cube (of mass in each pixel-channel
        cell). E.g. :func:`~martini.mochi.radiative_transfer.optically_thin`,
        can be found in the :mod:`~martini.mochi.radiative_transfer` module. If omitted,
        the optically thin method is adopted.

    refinement_strategy : Callable, optional
        Function that decides and applies the grid refinement criterion. E.g.
        :func:`~martini.mochi.refinement.refine_grid_to_half_particle_scale`, found in
        the :mod:`~martini.mochi.refinement` module. Ignored if ``adaptive_grid`` is
        ``False``.

    adaptive_grid : bool, optional
        If ``True`` (the default) the gas particle (or gas cell) properties are
        interpolated onto an adaptive-resolution grid. Otherwise a fixed-resolution grid
        is used.

    quiet : bool, optional
        If ``True``, suppress output to stdout.

    See Also
    --------
    martini.sources.sph_source.SPHSource
    martini.datacube.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models

    Examples
    --------
    ??
    """

    _REQUIRES_LOS_DISTANCE_PIXCOORDS = True

    def __init__(
        self,
        *,
        source: SPHSource,
        datacube: DataCube,
        beam: _BaseBeam | None = None,
        noise: _BaseNoise | None = None,
        sph_kernel: _BaseSPHKernel,
        spectral_model: _BaseSpectrum,
        interpolant: Callable,  # fill in arg & return types
        radiative_transfer: Callable[
            [CellGrid, DataCube, _BaseSpectrum, type], U.Quantity[U.Msun]
        ] = optically_thin,
        refinement_strategy: Callable[
            [np.ndarray, U.Quantity[U.pix], U.Quantity[U.pix]], np.ndarray
        ] = refine_grid_to_half_particle_scale,
        adaptive_grid: bool = True,
        quiet: bool = False,
    ) -> None:
        if datacube.n_px_x % 2 or datacube.n_px_y % 2:
            raise ValueError(
                "Mochi requires an even number of pixels along each spatial axis of the "
                f"datacube, got n_px_x={datacube.n_px_x}, n_px_y={datacube.n_px_y}."
            )
        is_dirac_delta = isinstance(sph_kernel, DiracDeltaKernel)
        if is_dirac_delta and interpolant in (
            interpolants.sph,
            interpolants.mfm,
            interpolants.manual_sph,
        ):
            raise ValueError(
                "Kernel interpolants cannot sample a DiracDeltaKernel, use "
                "interpolant=martini.mochi.interpolants.particle_in_cell "
                "(or voronoi_mesh)."
            )
        if not is_dirac_delta and interpolant is interpolants.particle_in_cell:
            warnings.warn(
                "The particle_in_cell interpolant ignores the SPH kernel and deposits "
                "particles as points, use DiracDeltaKernel to make this explicit."
            )
        super().__init__(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
            quiet=quiet,
        )
        self.interpolant = interpolant
        self.radiative_transfer = radiative_transfer
        self.refinement_strategy = refinement_strategy
        self.adaptive_grid = adaptive_grid
        self._init_coarse_grid()

        return

    def _init_coarse_grid(self) -> None:
        """
        Initialize datacube and interpolation grid for a coarse shape.

        Since MOCHI uses dynamic resolution based on bisection and user input cube shape
        may not allow factorisation, the cube is padded such that it is covered by a
        coarse grid of cells with power-of-two side lengths in pixels. Bisected cells
        then remain aligned with the pixel grid.
        Starting from a minimal resolution also reduces the amount of cell-particle
        distance computations.
        """
        if not self.adaptive_grid:
            return
        shape, self.initial_cell_size, pad_coarse = _coarse_shape(
            (self._datacube.datacube_shape[0], self._datacube.datacube_shape[1])
        )
        pad = (pad_coarse[0] + self.datacube.padx, pad_coarse[1] + self.datacube.pady)
        # Ideally this padding calculation should happen before the datacube array is
        # actually allocated. The shape is available before that. Otherwise we end up
        # allocating a larger cube, copying the smaller cube into it, then deleting the
        # smaller cube, which is wasted time and memory. Maybe this is already the case,
        # but should check.
        self.datacube.drop_pad()
        self.datacube.add_pad(pad)
        self.initial_grid_shape = (shape[0],) + shape  # los direction as z -- needs fix

    def insert_source_in_cube(
        self,
        skip_validation: bool = False,
        progressbar: bool | None = None,
        ncpu: int = 1,
        mem_lim_GB: float = 4.0,
    ) -> None:
        """
        Populate the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the :class:`~martini.datacube.DataCube`
            is approximated for increased speed. For some combinations of pixel size,
            distance and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            ``RuntimeError`` if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter ``True``.

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If :class:`~martini.martini.Martini` was
            initialised with ``quiet`` set to ``True``, progress bars are switched off
            unless explicitly turned on.

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the :mod:`multiprocess` module (n.b. not the same as
            ``multiprocessing``).
        """
        # need to revise irrelevant (?) arguments: skip_validation, progressbar, ncpu
        # TODO: resolution checks. Kernels are point-sampled at cell centres, so accuracy
        # depends on cell size relative to the kernel FWHM, not on pixel size as for
        # Martini's kernel integrals. Refinement (e.g. refine_grid_to_half_particle_scale)
        # refines relative to the radius of compact support, so cells end up between ~0.6
        # (Gaussian) and ~1.0 (WendlandC6) FWHM across, and the uniform grid has fixed
        # 1 px cells. Martini's kernel validation mechanism (_validate /
        # _confirm_validation with min_valid_size, honouring skip_validation) could be
        # used to check that the refinement strategy yields a fine enough grid for the
        # kernel at hand, e.g. by validating smoothing lengths in units of the sizes of
        # the cells that each particle deposits into (with Mochi-specific tolerances).
        self.sph_kernel._init_sm_ranges()
        cell_grid = (  # do any options need exposing?
            AdaptiveCellGrid(
                self.source,
                self._datacube,
                self.initial_grid_shape,
                initial_cell_size=self.initial_cell_size,
            )
            if self.adaptive_grid
            else CellGrid(self.source, self._datacube)
        )
        cell_grid.init_particle_locations(self.source, self.sph_kernel)
        if hasattr(cell_grid, "eval_grid_refinement"):
            cell_grid.eval_grid_refinement(
                self.source, self._datacube, self.refinement_strategy
            )
        cell_grid.interpolate_fields(self.source, self.sph_kernel, self.interpolant)

        datacube_array = cell_grid.eval_radiative_transfer(
            # haven't exposed the index datatype option
            self._datacube,
            self.spectral_model,
            self.radiative_transfer,
        )
        # can the unit conversion be merged with martini?
        # TODO: use distances of individual cells instead of the source distance. Cells
        # nearer/further along the LoS (and their volumes, via the distortion
        # corrections) are at different distances, so their flux should scale with
        # their own distance**-2.
        datacube_array *= self.source.distance.to(U.Mpc) ** -2
        datacube_array /= np.abs(np.diff(self._datacube.velocity_channel_edges)).to(
            U.km / U.s
        )[:, np.newaxis, np.newaxis]

        if np.isnan(datacube_array).any():
            raise RuntimeError(
                "NaN values in the MOCHI spectral cube, this indicates a bug (e.g. a "
                "kernel that the interpolant cannot evaluate)."
            )
        MHI_to_Jy_inplace(datacube_array)
        datacube_array /= U.pix**2
        datacube_array = datacube_array.to(
            U.Jy / U.arcsec**2, equivalencies=[self._datacube.arcsec2_to_pix]
        )
        datacube_array = np.flip(datacube_array, 2)
        datacube_array = np.moveaxis(
            datacube_array,
            (0, 1, 2),
            (2, 1, 0),
        )
        self._datacube._array = datacube_array
        return
