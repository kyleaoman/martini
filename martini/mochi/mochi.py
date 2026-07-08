"""
Create mock observations of HI sources from cosmological hydrodynamical simulations.

Provides :class:`~martini.martini.Mochi`, the main class of the package.
"""

import numpy as np
import cv2
from astropy import units as U, constants as C
from martini.martini import Martini
from martini.datacube import DataCube
from martini.beams import _BaseBeam
from martini.sources import SPHSource
from martini.sph_kernels import _BaseSPHKernel
from martini.spectral_models import _BaseSpectrum
from martini.noise import _BaseNoise
from typing import Callable
from functools import partial
from . import interpolants, radiative_transfer


CELL_DTYPE = [("x", float), ("y", float), ("z", float), ("size", float)]
RF = np.sqrt(3) / 2


def resize(cube: U.Quantity, target_shape: tuple[int]) -> U.Quantity:
    """
    Resize a data cube to the target shape.

    Interpolation is handled with :func:`cv2.resize`.

    Parameters
    ----------
    cube : ~astropy.units.Quantity
        The cube to be resized.

    target_shape : tuple
        The desired shape as a 3-tuple.

    Returns
    -------
    ~astropy.units.Quantity
        The cube interpolated into the new desired shape using :func:`cv2.resize`.
    """
    if np.all(np.array(cube.shape[1:]) == np.array(target_shape)):
        return cube
    target_shape = tuple(target_shape)
    unit = cube.unit
    unitless_cube = cube.to_value(cube.unit)
    result = np.zeros((cube.shape[0],) + target_shape)
    for i in range(cube.shape[0]):
        result[i] = cv2.resize(
            unitless_cube[i].astype(float),
            target_shape[::-1],
            interpolation=cv2.INTER_AREA,
        )
    return result * np.prod(cube.shape[1:]) / np.prod(target_shape) * unit


def _refine_grid_bisect(
    cell: np.ndarray,
    mask: np.ndarray,
    in_cell: np.ndarray,
    new_cells: list[tuple[float]],
    new_cells_over: list[bool],
    new_cells_masks: list,
) -> None:
    """
    Bisect operation for grid refinement algorithms.

    Parameters
    ----------
    cell : ~numpy.ndarray
        Lower left coordinate and size of cell. Coordinate elements can be accessed as
        ``cell["x"]``, etc., and size as ``cell["size"]``.

    mask : ??
        ??

    in_cell : ~numpy.ndarray
        Flag for each particle indicating whether it is in this cell.

    new_cells : list
        ??

    new_cells_over : list
        ??

    new_cells_mask : list
        ??
    """
    new_size = cell["size"] / 2.0
    new_cells.extend(
        [
            (
                cell["x"] + dx * new_size,
                cell["y"] + dy * new_size,
                cell["z"] + dz * new_size,
                new_size,
            )
            for dx in range(2)
            for dy in range(2)
            for dz in range(2)
        ]
    )
    new_cells_over.extend([False] * 8)
    new_cells_masks.extend([mask[in_cell]] * 8)
    return


def _pass_complete_cell(cells_lists: list, content_list: list) -> None:
    """
    Append new cells to a cell list.

    Parameters
    ----------
    cell_lists : list
        ??

    content_list : list
        ??
    """
    for i in range(len(cells_lists)):
        cells_lists[i].append(content_list[i])


def refine_grid(
    in_cell_function: Callable,  # fill in arg & return types
    bisect_condition: Callable,  # fill in arg & return types
    cells: np.ndarray,
    positions: U.Quantity[U.pix],
    radii: U.Quantity[U.pix],
    threshold: float,
    stop_iter: int = 8,
) -> np.ndarray:
    """
    Start from a coarse grid, refine until no cells satisfy ``bisect_condition``.

    Parameters
    ----------
    in_cell_function: Callable
        Delimits the particles to consider for the cell.
    bisect_condition: Callable
        If returns ``True``, the cell is bisected.
    cells: ~numpy.ndarray
        Contains entries with the 3D position of the lower corner (accessible as
        ``cells["x"]``, etc.) and sizes (accessible as ``cells["size"]``) of the initial
        cells.
    positions: ~astropy.units.Quantity
        Array of particle positions with units of pixels.
    radii: ~astropy.units.Quantity
        Array of particle smoothing lengths with units of pixels.
    threshold:
        Sensitivity threshold used by ``bisect_condition``.

    Returns
    -------
    ~numpy.ndarray
        Contains entries with the 3D position of the lower corner (accessible as
        ``cells["x"]``, etc.) and sizes (accessible as ``cells["size"]``) of the refined
        cell grid.
    """
    cells_number = len(cells)
    cells_over = np.zeros(cells_number, dtype=bool)
    cells_masks = [np.arange(len(radii))] * cells_number
    new_cells = []
    new_cells_over = []
    new_cells_masks = []

    iter = 0
    while iter < stop_iter:
        for n in range(cells_number):
            cell = cells[n]
            if cells_over[n]:
                _pass_complete_cell(
                    [new_cells, new_cells_over, new_cells_masks], [cells[n], True, True]
                )
                continue
            in_cell = in_cell_function(
                cells_masks[n], positions, radii, cell, threshold
            )
            if bisect_condition(
                cell["size"], in_cell, threshold, radii[cells_masks[n]]
            ):
                _refine_grid_bisect(
                    cell,
                    cells_masks[n],
                    in_cell,
                    new_cells,
                    new_cells_over,
                    new_cells_masks,
                )
            else:
                _pass_complete_cell(
                    [new_cells, new_cells_over, new_cells_masks], [cells[n], True, True]
                )
        cells = np.array(new_cells, dtype=CELL_DTYPE)

        if len(cells) == cells_number or iter == stop_iter:
            break
        cells_number = len(cells)
        cells_over = new_cells_over
        cells_masks = new_cells_masks
        new_cells = []
        new_cells_over = []
        new_cells_masks = []
        iter += 1
    return cells


def occupancy_in_cell(
    mask: np.ndarray,
    particles_pos: U.Quantity[U.pix],
    particles_radii: U.Quantity[U.pix],
    cell: np.void,
    threshold: float,  # is this needed by this or any other occupancy function?
) -> np.ndarray:
    """
    Describe.

    ??

    Parameters
    ----------
    mask : ~numpy.ndarray
        ??

    particles_pos : ~astropy.units.Quantity
        Particle locations in the cell grid with units of pixels.

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.

    cell : ~numpy.void
        The cell to consider. Cells are encoded as the coordinates of the lower corner
        (accessible as ``cell["x"]``, etc.) and the side length (``cell["size"]``).

    threshold : float
        ??
    """
    return (
        np.sum(
            np.abs(
                particles_pos[mask]
                - [cell["x"], cell["y"], cell["z"]]
                - cell["size"] / 2
            ),
            axis=1,
        )
        < cell["size"] * 2
    )


def is_not_single_occupancy(
    cell_size: float,
    in_cell: np.ndarray,
    threshold: float,
    particles_radii: U.Quantity[U.pix],
) -> bool:
    """
    Describe.

    ??

    Parameters
    ----------
    cell_size : float
        ??

    in_cell : ~numpy.ndarray
        ??

    threshold : float
        ??

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.
    """
    count = np.sum(in_cell)
    return count > threshold


refine_grid_to_occupancy = partial(
    refine_grid, occupancy_in_cell, is_not_single_occupancy
)


def intersect_in_cell(
    mask: np.ndarray,
    particles_pos: U.Quantity[U.pix],
    particles_radii: U.Quantity[U.pix],
    cell: np.void,
    threshold: float,  # is this needed by this or any other occupancy function?
) -> np.ndarray:
    """
    Describe.

    ??

    Parameters
    ----------
    mask : ~numpy.ndarray
        ??

    particles_pos : ~astropy.units.Quantity
        Particle locations in the cell grid with units of pixels.

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.

    cell : ~numpy.void
        The cell to consider. Cells are encoded as the coordinates of the lower corner
        (accessible as ``cell["x"]``, etc.) and the side length (``cell["size"]``).

    threshold : float
        ??
    """
    small_particle = (
        particles_radii[mask] * threshold < cell["size"]
    )  # No need to consider particles larger than cell
    return (
        np.linalg.norm(
            particles_pos[mask] - [cell["x"], cell["y"], cell["z"]] - cell["size"] / 2,
            axis=1,
        )
        < particles_radii[mask] + cell["size"] * RF
    ) & small_particle


def is_any_particle_included(
    cell_size: float,
    in_cell: np.ndarray,
    threshold: float,
    particles_radii: U.Quantity[U.pix],
) -> bool:
    """
    Describe.

    ??

    Parameters
    ----------
    cell_size : float
        ??

    in_cell : ~numpy.ndarray
        ??

    threshold : float
        ??

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.
    """
    return np.any(in_cell)


refine_grid_to_particle_scale = partial(
    refine_grid, intersect_in_cell, is_any_particle_included
)


class AdaptiveCellGrid:
    """
    Manage interpolating fields onto an adaptive grid to create a spectral cube.

    Parameters
    ----------
    datacube : ~martini.datacube.DataCube
        The target data cube object for the spectral cube.

    initial_grid_size : int
        The initial number of cells along each axis of the adaptive grid.
    """

    initial_cells: np.ndarray
    adaptive_cells: np.ndarray
    pix_range: list[tuple[int]]
    positions: U.Quantity[U.pix]
    radii: U.Quantity[U.pix]
    field_velocity: U.Quantity[U.km / U.s]
    field_mHI: U.Quantity[U.Msun]
    field_temperature: U.Quantity[U.K]
    final_cell_volume: U.Quantity[U.pix]
    final_grid_shape: tuple[int]

    def __init__(
        self,
        datacube: DataCube,
        initial_grid_size: int = 2,
    ) -> None:
        assert datacube.n_px_x == datacube.n_px_y
        assert datacube.padx == datacube.pady
        # For now we are restricted to a cube (not cuboid) voxel grid. Means we're padding
        # the z-direction, should amend this later.
        self.pix_range = [(0, datacube.n_px_x + 2 * datacube.padx)] * 3
        size = (
            self.pix_range[0][1] - self.pix_range[0][0]
        ) / initial_grid_size  # cubic!
        self.initial_cells = np.array(
            [
                (x, y, z, size)
                for x in np.linspace(
                    *self.pix_range[0], initial_grid_size, endpoint=False
                )
                for y in np.linspace(
                    *self.pix_range[1], initial_grid_size, endpoint=False
                )
                for z in np.linspace(
                    *self.pix_range[2], initial_grid_size, endpoint=False
                )
            ],
            dtype=CELL_DTYPE,
        )

    def init_particle_locations(
        self,
        source: SPHSource,
        sph_kernel: _BaseSPHKernel,
        min_radius: U.Quantity[U.pix] = 0.005 * U.pix,
    ) -> None:
        """
        Set up an array of particle locations in the pixel grid.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module containing the gas particles or cells.

        sph_kernel : _BaseSPHKernel
            The module specifying the SPH kernel.

        min_radius : ~astropy.units.Quantity
            Floor to apply to smoothing length values to avoid extremely deep refinement.
        """
        self.positions = np.column_stack(
            (
                source.los_pixcoords,
                source.pixcoords[0],
                source.pixcoords[1],
            )
        )
        self.radii = np.clip(sph_kernel.sm_ranges, min_radius, np.inf)

    def eval_grid_refinement(
        self,
        refine_algorithm: Callable[
            [np.ndarray, U.Quantity[U.pix], U.Quantity[U.pix], float], np.ndarray
        ] = refine_grid_to_particle_scale,
        threshold: float = 0.5,
    ) -> None:
        """
        Walk the initial cell grid and refine where required to make an adaptive grid.

        ??

        Parameters
        ----------
        refine_algorithm : Callable
            The method to decide and apply the refinement criterion.

        threshold : float
            ??
        """
        self.adaptive_cells = refine_algorithm(
            self.initial_cells,
            self.positions.to_value(U.pix),
            self.radii.to_value(U.pix),
            threshold=threshold,
        ).view(dtype=CELL_DTYPE)
        self.init_cell_centres()
        self.init_cell_volumes()

    def init_cell_centres(self) -> None:
        """
        Initialize an array with the central coordinates of each cell.

        The "raw" cell data is encoded as lower corners and sizes, calculate the centre
        as offset 0.5x the size from the corner along each axis. The centres are stored
        with units.
        """
        self.cell_centres = (
            np.column_stack(
                (
                    self.adaptive_cells["x"],
                    self.adaptive_cells["y"],
                    self.adaptive_cells["z"],
                )
            )
            + self.adaptive_cells["size"][:, np.newaxis] / 2
        ) * U.pix
        return

    def init_cell_volumes(self) -> None:
        """
        Initialize an array with the volume of each cell.

        The "raw" cell data is encoded as lower corners and sizes, calculate the volume
        as the cube of the side length.
        """
        # needs revision to be spherical
        self.cell_volumes = (self.adaptive_cells["size"] * U.pix) ** 3
        return

    def interpolate_fields(
        self,
        source: SPHSource,
        sph_kernel: _BaseSPHKernel,
        interpolant: Callable = interpolants.SPH,  # fill in arg & return types
    ) -> None:
        """
        Interpolate particle-carried fields onto the grid.

        ??

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module with the gas particle or cell properties for interpolation.

        sph_kernel : _BaseSPHKernel
            The module specifying the SPH kernel.

        interpolant : Callable
            ??
        """
        self.field_velocity, self.field_mHI, self.field_temperature = interpolant(
            positions=self.positions,
            smoothing_lengths=self.radii,
            field_positions=self.cell_centres,
            d_volume=self.cell_volumes,
            kernel=sph_kernel.kernel,
            velocities=source.skycoords.radial_velocity,
            masses_HI=source.mHI_g,
            temperatures=source.T_g * C.k_B / C.m_p,  # approx thermal energy?
            masses=source.mHI_g
            / 0.7,  # mass goes here, crude approx for now (H fraction)
        )

    def create_regular_array(self, dtype: type = np.uintc) -> None:
        """
        Re-grid an adaptive cell grid onto a regular grid.

        Parameters
        ----------
        dtype : type
            Data type for the regular cell grid.
        """
        x0 = np.min(self.adaptive_cells["x"])
        y0 = np.min(self.adaptive_cells["y"])
        z0 = np.min(self.adaptive_cells["z"])
        xyz_0 = np.array([x0, y0, z0])
        xyz_cells = np.column_stack([self.adaptive_cells[i] for i in "xyz"])
        dx = np.min(self.adaptive_cells["size"])
        self.final_grid_shape = [
            int((ax_range[1] - ax_range[0]) // dx) for ax_range in self.pix_range
        ]
        N = len(self.adaptive_cells)
        cell_range = np.arange(N, dtype=dtype)
        grid = np.empty(self.final_grid_shape, dtype=dtype)
        cells_begin = np.round((xyz_cells - xyz_0) / dx).astype(int)
        cells_end = np.round(
            (xyz_cells - xyz_0 + self.adaptive_cells["size"][:, np.newaxis]) / dx
        ).astype(int)
        for i in cell_range:
            x_start, y_start, z_start = cells_begin[i]
            x_end, y_end, z_end = cells_end[i]
            grid[x_start:x_end, y_start:y_end, z_start:z_end] = i
        self.final_cell_volume = dx**3 * self.cell_volumes.unit

    def eval_radiative_transfer(
        self,
        datacube: DataCube,
        radiative_transfer_model: Callable = radiative_transfer.adaptiveOpticallyThin,  # fill in arg & return types
    ) -> U.Quantity[U.Msun]:
        """
        "Collapse" the regular cell grid into a spectral cube.

        Integrates out the third spatial axis while simultaneously creating the spectral
        axis.

        Parameters
        ----------
        radiative_transfer_model : Callable
            ??

        Returns
        -------
        ~astropy.units.Quantity
            The spectral cube with units of solar masses (i.e. HI contained in each pixel
            and channel).
        """
        return (
            radiative_transfer_model(  # eventually store as state instead of returning
                self.field_mHI,
                self.field_velocity,
                self.field_temperature,
                datacube,
                self.final_cell_volume,
                self.final_grid_shape,
                cells=self.adaptive_cells,
                cell_unit=U.pix,
            )
        )


class Mochi(Martini):
    """
    Creates synthetic HI data cubes from simulation data.

    ??

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

    def __init__(
        self,
        *,
        source: SPHSource,
        datacube: DataCube,
        beam: _BaseBeam | None = None,
        noise: _BaseNoise | None = None,
        sph_kernel: _BaseSPHKernel,
        spectral_model: _BaseSpectrum,
        quiet: bool = False,
    ) -> None:
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

    def insert_source_in_cube(
        self,
        skip_validation: bool = False,
        progressbar: bool | None = None,
        ncpu: int = 1,
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
        self.source._init_los_pixcoords(self.datacube)
        self.sph_kernel._init_sm_ranges()
        adaptive_cell_grid = AdaptiveCellGrid(self.datacube)
        adaptive_cell_grid.init_particle_locations(self.source, self.sph_kernel)
        adaptive_cell_grid.eval_grid_refinement()
        adaptive_cell_grid.interpolate_fields(self.source, self.sph_kernel)
        adaptive_cell_grid.create_regular_array()
        cube = adaptive_cell_grid.eval_radiative_transfer(self.datacube)
        cube *= (
            self.source.distance.to(U.Mpc) ** -2
        )  # use distances of individual cells instead
        cube /= self.datacube.channel_width.to(
            U.km / U.s
        )  # handle non-constant widths?

        def MHI_to_Jy_inplace(x: U.Quantity[U.Msun]) -> None:
            """
            Apply the HI mass to flux density conversion, with no memory overhead.

            The conversion is:
            M_HI/Msun = 2.36x10^5 * (D/Mpc)^2 * (S_21/Jy km s^-1)

            Parameters
            ----------
            x : ~astropy.units.Quantity
                :class:`~astropy.units.Quantity`, with dimensions of
                mass / length^2 / velocity.
            """
            # duplicated from spectral_models.py, refactor to have one def
            input_units = U.Msun * U.Mpc**-2 * (U.km * U.s**-1) ** -1
            np.divide(x, 2.36e5, out=x)
            x *= U.Jy / input_units
            return

        MHI_to_Jy_inplace(cube)
        cube /= U.pix**2
        target_shape = (
            self.datacube.n_px_x + 2 * self.datacube.padx,
            self.datacube.n_px_y + 2 * self.datacube.pady,
        )
        cube = resize(cube, target_shape)
        cube = np.flip(cube, 2)
        cube = np.moveaxis(
            cube,
            (0, 1, 2),
            (2, 1, 0),
        )
        insertion_slice = np.s_[..., 0] if self.datacube.stokes_axis else np.s_[...]
        self.datacube._array[insertion_slice] += cube
        return
