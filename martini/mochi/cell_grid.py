"""
Create mock observations of HI sources from cosmological hydrodynamical simulations.

Provides :class:`~martini.martini.Mochi`, the main class of the package.
"""

import numpy as np
from astropy import units as U, constants as C
from martini.datacube import DataCube
from martini.sources import SPHSource
from martini.sph_kernels import _BaseSPHKernel
from martini.spectral_models import _BaseSpectrum
from typing import Callable
from martini.mochi._dtypes import CELL_DTYPE as _CELL_DTYPE


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
    pix_range: list[tuple[int, int]]
    positions: U.Quantity[U.pix]
    radii: U.Quantity[U.pix]
    interpolated_fields: dict[str, U.Quantity]
    final_cell_volume: U.Quantity[U.pix]
    final_grid_shape: list[int]

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
            dtype=_CELL_DTYPE,
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
        refinement_strategy: Callable[
            [np.ndarray, U.Quantity[U.pix], U.Quantity[U.pix], float], np.ndarray
        ]
    ) -> None:
        """
        Walk the initial cell grid and refine where required to make an adaptive grid.

        ??.

        Parameters
        ----------
        refinement_strategy : Callable
            The method to decide and apply the refinement criterion.

        threshold : float
            ??.
        """
        self.adaptive_cells = refinement_strategy(
            self.initial_cells,
            self.positions.to_value(U.pix),
            self.radii.to_value(U.pix),
        ).view(dtype=_CELL_DTYPE)
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
        interpolant: Callable,  # fill in arg & return types
    ) -> None:
        """
        Interpolate particle-carried fields onto the grid.

        ??.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module with the gas particle or cell properties for interpolation.

        sph_kernel : _BaseSPHKernel
            The module specifying the SPH kernel.

        interpolant : Callable
            Function that interpolates the gas particle (or gas cell) properties onto the
            grid. E.g. :func:`~martini.mochi.interpolants.sph` or
            :func:`~martini.mochi.interpolants.mfm` found in the
            :mod:`~martini.mochi.interpolants` module.
        """
        if source.skycoords is None:
            raise RuntimeError(
                "Initialize source skycoords before interpolating fields."
            )
        self.interpolated_fields = interpolant(
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
        cell_range: np.ndarray = np.arange(N, dtype=dtype)
        grid: np.ndarray = np.empty(self.final_grid_shape, dtype=dtype)
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
        spectral_model: _BaseSpectrum,
        radiative_transfer: Callable,  # fill in arg & return types
    ) -> U.Quantity[U.Msun]:
        """
        "Collapse" the regular cell grid into a spectral cube.

        Integrates out the third spatial axis while simultaneously creating the spectral
        axis.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            Target data cube object, used to determine target shape and spectral binning.

        spectral_model : _BaseSpectrum
            Spectral model used to evaluate spectra.

        radiative_transfer : Callable
            Function that evaluates a mock spectral cube (of mass in each pixel-channel
            cell). E.g. :func:`~martini.mochi.radiative_transfer.adaptive_optically_thin`,
            can be found in the :mod:`~martini.mochi.radiative_transfer` module.

        Returns
        -------
        ~astropy.units.Quantity
            The spectral cube with units of solar masses (i.e. HI contained in each pixel
            and channel).
        """
        return radiative_transfer(  # eventually store as state instead of returning?
            self,
            datacube,
            spectral_model,
        )
