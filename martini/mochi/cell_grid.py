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


class CellGrid:
    """
    Manage interpolating fields onto a fixed grid to create a spectral cube.

    Parameters
    ----------
    source : ~martini.sources.sph_source.SPHSource
        The source module containing the gas particles or cells.

    datacube : ~martini.datacube.DataCube
        The target data cube object for the spectral cube.

    grid_size : int or None
        The number of cells along each axis of the grid. If ``None``, it is determined
        from the ``datacube`` dimensions.
    """

    cells: np.ndarray
    pix_range: list[tuple[int, int]]
    positions: U.Quantity[U.pix]
    radii: U.Quantity[U.pix]
    interpolated_fields: dict[str, U.Quantity]
    cell_volumes: U.Quantity[U.pix**3]
    cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled]
    cell_centres: U.Quantity[U.pix]
    grid_shape: tuple[int, int, int]

    def __init__(
        self,
        source: SPHSource,
        datacube: DataCube,
        grid_shape: tuple[int, int, int] | int | None = None,
    ) -> None:
        datacube_shape = datacube.datacube_shape[:3]
        full_grid_shape: tuple[int, int, int]
        if grid_shape is None:
            full_grid_shape = (
                datacube_shape[0],  # copy ra dim length into los dim length
                datacube_shape[0],
                datacube_shape[1],
            )
        elif type(grid_shape) is int:
            full_grid_shape = (grid_shape, grid_shape, grid_shape)
        else:
            full_grid_shape = grid_shape
        size = datacube_shape[0] / full_grid_shape[1]  # offset because 0 is LOS
        size_y = datacube_shape[1] / full_grid_shape[2]
        assert size == size_y  # Enforce cuboid cells

        self.pix_range = [(0, n_px) for n_px in full_grid_shape]

        self.cells = np.array(
            [
                (x, y, z, size)
                for x in (np.arange(*self.pix_range[0], dtype=float) * size) - 0.5
                for y in (np.arange(*self.pix_range[1], dtype=float) * size) - 0.5
                for z in (np.arange(*self.pix_range[2], dtype=float) * size) - 0.5
            ],
            dtype=_CELL_DTYPE,
        )
        self.init_cell_centres()
        self.grid_shape = full_grid_shape
        self.init_cell_volumes()
        self.init_cell_distortion_corrections(source, datacube)

    def init_particle_locations(
        self,
        source: SPHSource,
        sph_kernel: _BaseSPHKernel,
    ) -> None:
        """
        Set up an array of particle locations in the pixel grid.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module containing the gas particles or cells.

        sph_kernel : _BaseSPHKernel
            The module specifying the SPH kernel.

        """
        # Perhaps undesirable copy of the pixel coordinates, but awkward to avoid.
        self.positions = np.column_stack(
            (
                source.los_distance_pixcoords,
                source.pixcoords[0],
                source.pixcoords[1],
            )
        )
        self.radii = sph_kernel.sm_ranges

    def init_cell_centres(self, prefix: str = "") -> None:
        """
        Initialize an array with the central coordinates of each cell.

        The "raw" cell data is encoded as lower corners and sizes, calculate the centre
        as offset 0.5x the size from the corner along each axis. The centres are stored
        with units.

        Parameters
        ----------
        prefix : str
            By default sets ``self.cell_centres`` from the ``self.cells``. A prefix, e.g.
            ``"adaptive"`` can be provided, which would set ``self.adaptive_cell_centres``
            from ``self.adaptive_cells``.
        """
        if prefix != "":
            prefix = f"{prefix}_"
        setattr(
            self,
            f"{prefix}cell_centres",
            (
                (
                    np.column_stack(
                        (
                            getattr(self, f"{prefix}cells")["x"],
                            getattr(self, f"{prefix}cells")["y"],
                            getattr(self, f"{prefix}cells")["z"],
                        )
                    )
                )
                + getattr(self, f"{prefix}cells")["size"][:, None] / 2
            )
            * U.pix,
        )
        return

    def init_cell_volumes(self, prefix: str = "") -> None:
        """
        Initialize the rectilinear volume of each cell.

        The "raw" cell data is encoded as lower corners and sizes. The cells are cubic in
        pixel coordinates, so their rectilinear volume is their side length cubed. See
        :meth:`init_cell_distortion_corrections` for the correction to account for the
        spherical geometry of the RA-Dec-distance grid.

        Parameters
        ----------
        prefix : str
            By default sets ``self.cell_volumes`` from the ``self.cells``. A prefix, e.g.
            ``"adaptive"`` can be provided, which would set ``self.adaptive_cell_volumes``
            from ``self.adaptive_cells``.
        """
        if prefix != "":
            prefix = f"{prefix}_"
        setattr(
            self,
            f"{prefix}cell_volumes",
            (getattr(self, f"{prefix}cells")["size"] * U.pix) ** 3,
        )
        return

    def init_cell_distortion_corrections(
        self, source: SPHSource, datacube: DataCube, prefix: str = ""
    ) -> None:
        """
        Initialize the spherical distortion correction of each cell volume.

        The cell grid is in RA-Dec-distance space. We assume that the angular size of each
        individual cell is small, so cells are locally cubic, but the grid may cover a
        total angular extent that is not small. The volume of each cell is the product of
        its rectilinear volume (see :meth:`init_cell_volumes`) and this dimensionless
        correction. Cells at declinations away from the equator have their volumes
        reduced by a factor of the cosine of the declination. The correction is unity for
        a cell centred on the grid centre along the LoS direction at declination zero.
        Cells closer and further away than this reference distance have their volumes
        adjusted by the cube of the ratio of their distance from the observer relative to
        the reference distance.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module, needed for reference distance.

        datacube : ~martini.datacube.DataCube
            The datacube module, needed for WCS and grid spacing information.

        prefix : str
            By default sets ``self.cell_distortion_corrections`` from the
            ``self.cells``. A prefix, e.g. ``"adaptive"`` can be provided, which would set
            ``self.adaptive_cell_distortion_corrections`` from ``self.adaptive_cells``.
        """
        if prefix != "":
            prefix = f"{prefix}_"
        # all_pix2world gives values in degrees:
        _, cell_dec = (
            datacube.wcs.sub(2)
            .all_pix2world(
                getattr(self, f"{prefix}cell_centres")[:, :2],
                1,
            )
            .T
        )
        ref_cell = (datacube.datacube_shape[0] / 2 - 0.5) * U.pix
        dist_offsets = (
            (getattr(self, f"{prefix}cell_centres")[:, 0] - ref_cell)
            / U.pix
            * datacube.px_size
            * source.distance
        ).to(source.distance.unit, U.dimensionless_angles())
        size_correction = ((dist_offsets + source.distance) / source.distance).to_value(
            U.dimensionless_unscaled
        )
        dec_correction = np.cos(np.radians(cell_dec))
        if (size_correction < 0).any():
            raise RuntimeError(
                "Observer must not be inside MOCHI grid (insufficient source distance)."
            )
        setattr(
            self,
            f"{prefix}cell_distortion_corrections",
            dec_correction * size_correction**3,
        )
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
        return self._interpolate_fields(
            source,
            sph_kernel,
            interpolant,
            self.cell_centres,
            self.cell_volumes,
            self.cell_distortion_corrections,
        )

    def _interpolate_fields(
        self,
        source: SPHSource,
        sph_kernel: _BaseSPHKernel,
        interpolant: Callable,  # fill in arg & return types
        cell_centres: U.Quantity[U.pix],
        cell_volumes: U.Quantity[U.pix**3],
        cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled],
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

        cell_centres : ~astropy.units.Quantity
            Centre coordinates of cells to use in interpolation, with units of pixels.

        cell_volumes : ~astropy.units.Quantity
            Rectilinear cell volumes to use in interpolation, with units of pixels cubed.

        cell_distortion_corrections : ~astropy.units.Quantity
            Dimensionless spherical distortion corrections to the cell volumes.
        """
        if source.skycoords is None:
            raise RuntimeError(
                "Initialize source skycoords before interpolating fields."
            )
        self.interpolated_fields = interpolant(
            positions=self.positions,
            smoothing_lengths=self.radii,
            field_positions=cell_centres,
            d_volume=cell_volumes,
            distortion_corrections=cell_distortion_corrections,
            kernel=sph_kernel.kernel,
            velocities=source.skycoords.radial_velocity,
            masses_HI=source.mHI_g,
            temperatures=source.T_g * C.k_B / C.m_p,  # approx thermal energy?
            masses=source.mHI_g,  # mass goes here, crude approx for now
        )

    def eval_radiative_transfer(
        self,
        datacube: DataCube,
        spectral_model: _BaseSpectrum,
        radiative_transfer: Callable,  # fill in arg & return types
    ) -> U.Quantity[U.Msun]:
        """
        "Collapse" the cell grid into a spectral cube.

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


class AdaptiveCellGrid(CellGrid):
    """
    Manage interpolating fields onto an adaptive grid to create a spectral cube.

    Parameters
    ----------
    source : ~martini.sources.sph_source.SPHSource
        The source module containing the gas particles or cells.

    datacube : ~martini.datacube.DataCube
        The target data cube object for the spectral cube.

    initial_grid_size : int
        The initial number of cells along each axis of the adaptive grid.
    """

    initial_cells: np.ndarray
    adaptive_cells: np.ndarray
    adaptive_cell_volumes: U.Quantity[U.pix**3]
    adaptive_cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled]
    adaptive_cell_centres: U.Quantity[U.pix]

    def __init__(
        self,
        source: SPHSource,
        datacube: DataCube,
        initial_grid_size: int | tuple[int, int, int] = 2,
    ) -> None:
        super().__init__(source, datacube, grid_shape=initial_grid_size)
        self.initial_cells = self.cells

    def eval_grid_refinement(
        self,
        source: SPHSource,
        datacube: DataCube,
        refinement_strategy: Callable[
            [np.ndarray, U.Quantity[U.pix], U.Quantity[U.pix]], np.ndarray
        ],
    ) -> None:
        """
        Walk the initial cell grid and refine where required to make an adaptive grid.

        ??.

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module containing the gas particles or cells.

        datacube : ~martini.datacube.DataCube
            The target data cube object for the spectral cube.

        refinement_strategy : Callable
            The method to decide and apply the refinement criterion.
        """
        self.adaptive_cells = refinement_strategy(
            self.initial_cells,
            self.positions.to_value(U.pix),
            self.radii.to_value(U.pix),
        ).view(dtype=_CELL_DTYPE)
        self.init_cell_centres(prefix="adaptive")
        self.init_cell_volumes(prefix="adaptive")
        self.init_cell_distortion_corrections(source, datacube, prefix="adaptive")

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
        return self._interpolate_fields(
            source,
            sph_kernel,
            interpolant,
            self.adaptive_cell_centres,
            self.adaptive_cell_volumes,
            self.adaptive_cell_distortion_corrections,
        )
