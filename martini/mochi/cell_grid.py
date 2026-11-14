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


def _pixel_solid_angle_ratio(
    datacube: DataCube, pix: np.ndarray, h: float = 0.01
) -> np.ndarray:
    """
    Evaluate the solid angle per unit pixel area relative to the nominal pixel size.

    The Jacobian of the datacube WCS is evaluated by central differences, so any
    celestial projection is supported. For the default TAN (gnomonic) projection the
    ratio is the cube of the cosine of the angular distance from the WCS reference
    point.

    Parameters
    ----------
    datacube : ~martini.datacube.DataCube
        The datacube module, needed for WCS information.

    pix : ~numpy.ndarray
        Pixel coordinates (indexed from 0) along the RA and Dec axes, with shape
        ``(N, 2)``.

    h : float, optional
        Half-width used to evaluate central differences, in pixels.

    Returns
    -------
    ~numpy.ndarray
        The dimensionless solid angle ratio at each location, with shape ``(N,)``.
    """
    celestial = datacube.wcs.celestial
    to_rad = [U.Unit(cunit).to(U.rad) for cunit in celestial.wcs.cunit]
    derivatives = []
    for step in ((h, 0), (0, h)):
        world_plus = celestial.all_pix2world(pix + step, 0) * to_rad
        world_minus = celestial.all_pix2world(pix - step, 0) * to_rad
        # wrap longitude differences across 2pi -> 0 in longitude:
        dlng = (
            world_plus[:, celestial.wcs.lng] - world_minus[:, celestial.wcs.lng] + np.pi
        ) % (2 * np.pi) - np.pi
        dlat = world_plus[:, celestial.wcs.lat] - world_minus[:, celestial.wcs.lat]
        derivatives.append((dlng / (2 * h), dlat / (2 * h)))
    (dlng_dx, dlat_dx), (dlng_dy, dlat_dy) = derivatives
    lat = (
        celestial.all_pix2world(pix, 0)[:, celestial.wcs.lat]
        * to_rad[celestial.wcs.lat]
    )
    solid_angle = np.abs(dlng_dx * dlat_dy - dlng_dy * dlat_dx) * np.cos(lat)
    return solid_angle / datacube.px_size.to_value(U.rad) ** 2


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

    cell_size : float or None
        The side length of the cells in pixels. If ``None``, it is determined such that
        the grid exactly covers the datacube pixel grid. The cell grid may extend
        beyond the datacube pixel grid if a larger size is given.
    """

    cells: np.ndarray
    pix_range: list[tuple[int, int]]
    positions: U.Quantity[U.pix]
    radii: U.Quantity[U.pix]
    interpolated_fields: dict[str, U.Quantity]
    cell_volumes: U.Quantity[U.pix**3]
    cell_distances: U.Quantity[U.Mpc]
    cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled]
    cell_centres: U.Quantity[U.pix]
    grid_shape: tuple[int, int, int]

    def __init__(
        self,
        source: SPHSource,
        datacube: DataCube,
        grid_shape: tuple[int, int, int] | int | None = None,
        cell_size: float | None = None,
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
        if cell_size is None:
            size = datacube_shape[0] / full_grid_shape[1]  # offset because 0 is LOS
            size_y = datacube_shape[1] / full_grid_shape[2]
            assert size == size_y  # Enforce cuboid cells
        else:
            size = cell_size

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
        self.init_cell_distances(source, datacube)
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

    def init_cell_distances(
        self, source: SPHSource, datacube: DataCube, prefix: str = ""
    ) -> None:
        """
        Initialize the distance from the observer to the centre of each cell.

        The LoS axis of the cell grid has the same number of pixels as the RA axis of
        the datacube, centred on the source distance. A cell centred on the grid centre
        along the LoS direction is at the source distance. Pixels are locally cubic, so
        the LoS pixel coordinate is logarithmic in distance (see
        :meth:`~martini.sources.sph_source.SPHSource._init_pixcoords`).

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module, needed for reference distance.

        datacube : ~martini.datacube.DataCube
            The datacube module, needed for grid spacing information.

        prefix : str
            By default sets ``self.cell_distances`` from the ``self.cells``. A prefix,
            e.g. ``"adaptive"`` can be provided, which would set
            ``self.adaptive_cell_distances`` from ``self.adaptive_cells``.
        """
        if prefix != "":
            prefix = f"{prefix}_"
        ref_cell = (datacube.datacube_shape[0] / 2 - 0.5) * U.pix
        log_distance_ratios = (
            (getattr(self, f"{prefix}cell_centres")[:, 0] - ref_cell)
            / U.pix
            * datacube.px_size
        ).to_value(U.rad)
        setattr(
            self,
            f"{prefix}cell_distances",
            source.distance * np.exp(log_distance_ratios),
        )
        return

    def init_cell_distortion_corrections(
        self, source: SPHSource, datacube: DataCube, prefix: str = ""
    ) -> None:
        """
        Initialize the spherical distortion correction of each cell volume.

        The cell grid is in projected sky-plane pixel coordinates (following the datacube
        WCS) and (logarithmic) distance. We assume that the angular size of each
        individual cell is small, so cells are locally cubic, but the grid may cover a
        total angular extent that is not small. The volume of each cell is the product of
        its rectilinear volume (see :meth:`init_cell_volumes`) and this dimensionless
        correction. The solid angle subtended by a pixel depends on the WCS projection,
        and is evaluated relative to the nominal pixel size. Cells have side lengths
        proportional to their distance from the observer, so their volumes are adjusted
        by the cube of the ratio of their distance relative to the source distance.
        Requires the cell distances to be initialized first (see
        :meth:`init_cell_distances`).

        Parameters
        ----------
        source : ~martini.sources.sph_source.SPHSource
            The source module, needed for reference distance.

        datacube : ~martini.datacube.DataCube
            The datacube module, needed for WCS information.

        prefix : str
            By default sets ``self.cell_distortion_corrections`` from the
            ``self.cells``. A prefix, e.g. ``"adaptive"`` can be provided, which would set
            ``self.adaptive_cell_distortion_corrections`` from ``self.adaptive_cells``.
        """
        if prefix != "":
            prefix = f"{prefix}_"
        # cell centres are (LoS, RA, Dec) pixel coordinates:
        angular_correction = _pixel_solid_angle_ratio(
            datacube, getattr(self, f"{prefix}cell_centres")[:, 1:].to_value(U.pix)
        )
        size_correction = (
            getattr(self, f"{prefix}cell_distances") / source.distance
        ).to_value(U.dimensionless_unscaled)
        setattr(
            self,
            f"{prefix}cell_distortion_corrections",
            angular_correction * size_correction**3 * U.dimensionless_unscaled,
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
            smoothing_lengths=sph_kernel.sm_lengths,
            field_positions=cell_centres,
            d_volume=cell_volumes,
            distortion_corrections=cell_distortion_corrections,
            kernel=sph_kernel,
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
    ) -> U.Quantity[U.Msun / U.Mpc**2]:
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
            Function that evaluates a mock spectral cube (of mass divided by distance
            squared in each pixel-channel cell). E.g.
            :func:`~martini.mochi.radiative_transfer.optically_thin`, can be found in the
            :mod:`~martini.mochi.radiative_transfer` module.

        Returns
        -------
        ~astropy.units.Quantity
            The spectral cube with units of solar masses per square megaparsec (i.e. HI
            mass contained in each pixel and channel, with each contribution divided by
            the square of its distance from the observer).
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

    initial_cell_size : float or None
        The side length of the initial cells in pixels. If ``None``, it is determined
        such that the grid exactly covers the datacube pixel grid.
    """

    initial_cells: np.ndarray
    adaptive_cells: np.ndarray
    adaptive_cell_volumes: U.Quantity[U.pix**3]
    adaptive_cell_distances: U.Quantity[U.Mpc]
    adaptive_cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled]
    adaptive_cell_centres: U.Quantity[U.pix]

    def __init__(
        self,
        source: SPHSource,
        datacube: DataCube,
        initial_grid_size: int | tuple[int, int, int] = 2,
        initial_cell_size: float | None = None,
    ) -> None:
        super().__init__(
            source, datacube, grid_shape=initial_grid_size, cell_size=initial_cell_size
        )
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
        self.init_cell_distances(source, datacube, prefix="adaptive")
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
