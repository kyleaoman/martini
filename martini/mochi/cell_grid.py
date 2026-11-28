"""
Create mock observations of HI sources from cosmological hydrodynamical simulations.

Provides :class:`~martini.martini.Mochi`, the main class of the package.
"""

import numpy as np
from astropy import units as U
from martini.datacube import DataCube
from martini.sources import SPHSource
from martini.sph_kernels import _BaseSPHKernel
from martini.spectral_models import _BaseSpectrum
from typing import Callable
from martini.mochi._dtypes import CELL_DTYPE as _CELL_DTYPE
from martini._grid_search import Octree
from martini.mochi.refinement import _octree_cells


def _distinct_nbytes(arrays: list[np.ndarray]) -> int:
    """
    Add up the memory of arrays, counting memory shared by several arrays once.

    Arrays that are views of another array are counted as the array that owns their
    memory.

    Parameters
    ----------
    arrays : list
        The arrays.

    Returns
    -------
    int
        The memory of the arrays, in bytes.
    """
    owners = {}
    for array in arrays:
        while isinstance(array.base, np.ndarray):
            array = array.base
        owners[id(array)] = array
    return sum(array.nbytes for array in owners.values())


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

    los_region : tuple[float, float]
        The edges of the region covered by the mock observation along the line of sight,
        in line-of-sight pixel coordinates of the particles. The grid may extend beyond
        this if required (e.g. divides into a non-integer number of cells).

    grid_shape : tuple[int, int, int] or int
        The number of cells along each axis of the grid, ordered (LoS, RA, Dec). The grid
        starts at the lower edge of the first pixel along each axis. Along the line of
        sight it is assumed to be centred on the source distance: the line-of-sight pixel
        coordinates of the particles must be centred on ``(los_extent - 1) / 2``, where
        ``los_extent = grid_shape[0] * cell_size`` is the extent of the grid along the
        line of sight in pixels.

    cell_size : float or None
        The side length of the cells in pixels. If ``None``, it is determined such that
        the grid exactly covers the datacube pixel grid.
    """

    cells: np.ndarray
    pix_range: list[tuple[int, int]]
    positions: U.Quantity[U.pix]
    radii: U.Quantity[U.pix]
    interpolated_fields: dict[str, U.Quantity]
    interpolated_extra_fields: dict[str, U.Quantity]
    cell_volumes: U.Quantity[U.pix**3]
    cell_distances: U.Quantity[U.Mpc]
    cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled]
    cell_centres: U.Quantity[U.pix]
    grid_shape: tuple[int, int, int]
    cell_size: float
    los_region: tuple[float, float]
    los_extent: float

    def __init__(
        self,
        source: SPHSource,
        datacube: DataCube,
        los_region: tuple[float, float],
        grid_shape: tuple[int, int, int] | int,
        cell_size: float | None = None,
    ) -> None:
        datacube_shape = datacube.datacube_shape[:3]
        self.los_region = los_region
        full_grid_shape: tuple[int, int, int] = (
            (grid_shape, grid_shape, grid_shape)
            if isinstance(grid_shape, int)
            else grid_shape
        )
        if cell_size is None:
            size = datacube_shape[0] / full_grid_shape[1]  # offset because 0 is LOS
            size_y = datacube_shape[1] / full_grid_shape[2]
            assert size == size_y  # Enforce cuboid cells
        else:
            size = cell_size
        self.los_extent = full_grid_shape[0] * size

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
        self.cell_size = size
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

        The cell grid is ``los_extent`` pixels long along the LoS axis, centred on the
        source distance. A cell centred on the centre of the grid along the LoS direction
        is at the source distance. Pixels are locally cubic, so
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
        ref_cell = (self.los_extent / 2 - 0.5) * U.pix
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

    def _arrays(self) -> list[np.ndarray]:
        """
        List the arrays held by the cell grid.

        Returns
        -------
        list
            The array attributes of the grid and the interpolated (and extra) fields.
        """
        return [
            value for value in vars(self).values() if isinstance(value, np.ndarray)
        ] + [
            value
            for fields in ("interpolated_fields", "interpolated_extra_fields")
            for value in getattr(self, fields, {}).values()
            if isinstance(value, np.ndarray)
        ]

    def _mem_usage_GB(self) -> float:
        """
        Get the memory held by the arrays of the cell grid.

        Memory shared by several arrays (e.g. views) is counted once.

        Returns
        -------
        float
            The memory, in GB.
        """
        return _distinct_nbytes(self._arrays()) / 1024**3

    @property
    def field_positions(self) -> U.Quantity[U.pix]:
        """
        Centre coordinates of the cells onto which fields are interpolated.

        Returns
        -------
        ~astropy.units.Quantity
            Cell centre coordinates, with units of pixels.
        """
        return self.cell_centres

    @property
    def field_volumes(self) -> U.Quantity[U.pix**3]:
        """
        Rectilinear volumes of the cells onto which fields are interpolated.

        Returns
        -------
        ~astropy.units.Quantity
            Cell volumes, with units of pixels cubed.
        """
        return self.cell_volumes

    @property
    def field_sizes(self) -> U.Quantity[U.pix]:
        """
        Side lengths of the cells onto which fields are interpolated.

        Returns
        -------
        ~astropy.units.Quantity
            Cell side lengths, with units of pixels.
        """
        return self.cells["size"] << U.pix

    @property
    def field_distortion_corrections(self) -> U.Quantity[U.dimensionless_unscaled]:
        """
        Spherical distortion corrections to the volumes of the cells.

        Returns
        -------
        ~astropy.units.Quantity
            Dimensionless distortion corrections to the cell volumes.
        """
        return self.cell_distortion_corrections

    @property
    def uniform_grid(self) -> tuple[tuple[int, int, int], float] | None:
        """
        Describe the uniform grid of cells onto which fields are interpolated.

        Interpolants can use this to search for cells efficiently. The cells are ordered
        as in an array of shape ``grid_shape`` (C-order), with centres at
        ``(index + 0.5) * cell_size - 0.5`` (pixels) along each axis.

        Returns
        -------
        tuple or None
            The shape of the grid and the size of the cells in pixels, ``(grid_shape,
            cell_size)``, or ``None`` if the cells are not a uniform grid.
        """
        return self.grid_shape, self.cell_size

    def interpolate_fields(
        self,
        source: SPHSource,
        sph_kernel: _BaseSPHKernel,
        interpolant: Callable,  # fill in arg & return types
        source_fields: tuple[str, ...] = (),
        **interpolant_kwargs: int | bool | float,
    ) -> None:
        """
        Interpolate particle-carried fields onto the grid.

        The interpolant receives this cell grid (``cell_grid`` argument) and uses its
        ``field_positions``, ``field_volumes``, ``field_distortion_corrections`` and
        ``uniform_grid`` (and ``octree`` of adaptive grids). The HI densities and
        velocities are stored in ``interpolated_fields``, the further source attributes
        requested in ``source_fields`` in ``interpolated_extra_fields``.

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
            :mod:`~martini.mochi.interpolants` module. It returns a tuple of two dicts:
            the interpolated HI densities (``"masses_HI"``) and velocities
            (``"velocities"``), and the interpolated ``extra_fields`` that it was given,
            under the same names.

        source_fields : tuple, optional
            Names of further attributes of the source to interpolate onto the grid (e.g.
            ``"T_g"``), as needed by the spectral model (see
            :meth:`~martini.spectral_models._BaseSpectrum.extra_data_source_fields`).

        **interpolant_kwargs : dict
            Additional keyword arguments passed to the interpolant (e.g. ``ncpu``).
        """
        if source.skycoords is None:
            raise RuntimeError(
                "Initialize source skycoords before interpolating fields."
            )
        fields, extra_fields = interpolant(
            positions=self.positions,
            smoothing_lengths=sph_kernel.sm_lengths,
            cell_grid=self,
            kernel=sph_kernel,
            velocities=source.skycoords.radial_velocity,
            masses_HI=source.mHI_g,
            extra_fields={name: getattr(source, name) for name in source_fields},
            masses=source.mHI_g,  # mass goes here, crude approx for now
            **interpolant_kwargs,
        )
        missing = [name for name in source_fields if name not in extra_fields]
        if missing:
            raise ValueError(
                f"The interpolant did not interpolate the requested fields {missing}."
            )
        self.interpolated_fields, self.interpolated_extra_fields = fields, extra_fields

    def eval_radiative_transfer(
        self,
        datacube: DataCube,
        spectral_model: _BaseSpectrum,
        radiative_transfer: Callable,  # fill in arg & return types
        ncpu: int = 1,
        mem_allowance_GB: float = np.inf,
        **radiative_transfer_kwargs: bool,
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
            :mod:`~martini.mochi.radiative_transfer` module. It is called with the cell
            grid, ``datacube`` and ``spectral_model``, and the keyword arguments ``ncpu``,
            the number of threads that it may use, and ``mem_allowance_GB``, the memory
            that it may use (in GB).

        ncpu : int, optional
            Number of threads passed to ``radiative_transfer``. A value of ``-1`` means
            as many threads as available cores.

        mem_allowance_GB : float, optional
            Memory allowance passed to ``radiative_transfer``, in GB.

        **radiative_transfer_kwargs : dict
            Additional keyword arguments passed to ``radiative_transfer`` (e.g.
            ``progressbar``).

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
            ncpu=ncpu,
            mem_allowance_GB=mem_allowance_GB,
            **radiative_transfer_kwargs,
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

    los_region : tuple[float, float]
        The edges of the region covered by the mock observation along the line of sight,
        in line-of-sight pixel coordinates of the particles. The grid may extend beyond
        this if required (e.g. divides into a non-integer number of cells).

    initial_grid_size : tuple[int, int, int] or int
        The initial number of cells along each axis of the adaptive grid, ordered (LoS,
        RA, Dec). The grid starts at the lower edge of the first pixel along each axis.
        Along the line of sight it is assumed to be centred on the source distance (see
        :class:`CellGrid`).

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
    _octree: Octree | None = None

    def __init__(
        self,
        source: SPHSource,
        datacube: DataCube,
        los_region: tuple[float, float],
        initial_grid_size: tuple[int, int, int] | int,
        initial_cell_size: float | None = None,
    ) -> None:
        super().__init__(
            source,
            datacube,
            los_region,
            grid_shape=initial_grid_size,
            cell_size=initial_cell_size,
        )
        self.initial_cells = self.cells

    def eval_grid_refinement(
        self,
        source: SPHSource,
        datacube: DataCube,
        refinement_strategy: Callable[..., Octree | np.ndarray],
        ncpu: int = 1,
        mem_allowance_GB: float = np.inf,
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
            The method to decide and apply the refinement criterion. It is called with
            the initial cells, and the particle positions and radii (smoothing ranges) in
            pixels, and the keyword arguments ``ncpu``, the number of threads that it may
            use, and ``mem_allowance_GB``, the memory that it may use (in GB). It returns
            the refined cells, either as an
            :class:`~martini._grid_search.Octree` (as the strategies in
            :mod:`~martini.mochi.refinement` do), which is then reused to search the
            cells, or as an array like ``initial_cells``, from which a tree is built when
            needed.

        ncpu : int, optional
            Number of threads passed to ``refinement_strategy``. A value of ``-1`` means
            as many threads as available cores.

        mem_allowance_GB : float, optional
            Memory allowance passed to ``refinement_strategy``, in GB.
        """
        refined = refinement_strategy(
            self.initial_cells,
            self.positions.to_value(U.pix),
            self.radii.to_value(U.pix),
            ncpu=ncpu,
            mem_allowance_GB=mem_allowance_GB,
        )
        if isinstance(refined, Octree):
            self.adaptive_cells = _octree_cells(refined)
            self._octree = refined
        else:
            self.adaptive_cells = refined.view(dtype=_CELL_DTYPE)
            self._octree = None  # built from the new cells when needed
        self.init_cell_centres(prefix="adaptive")
        self.init_cell_volumes(prefix="adaptive")
        self.init_cell_distances(source, datacube, prefix="adaptive")
        self.init_cell_distortion_corrections(source, datacube, prefix="adaptive")

    @property
    def field_positions(self) -> U.Quantity[U.pix]:
        """
        Centre coordinates of the adaptive cells onto which fields are interpolated.

        Returns
        -------
        ~astropy.units.Quantity
            Cell centre coordinates, with units of pixels.
        """
        return self.adaptive_cell_centres

    @property
    def field_volumes(self) -> U.Quantity[U.pix**3]:
        """
        Rectilinear volumes of the adaptive cells onto which fields are interpolated.

        Returns
        -------
        ~astropy.units.Quantity
            Cell volumes, with units of pixels cubed.
        """
        return self.adaptive_cell_volumes

    @property
    def field_sizes(self) -> U.Quantity[U.pix]:
        """
        Side lengths of the cells onto which fields are interpolated.

        Returns
        -------
        ~astropy.units.Quantity
            Cell side lengths, with units of pixels.
        """
        return self.adaptive_cells["size"] << U.pix

    @property
    def field_distortion_corrections(self) -> U.Quantity[U.dimensionless_unscaled]:
        """
        Spherical distortion corrections to the volumes of the adaptive cells.

        Returns
        -------
        ~astropy.units.Quantity
            Dimensionless distortion corrections to the cell volumes.
        """
        return self.adaptive_cell_distortion_corrections

    @property
    def uniform_grid(self) -> None:
        """
        Describe the grid of cells onto which fields are interpolated.

        Returns
        -------
        None
            Adaptive cells do not form a uniform grid.
        """
        return None

    def _arrays(self) -> list[np.ndarray]:
        """
        List the arrays held by the cell grid, including its tree if built.

        Returns
        -------
        list
            The array attributes of the grid and of its tree, and the interpolated fields.
        """
        arrays = super()._arrays()
        if self._octree is not None:
            arrays += [
                value
                for value in (*vars(self._octree).values(), *self._octree._octree)
                if isinstance(value, np.ndarray)
            ]
        return arrays

    @property
    def octree(self) -> Octree:
        """
        Tree of the adaptive cells, to search for the cells near particles.

        Built from the refined cells when first needed, then reused.

        Returns
        -------
        ~martini._grid_search.Octree
            The tree of the cells onto which fields are interpolated.
        """
        if self._octree is None:
            self._octree = Octree(
                self.field_positions.to_value(U.pix), self.field_sizes.to_value(U.pix)
            )
        return self._octree
