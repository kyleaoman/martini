"""Interpolant functions to render particle properties onto a grid."""

from scipy.spatial import distance, KDTree
from martini._grid_search import (
    build_tree,
    find_grid_intersections,
    FindGridIntersectionsResult,
)
from astropy import units as U
import numpy as np
from functools import partial
from typing import Callable
from collections.abc import Iterable
from martini.sph_kernels import _BaseSPHKernel


def _eval_kernel(
    x_eval: U.Quantity[U.pix],
    x_particle: U.Quantity[U.pix],
    h: U.Quantity[U.pix],
    kernel: _BaseSPHKernel,
) -> U.Quantity[U.pix**-3]:
    """
    Evaluate the kernel.

    Parameters
    ----------
    x_eval : ~astropy.units.Quantity[]
        Positions at which to evaluate kernel, with units of pixels.
    x_particle : ~astropy.units.Quantity
        Positions of particles for which to evaluate kernel, with units of pixels.
    h : ~astropy.units.Quantity
        Particle smoothing lengths (FWHM), with units of pixels.
    kernel : _BaseSPHKernel
        The SPH kernel, evaluated with
        :meth:`~martini.sph_kernels._BaseSPHKernel.eval_kernel`.

    Returns
    -------
    ~astropy.units.Quantity
        Evaluated kernel at ``x_eval`` for particles at positions ``x_particle`` with
        smoothing lengths ``h``. Expected units of ``pixels**-3``.
    """
    r = distance.cdist(x_eval.to_value(h.unit), x_particle.to_value(h.unit))
    # TODO: adaptive kernels evaluate only their preferred kernel, see particle_scatter.
    return kernel.eval_kernel(r, h.to_value(h.unit)) * h.unit**-3


def _eval_cache_kernel(
    u: np.ndarray, kernel_cache: np.ndarray, kernel_cache_extent: float
) -> np.ndarray:
    """
    Get the value of the kernel function on a pre-computed discrete grid.

    Since kernels are generally well behaved compact functions,
    computing the kernel thousands of times can be needlessly expensive
    especially for more computationally expensive kernels. The kernel is linearly
    interpolated between the grid points, and is zero beyond the extent of the grid.
    The grid is uniform, so the grid points bracketing each location can be computed
    directly.

    Parameters
    ----------
    u : ~numpy.ndarray
        Evaluation location in units of the smoothing length (FWHM).

    kernel_cache : ~numpy.ndarray
        The kernel evaluated on a uniform grid spanning ``0 <= u <= kernel_cache_extent``,
        including both endpoints.

    kernel_cache_extent : float
        The extent of the grid on which the kernel is evaluated, in units of the
        smoothing length (FWHM). Should be at least the radius of compact support.

    Returns
    -------
    ~numpy.ndarray
        The approximate kernel amplitude at location ``u``.
    """
    n_intervals = len(kernel_cache) - 1
    x = u * (n_intervals / kernel_cache_extent)  # position in units of grid intervals
    beyond_extent = x > n_intervals
    # left grid point of interval, last interval includes its right end:
    i = np.minimum(x.astype(np.intp), n_intervals - 1)
    x -= i  # now fractional position within interval
    W = kernel_cache[i] * (1 - x) + kernel_cache[i + 1] * x
    W[beyond_extent] = 0.0
    return W


def sph_loop(
    masses: np.ndarray,
    masses_HI: np.ndarray,
    velocities: np.ndarray,
    smoothing_lengths: np.ndarray,
    support_radii: np.ndarray,
    mask_out_of_bound: np.ndarray,
    gs: FindGridIntersectionsResult,
    cell_volumes: np.ndarray,
    cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled],
    kernel_cache: np.ndarray,
    kernel_cache_extent: float,
    extra_fields: dict[str, np.ndarray],
    mfm: bool = False,
) -> dict[str, np.ndarray]:
    """
    Use SPH formalism to scatter particles onto the grid.

    ??.

    Parameters
    ----------
    masses : ~numpy.ndarray
        Particle masses as an array with implicit units.

    masses_HI : ~numpy.ndarray
        Particle HI masses as an array with implicit units.

    velocities : ~numpy.ndarray
        Particle line-of-sight velocities as an array with implicit units.

    smoothing_lengths : ~numpy.ndarray
        Particle smoothing lengths (FWHM) as an array with implicit units.

    support_radii : ~numpy.ndarray
        Particle radii of compact support as an array with implicit units (same as
        ``smoothing_lengths``).

    mask_out_of_bound : ~numpy.ndarray
        A boolean mask selecting particles whose kernels extend outside of the region
        covered by the grid.

    gs : ~martini._grid_search.FindGridIntersectionsResult
        A ``NamedTuple`` containing the results of a tree search for particles touching
        grid cell points within given search radii, see
        :class:`~martini._grid_search.FindGridIntersectionResult` for details.

    cell_volumes : ~numpy.ndarray
        Rectilinear volume of cells onto which fields are interpolated.

    cell_distortion_corrections : ~numpy.ndarray
        Spherical distortion corrections of cells onto which fields are interpolated. The
        interpolated HI density is per unit of corrected volume, such that the HI mass in
        a cell is the density times ``cell_volumes * cell_distortion_corrections``.

    kernel_cache : ~numpy.ndarray
        Kernel amplitude pre-computed on a discrete grid for fast lookup, see
        :func:`_eval_cache_kernel`.

    kernel_cache_extent : float
        Extent of the grid of ``kernel_cache`` in units of the smoothing length (FWHM).

    extra_fields : dict
        Additional fields that should be interpolated onto the grid (e.g. temperature).
        The keys are used to insert the interpolated grids into the return dictionary,
        the values are particle-carried fields to interpolate.

    mfm : bool, optional
        If ``True``, a simplified MFM (without tensor gradient interpolation but with
        volume weights) deposition is performed, otherwise (by default) SPH deposition
        is performed.

    Returns
    -------
    dict
        Contains the interpolated fields as bare arrays.
    """
    # Technically don't need the zero-initialized arrays here but keep for now as
    # arrays to accumulate will be needed to break particles into batches for processing,
    # which will be needed when the tree search result gets huge.
    n_pos = len(cell_volumes)
    field_masses_HI = np.zeros(n_pos)
    field_masses = np.zeros(n_pos)
    field_velocities = np.zeros(n_pos)
    field_extra = {k: np.zeros(n_pos) for k in extra_fields.keys()}
    total_kernel = np.zeros(n_pos) if mfm else np.array(1.0)
    kernel_weights = (
        _eval_cache_kernel(
            np.sqrt((gs.distances**2).sum(axis=1))
            / smoothing_lengths[gs.intersections],
            kernel_cache,
            kernel_cache_extent,
        )
        / (smoothing_lengths**3)[gs.intersections]
    )
    if mfm:
        total_kernel[gs.cell_indices] += np.add.reduceat(
            kernel_weights, gs.strides[:, 0]
        )
    # Need to break here if splitting particles into batches in "mfm" mode:
    # the total kernel contribution in each cell must be fully accumulated before
    # calculating volumes & applying weights. However, we may then proceed safely in
    # batches.
    if mfm:
        volumes = np.bincount(
            gs.intersections,
            weights=kernel_weights
            * np.repeat(
                (cell_volumes * cell_distortion_corrections)[gs.cell_indices]
                / total_kernel[gs.cell_indices],
                np.diff(gs.strides, axis=1)[:, 0],
            ),
            minlength=masses.size,
        )
        volumes[mask_out_of_bound] *= (
            np.pi
            * 4
            / 3
            * support_radii**3
            / np.bincount(
                gs.intersections,
                # rectilinear volume to match the kernel volume in the numerator:
                weights=np.where(
                    kernel_weights,
                    np.repeat(
                        cell_volumes[gs.cell_indices], np.diff(gs.strides, axis=1)[:, 0]
                    ),
                    0,
                ),
                minlength=masses.size,
            )
        )[mask_out_of_bound]
        kernel_weights /= volumes[gs.intersections]
    field_masses[gs.cell_indices] += np.add.reduceat(
        kernel_weights * masses[gs.intersections], gs.strides[:, 0]
    )
    field_masses_HI[gs.cell_indices] += np.add.reduceat(
        kernel_weights * masses_HI[gs.intersections], gs.strides[:, 0]
    )
    field_velocities[gs.cell_indices] += np.add.reduceat(
        kernel_weights * (velocities * masses)[gs.intersections], gs.strides[:, 0]
    )
    for k, v in extra_fields.items():
        field_extra[k][gs.cell_indices] += np.add.reduceat(
            kernel_weights * (v * masses)[gs.intersections], gs.strides[:, 0]
        )
    if mfm:
        kernel_slice = total_kernel != 0
        field_densities_HI = np.where(kernel_slice, field_masses_HI / total_kernel, 0)
    else:
        kernel_slice = field_masses != 0
        # SPH kernels are normalized in pixel coordinates, so field_masses_HI is a density
        # per unit rectilinear volume. Convert to a density per unit corrected volume.
        field_densities_HI = field_masses_HI / cell_distortion_corrections
    return {
        "masses_HI": field_densities_HI,
        "velocities": np.where(kernel_slice, field_velocities / field_masses, 0),
        **{
            k: np.where(kernel_slice, v / field_masses, 0)
            for k, v in field_extra.items()
        },
    }


def _get_out_of_bound_particles(
    particle_positions: np.ndarray,
    particle_radii: np.ndarray,
    field_positions: np.ndarray,
) -> np.ndarray:
    """
    Find particles that fall outside of the region where fields are being evaluated.

    Parameters
    ----------
    particle_positions : ~numpy.ndarray
        Array of particle positions.

    particle_radii : ~numpy.ndarray
        Array of particle sizes (radii of compact support).

    field_positions : ~numpy.ndarray
        Array of locations where the fields are being evaluated.

    Returns
    -------
    ~numpy.ndarray
        Array containing booleans, ``True`` for particles that are outside the region.
    """
    lowBound = np.min(field_positions, axis=0)
    topBound = np.max(field_positions, axis=0)
    mask_out_of_bound = (
        (particle_positions + particle_radii[:, np.newaxis]) > topBound
    ) | ((particle_positions - particle_radii[:, np.newaxis]) < lowBound)
    mask_out_of_bound = np.any(mask_out_of_bound, axis=1)
    return mask_out_of_bound


def particle_scatter(
    main_loop: Callable,  # fill in arg & return types
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: _BaseSPHKernel,
    field_positions: U.Quantity[U.pix],
    d_volume: U.Quantity[U.pix**3],
    distortion_corrections: U.Quantity[U.dimensionless_unscaled],
    *,
    kernel_cache_resolution: int = 256,
) -> dict[str, U.Quantity]:
    """
    Scatter particles onto the cell grid. Can use SPH, MFM or other backends.

    Wrapper for main_loop functions to avoid code duplication.

    Parameters
    ----------
    main_loop : Callable
        The function handling the scatter operation, e.g. ``sph_loop`` or ``mfm_loop``.

    positions : ~astropy.units.Quantity
        Particle positions with units of pixels.

    velocities : ~astropy.units.Quantity
        Particle radial velocities with dimensions of speed.

    smoothing_lengths : ~astropy.units.Quantity
        Particle smoothing lengths with units of pixels.

    masses_HI : ~astropy.units.Quantity
        Particle HI masses with dimensions of mass.

    temperatures : ~astropy.units.Quantity
        Particle temperatures (thermal velocity dispersions) with dimensions of speed
        squared.

    masses : ~astropy.units.Quantity
        Particle masses with dimensions of mass.

    kernel : _BaseSPHKernel
        The SPH kernel, evaluated with
        :meth:`~martini.sph_kernels._BaseSPHKernel.eval_kernel`. Its ``size_in_fwhm``
        gives the radii of compact support in units of ``smoothing_lengths``.

    field_positions : ~numpy.ndarray
        Positions at which to interpolate fields, implicitly with units of pixels.

    d_volume : ~astropy.units.Quantity
        Rectilinear volume element size for ``field_positions`` with units of pixels
        cubed.

    distortion_corrections : ~astropy.units.Quantity
        Dimensionless spherical distortion corrections to ``d_volume``.

    kernel_cache_resolution : int
        Number of intervals on which to sample the kernel for fast lookup.

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    assert kernel.size_in_fwhm is not None
    size_in_fwhm = np.asarray(kernel.size_in_fwhm)
    support_radii = smoothing_lengths * size_in_fwhm
    # cache kernel in units of the smoothing length (FWHM):
    # TODO: adaptive kernels evaluate only their preferred kernel (as in
    # _AdaptiveKernel.eval_kernel), particles assigned a fallback kernel (e.g. the wider
    # Gaussian) are still evaluated with the preferred one. Could cache one kernel per
    # kernel_indices value and select per particle.
    kernel_cache_extent = float(np.max(size_in_fwhm))
    kernel_cache = kernel.eval_kernel(
        np.linspace(0, kernel_cache_extent, kernel_cache_resolution + 1), 1.0
    )
    mask_out_of_bound = _get_out_of_bound_particles(
        positions, support_radii, field_positions
    )
    n_pos = len(field_positions)
    if not isinstance(d_volume, Iterable):
        # Is this required, or will it just broadcast if scalar?
        d_volume = np.ones(n_pos) * d_volume
    if not isinstance(distortion_corrections, Iterable):
        distortion_corrections = np.ones(n_pos) * distortion_corrections
    assert field_positions.unit == positions.unit
    assert field_positions.unit == smoothing_lengths.unit
    tree = build_tree(field_positions.value)
    gs = find_grid_intersections(
        tree, field_positions.value, positions.value, support_radii.value
    )
    array_results = main_loop(
        masses.value,
        masses_HI.value,
        velocities.value,
        smoothing_lengths.value,
        support_radii.value,
        mask_out_of_bound,
        gs,
        d_volume.to_value(positions.unit**3),
        distortion_corrections.to_value(U.dimensionless_unscaled),
        kernel_cache,
        kernel_cache_extent,
        # have in mind selectively omitting temperatures when not needed by spectral model
        extra_fields={"temperatures": temperatures.to_value(velocities.unit**2)},
    )
    array_results["masses_HI"] = U.Quantity(
        array_results["masses_HI"],
        masses_HI.unit / smoothing_lengths.unit**3,
        copy=False,
    )
    array_results["velocities"] = U.Quantity(
        array_results["velocities"],
        velocities.unit,
        copy=False,
    )
    if "temperatures" in array_results:
        array_results["temperatures"] = U.Quantity(
            array_results["temperatures"],
            velocities.unit**2,
            copy=False,
        )
    return array_results


sph = partial(particle_scatter, partial(sph_loop, mfm=False))
mfm = partial(particle_scatter, partial(sph_loop, mfm=True))


def _eval_voronoi_field(
    particle_quantities: U.Quantity,
    nearest_particle_indices: np.ndarray | np.int32 | np.int64,
    missed_particle_cell_indices: np.ndarray | np.int32 | np.int64,
    missed_particle_mask: np.ndarray,
    field_n_particle: np.ndarray,
) -> U.Quantity:
    """
    Evaluate the field at grid points from the particles nearest to them.

    This is a Voronoi tesselation-based interpolation. The arguments use "particle" but
    these refer to Voronoi cells, this is to avoid ambiguity with the grid cells that are
    being interpolated onto.

    Voronoi tesselation works on nearest neighbor assignment. However, on a coarse grid,
    this assignment can skip particles; a particle is not the nearest neighbor of any
    grid cell. These particles are assigned to their nearest neighbor grid cell.
    The final value of a grid cell with multiple particles is taken as the average of its
    assigned particles; the grid cell is estimated to be equi-partioned between its
    assigned particles.

    Parameters
    ----------
    particle_quantities : ~astropy.units.Quantity
        The values of the field as carried by the Voronoi cells.

    nearest_particle_indices : ~numpy.ndarray or int
        The index of the Voronoi cell enclosing each grid cell.

    missed_particle_cell_indices : ~numpy.ndarray or int
        Nearest neighbor cell indices for each missed particle.

    missed_particle_mask : ~numpy.ndarray
        Mask of particles that were not assigned to any grid cell.

    field_n_particle : ~numpy.ndarray
        Number of particles each grid cell receives.

    Returns
    -------
    ~astropy.units.Quantity
        Field interpolated onto the cell grid.
    """
    field_quantity = particle_quantities[nearest_particle_indices]
    field_quantity[missed_particle_cell_indices] += particle_quantities[
        missed_particle_mask
    ]
    field_quantity /= field_n_particle
    return field_quantity


def voronoi_mesh(
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: _BaseSPHKernel,
    field_positions: U.Quantity[U.pix],
    d_volume: U.Quantity[U.pix**3],
    distortion_corrections: U.Quantity[U.dimensionless_unscaled],
    **kwargs: int,
) -> dict[str, U.Quantity]:
    """
    Compute the interpolated fields using a Voronoi mesh.

    Assumes that ``field_positions`` creates a box.

    ??.

    Parameters
    ----------
    positions : ~astropy.units.Quantity
        Voronoi cell positions with units of pixels.

    velocities : ~astropy.units.Quantity
        Voronoi cell radial velocities with dimensions of speed.

    smoothing_lengths : ~astropy.units.Quantity
        Unused.

    masses_HI : ~astropy.units.Quantity
        Voronoi cell HI masses with dimensions of mass.

    temperatures : ~astropy.units.Quantity
        Voronoi cell temperatures (thermal velocity dispersions) with dimensions of speed
        squared.

    masses : ~astropy.units.Quantity
        Unused.

    kernel : _BaseSPHKernel
        Unused.

    field_positions : ~numpy.ndarray
        Positions at which to interpolate fields, implicitly with units of pixels.

    d_volume : ~astropy.units.Quantity
        Rectilinear volume element size for ``field_positions`` with units of pixels
        cubed.

    distortion_corrections : ~astropy.units.Quantity
        Dimensionless spherical distortion corrections to ``d_volume``.

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    masses *= U.dimensionless_unscaled
    if velocities.ndim != 1:
        # more than one dimension of velocity is given, use radial velocity
        velocities = velocities[:, 0]
    particle_indices = np.arange(len(positions))
    _, nearest_particle_indices = KDTree(positions).query(
        field_positions
    )  # nearest neighbor assignment of particles to field pos

    # construct a mask for inbound particles but not assigned to a cell
    inbound_particle_mask = np.all(
        positions > field_positions.min(axis=0), axis=1
    ) & np.all(
        positions < field_positions.max(axis=0), axis=1
    )  # assume box shape for field pos
    used_particle_mask = np.isin(particle_indices, nearest_particle_indices)
    missed_particle_mask = inbound_particle_mask & ~used_particle_mask
    missed_particle_indices = particle_indices[missed_particle_mask]
    _, missed_particle_cell_indices = KDTree(field_positions).query(
        positions[missed_particle_mask]
    )

    particle_masks = nearest_particle_indices == particle_indices[:, np.newaxis]
    particle_masks[missed_particle_indices, missed_particle_cell_indices] = True

    field_n_particle = np.ones(len(field_positions), dtype=np.uint64)
    field_n_particle[missed_particle_cell_indices] += 1

    particle_volumes = np.einsum(
        "ij,j->i", particle_masks, d_volume * distortion_corrections / field_n_particle
    )  # for shared cells, the volume is divided between the particles
    densities = np.zeros(masses_HI.shape) * masses_HI.unit / particle_volumes.unit
    volume_mask = ~(particle_volumes == 0)
    densities[volume_mask] = masses_HI[volume_mask] / particle_volumes[volume_mask]
    field_velocities = _eval_voronoi_field(
        velocities,
        nearest_particle_indices,
        missed_particle_cell_indices,
        missed_particle_mask,
        field_n_particle,
    )
    field_masses_HI = _eval_voronoi_field(
        densities,
        nearest_particle_indices,
        missed_particle_cell_indices,
        missed_particle_mask,
        field_n_particle,
    )
    field_temperatures = _eval_voronoi_field(
        temperatures,
        nearest_particle_indices,
        missed_particle_cell_indices,
        missed_particle_mask,
        field_n_particle,
    )
    return {
        "velocities": field_velocities,
        "masses_HI": field_masses_HI,
        "temperatures": field_temperatures,
    }


def _find_containing_cells(
    coords: np.ndarray, cell_corners: np.ndarray, cell_sizes: np.ndarray
) -> np.ndarray:
    """
    Find the index of the cell containing each coordinate.

    Cells are cubes that tile a region without overlapping, each with its lower corner at
    an integer multiple of its size from the lower corner of the region. Cells of each
    size therefore lie on a regular lattice, so containing cells are found by computing
    lattice indices one cell size at a time.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Coordinates to place in cells.

    cell_corners : ~numpy.ndarray
        Lower corners of the cells (same implicit units as ``coords``).

    cell_sizes : ~numpy.ndarray
        Side lengths of the cells (same implicit units as ``coords``).

    Returns
    -------
    ~numpy.ndarray
        Index of the cell containing each coordinate, or ``-1`` for coordinates outside of
        all cells.
    """
    origin = cell_corners.min(axis=0)
    containing_cells = np.full(len(coords), -1, dtype=np.intp)
    for size in np.unique(cell_sizes):
        level_cells = np.flatnonzero(cell_sizes == size)
        cell_ijk = np.rint((cell_corners[level_cells] - origin) / size).astype(np.intp)
        coord_ijk = np.floor((coords - origin) / size).astype(np.intp)
        lattice_shape = tuple(cell_ijk.max(axis=0) + 1)
        in_lattice = np.all(
            np.logical_and(coord_ijk >= 0, coord_ijk < lattice_shape), axis=1
        )
        cell_keys = np.ravel_multi_index(tuple(cell_ijk.T), lattice_shape)
        coord_keys = np.ravel_multi_index(tuple(coord_ijk[in_lattice].T), lattice_shape)
        sort_order = np.argsort(cell_keys)
        found = np.minimum(
            np.searchsorted(cell_keys, coord_keys, sorter=sort_order),
            len(cell_keys) - 1,
        )
        match = cell_keys[sort_order[found]] == coord_keys
        containing_cells[np.flatnonzero(in_lattice)[match]] = level_cells[
            sort_order[found[match]]
        ]
    return containing_cells


def particle_in_cell(
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: _BaseSPHKernel,
    field_positions: U.Quantity[U.pix],
    d_volume: U.Quantity[U.pix**3],
    distortion_corrections: U.Quantity[U.dimensionless_unscaled],
    **kwargs: int,
) -> dict[str, U.Quantity]:
    """
    Deposit particles into the cells containing them.

    This is zeroth-order particle-in-cell assignment: each particle is deposited entirely
    in the cell that contains it, with no spreading to neighbouring cells. On adaptive
    grids this differs from "nearest grid point" assignment, because the nearest cell
    centre is not necessarily the centre of the containing cell. It is intended for use
    with :class:`~martini.sph_kernels.DiracDeltaKernel`, the SPH kernel is otherwise
    ignored. Cells containing several particles take their mass-weighted mean properties
    (e.g. velocity, temperature), so a refinement strategy such as
    :func:`~martini.mochi.refinement.refine_grid_to_single_occupancy` may be preferable.

    Parameters
    ----------
    positions : ~astropy.units.Quantity
        Particle positions with units of pixels.

    velocities : ~astropy.units.Quantity
        Particle radial velocities with dimensions of speed.

    smoothing_lengths : ~astropy.units.Quantity
        Unused.

    masses_HI : ~astropy.units.Quantity
        Particle HI masses with dimensions of mass.

    temperatures : ~astropy.units.Quantity
        Particle temperatures (thermal velocity dispersions) with dimensions of speed
        squared.

    masses : ~astropy.units.Quantity
        Particle masses with dimensions of mass.

    kernel : _BaseSPHKernel
        Unused.

    field_positions : ~numpy.ndarray
        Positions at which to interpolate fields (cell centres), implicitly with units of
        pixels.

    d_volume : ~astropy.units.Quantity
        Rectilinear volume element size for ``field_positions`` with units of pixels
        cubed. Cells are cubic, with side lengths the cube root of this volume.

    distortion_corrections : ~astropy.units.Quantity
        Dimensionless spherical distortion corrections to ``d_volume``.

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    if velocities.ndim != 1:
        # more than one dimension of velocity is given, use radial velocity
        velocities = velocities[:, 0]
    n_pos = len(field_positions)
    if not isinstance(d_volume, Iterable):
        d_volume = np.ones(n_pos) * d_volume
    if not isinstance(distortion_corrections, Iterable):
        distortion_corrections = np.ones(n_pos) * distortion_corrections
    length_unit = field_positions.unit
    cell_sizes = np.cbrt(d_volume.to_value(length_unit**3))
    containing_cells = _find_containing_cells(
        positions.to_value(length_unit),
        field_positions.to_value(length_unit) - cell_sizes[:, np.newaxis] / 2,
        cell_sizes,
    )
    in_grid = containing_cells >= 0
    cells = containing_cells[in_grid]

    def deposit(values: np.ndarray) -> np.ndarray:
        return np.bincount(cells, weights=values[in_grid], minlength=n_pos)

    m = masses.to_value(masses.unit)
    field_masses = deposit(m)
    field_masses_HI = deposit(masses_HI.to_value(masses_HI.unit))
    field_momenta = deposit(m * velocities.to_value(velocities.unit))
    field_energies = deposit(m * temperatures.to_value(velocities.unit**2))
    kernel_slice = field_masses != 0
    corrected_volumes = (d_volume * distortion_corrections).to_value(length_unit**3)
    return {
        "velocities": U.Quantity(
            np.divide(
                field_momenta, field_masses, out=np.zeros(n_pos), where=kernel_slice
            ),
            velocities.unit,
            copy=False,
        ),
        "masses_HI": U.Quantity(
            field_masses_HI / corrected_volumes,
            masses_HI.unit / length_unit**3,
            copy=False,
        ),
        "temperatures": U.Quantity(
            np.divide(
                field_energies, field_masses, out=np.zeros(n_pos), where=kernel_slice
            ),
            velocities.unit**2,
            copy=False,
        ),
    }


def manual_sph(
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: _BaseSPHKernel,
    field_positions: U.Quantity[U.pix],
    d_volume: U.Quantity[U.pix**3],
    distortion_corrections: U.Quantity[U.dimensionless_unscaled],
    **kwargs: int,
) -> dict[str, U.Quantity]:
    """
    Compute the interpolated fields using SPH interpolation.

    Different SPH schemes have different definitions for velocity interpolation. This
    interpolant assumes that the conserved quantities are interpolated. This SPH
    interpolant serves for testing purposes and writes the equations out explicitely.
    Consequently, it is slow but safe.

    Parameters
    ----------
    positions : ~astropy.units.Quantity
        Particle positions with units of pixels.

    velocities : ~astropy.units.Quantity
        Particle radial velocities with dimensions of speed.

    smoothing_lengths : ~astropy.units.Quantity
        Particle smoothing lengths with units of pixels.

    masses_HI : ~astropy.units.Quantity
        Particle HI masses with dimensions of mass.

    temperatures : ~astropy.units.Quantity
        Particle temperatures (thermal velocity dispersions) with dimensions of speed
        squared.

    masses : ~astropy.units.Quantity
        Particle masses with dimensions of mass.

    kernel : _BaseSPHKernel
        The SPH kernel, evaluated with
        :meth:`~martini.sph_kernels._BaseSPHKernel.eval_kernel`. Its ``size_in_fwhm``
        gives the radii of compact support in units of ``smoothing_lengths``.

    field_positions : ~numpy.ndarray
        Positions at which to interpolate fields, implicitly with units of pixels.

    d_volume : ~astropy.units.Quantity
        Rectilinear volume element size for ``field_positions`` with units of pixels
        cubed.

    distortion_corrections : ~astropy.units.Quantity
        Dimensionless spherical distortion corrections to ``d_volume``. Used to convert
        SPH densities (per unit rectilinear volume) to densities per unit corrected
        volume.

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    masses *= U.dimensionless_unscaled
    n_part, n_dim = positions.shape
    if velocities.ndim != 1:
        # more than one dimension of velocity is given, use radial velocity
        velocities = velocities[:, 0]
    n_pos = len(field_positions)
    if not isinstance(d_volume, Iterable):
        d_volume = np.ones(n_pos) * d_volume
    if not isinstance(distortion_corrections, Iterable):
        distortion_corrections = np.ones(n_pos) * distortion_corrections
    assert kernel.size_in_fwhm is not None
    support_radii = smoothing_lengths * np.asarray(kernel.size_in_fwhm)
    slices = KDTree(field_positions).query_ball_point(positions, support_radii)
    field_masses_HI = np.zeros(n_pos) * masses_HI.unit / d_volume.unit
    field_masses = np.zeros(n_pos) * masses.unit / d_volume.unit
    field_velocities = np.zeros(n_pos) * velocities.unit * masses.unit / d_volume.unit
    field_temperatures = (
        np.zeros(n_pos) * velocities.unit**2 * masses.unit / d_volume.unit
    )
    for i in range(n_part):
        particle_kernel = _eval_kernel(
            field_positions[slices[i]],
            positions[i].reshape((1, n_dim)),
            smoothing_lengths[i],
            kernel,
        )[:, 0]
        field_masses[slices[i]] += particle_kernel * masses[i]
        field_masses_HI[slices[i]] += particle_kernel * masses_HI[i]
        field_velocities[slices[i]] += (
            particle_kernel * velocities[i] * masses[i]
        )  # quantity of movement is conserved
        field_temperatures[slices[i]] += (
            particle_kernel * temperatures[i] * masses[i]
        )  # thermal energy is conserved
    del slices
    kernel_slice = field_masses != 0
    final_velocities = np.zeros(n_pos) * velocities.unit
    final_temperatures = np.zeros(n_pos) * velocities.unit**2
    final_velocities[kernel_slice] = (
        field_velocities[kernel_slice] / field_masses[kernel_slice]
    )
    final_temperatures[kernel_slice] = (
        field_temperatures[kernel_slice] / field_masses[kernel_slice]
    )
    # SPH kernels are normalized in pixel coordinates, convert density to be per unit
    # corrected volume:
    field_masses_HI /= distortion_corrections
    return {
        "velocities": final_velocities,
        "masses_HI": field_masses_HI,
        "temperatures": final_temperatures,
    }
