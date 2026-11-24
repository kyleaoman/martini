"""Interpolant functions to render particle properties onto a grid."""

import contextlib
import itertools
import warnings
from scipy.spatial import distance, KDTree
from martini._grid_search import (
    find_grid_intersections,
    FindGridIntersectionsResult,
    _balanced_blocks,
)
from astropy import units as U
import numpy as np
from functools import partial
from typing import Callable, Generator, NamedTuple, TYPE_CHECKING
from collections.abc import Iterable
from martini.sph_kernels import (
    _AdaptiveKernel,
    _BaseSPHKernel,
    DiracDeltaKernel,
    ScalarKernel,
)
from martini._util import NUMBA_AVAILABLE, numba_threads

if NUMBA_AVAILABLE:
    from numba import njit
    from numba.core.errors import NumbaExperimentalFeatureWarning
if TYPE_CHECKING or not NUMBA_AVAILABLE:
    prange = range  # pragma: no cover
else:
    from numba import prange

if TYPE_CHECKING:
    from martini.mochi.cell_grid import CellGrid

# Distance from the centre to the corners of a cell, in units of its size (slightly
# enlarged to be safe from rounding):
_CELL_REACH = np.sqrt(3) / 2 * (1 + 1e-9)


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
    # Adaptive kernels evaluate only their preferred kernel, see _preferred_kernel.
    return kernel.eval_kernel(r, h.to_value(h.unit)) * h.unit**-3


def _preferred_kernel(kernel: _BaseSPHKernel) -> _BaseSPHKernel:
    """
    Get the kernel that is evaluated to sample a (possibly adaptive) kernel in 3D.

    As in :meth:`~martini.sph_kernels._AdaptiveKernel.eval_kernel`, their preferred
    (first) kernel is used for all particles, ignoring the martini approach to kernel
    validation. A decision on how to handle kernels too small for their cells will be
    needed (or the refinement must guarantee that this is not a problem).

    Parameters
    ----------
    kernel : _BaseSPHKernel
        The SPH kernel.

    Returns
    -------
    _BaseSPHKernel
        The kernel evaluated for all particles.
    """
    return kernel.kernels[0] if isinstance(kernel, _AdaptiveKernel) else kernel


def _numpy_kernel_weights(
    offsets: np.ndarray,
    cell_sizes: np.ndarray,
    smoothing_lengths: np.ndarray,
    kernel: _BaseSPHKernel,
) -> np.ndarray:
    """
    Evaluate kernel weights of particle-cell pairs with :mod:`numpy`.

    The equivalent of the compiled ``_scalar_kernel`` of the kernel classes (see
    :meth:`~martini.sph_kernels._BaseSPHKernel._numba_spherical_kernel_arrays`), used when
    compiled code is unavailable or disabled: the kernel is evaluated at the cell centre,
    except that a :class:`~martini.sph_kernels.DiracDeltaKernel` fills the (half-open)
    cell containing the particle.

    Parameters
    ----------
    offsets : ~numpy.ndarray
        Offsets from the cell centre to the particle of each pair, shape ``(M, 3)``.

    cell_sizes : ~numpy.ndarray
        Side length of the cell of each pair.

    smoothing_lengths : ~numpy.ndarray
        Smoothing length (FWHM) of the particle of each pair.

    kernel : _BaseSPHKernel
        The SPH kernel.

    Returns
    -------
    ~numpy.ndarray
        The kernel weight (per unit volume) of each pair.
    """
    preferred = _preferred_kernel(kernel)
    if isinstance(preferred, DiracDeltaKernel):
        half = cell_sizes[:, np.newaxis] / 2
        inside = np.all((-half <= offsets) & (offsets < half), axis=1)
        return np.where(inside, np.power(cell_sizes, -3.0), 0.0)
    return np.asarray(
        preferred.eval_kernel(np.sqrt((offsets**2).sum(axis=1)), smoothing_lengths)
    )


class _CompiledKernel(NamedTuple):
    """
    Collect data required for kernel weight evaluation in compiled code.

    Attributes
    ----------
    coords : ~numpy.ndarray
        Particle positions, shape ``(N, 3)``.

    centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    sizes : ~numpy.ndarray
        Cell side lengths.

    kernel_slot : ~numpy.ndarray
        For each particle, the entry of ``kernels`` and ``kernel_params`` to use.

    kernels : tuple
        Table of compiled kernel evaluations.

    kernel_params : ~numpy.ndarray
        Table of kernel parameters, shape ``(n_kernels, 2)``.

    h : ~numpy.ndarray
        Rescaled smoothing lengths of the particles.

    ncpu : int
        Number of threads to use.
    """

    coords: np.ndarray
    centres: np.ndarray
    sizes: np.ndarray
    kernel_slot: np.ndarray
    kernels: tuple[ScalarKernel, ...]
    kernel_params: np.ndarray
    h: np.ndarray
    ncpu: int


def _cell_kernel_sums_loop(
    total: np.ndarray,
    intersections: np.ndarray,
    cell_indices: np.ndarray,
    strides: np.ndarray,
    coords: np.ndarray,
    centres: np.ndarray,
    sizes: np.ndarray,
    h: np.ndarray,
    kernel_slot: np.ndarray,
    kernels: tuple[ScalarKernel, ...],
    kernel_params: np.ndarray,
    block_edges: np.ndarray,
) -> None:
    """
    Sum the kernel weights of the particles overlapping each cell.

    Kernel weights are evaluated on the fly and are not stored. Each cell is visited by
    one thread only, so the writes are thread safe. The compiled kernels are passed into
    the compiled loops as numba "first-class functions".

    Parameters
    ----------
    total : ~numpy.ndarray
        Sum of kernel weights for each cell, modified in place.

    intersections : ~numpy.ndarray
        Particle-cell overlaps grouped by cell, see
        :func:`~martini._grid_search.find_grid_intersections`.

    cell_indices : ~numpy.ndarray
        The index of each cell with overlaps.

    strides : ~numpy.ndarray
        The range of ``intersections`` for each cell.

    coords : ~numpy.ndarray
        Particle positions, shape ``(N, 3)``.

    centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    sizes : ~numpy.ndarray
        Cell side lengths.

    h : ~numpy.ndarray
        Rescaled smoothing lengths of the particles.

    kernel_slot : ~numpy.ndarray
        For each particle, the entry of ``kernels`` and ``kernel_params`` to use.

    kernels : tuple
        Table of compiled kernel evaluations.

    kernel_params : ~numpy.ndarray
        Table of kernel parameters, shape ``(n_kernels, 2)``.

    block_edges : ~numpy.ndarray
        Cells are processed in parallel in blocks ``block_edges[b]:block_edges[b + 1]``
        (indices into ``cell_indices``).
    """
    for b in prange(block_edges.size - 1):
        for c in range(block_edges[b], block_edges[b + 1]):
            cell = cell_indices[c]
            cell_total = 0.0
            for m in range(strides[c, 0], strides[c, 1]):
                p = intersections[m]
                slot = kernel_slot[p]
                cell_total += kernels[slot](
                    coords[p, 0] - centres[cell, 0],
                    coords[p, 1] - centres[cell, 1],
                    coords[p, 2] - centres[cell, 2],
                    sizes[cell],
                    h[p],
                    kernel_params[slot, 0],
                    kernel_params[slot, 1],
                )
            total[cell] += cell_total


def _particle_volumes_loop(
    volumes: np.ndarray,
    rectilinear_volumes: np.ndarray,
    total: np.ndarray,
    corrected_cell_volumes: np.ndarray,
    cell_volumes: np.ndarray,
    intersections: np.ndarray,
    cell_indices: np.ndarray,
    strides: np.ndarray,
    coords: np.ndarray,
    centres: np.ndarray,
    sizes: np.ndarray,
    h: np.ndarray,
    kernel_slot: np.ndarray,
    kernels: tuple[ScalarKernel, ...],
    kernel_params: np.ndarray,
) -> None:
    """
    Accumulate the volume of each particle from its share of the cells it overlaps.

    Kernel weights are evaluated on the fly and are not stored. The writes are to
    particles, which can overlap cells processed by different threads, so this loop is
    serial.

    Parameters
    ----------
    volumes : ~numpy.ndarray
        Volume of each particle (its kernel-weighted share of the corrected volumes of
        the cells it overlaps), modified in place.

    rectilinear_volumes : ~numpy.ndarray
        Total rectilinear volume of the cells that each particle overlaps with a non-zero
        weight, modified in place.

    total : ~numpy.ndarray
        Sum of kernel weights for each cell, see :func:`_cell_kernel_sums_loop`.

    corrected_cell_volumes : ~numpy.ndarray
        Cell volumes including the spherical distortion corrections.

    cell_volumes : ~numpy.ndarray
        Rectilinear cell volumes.

    intersections : ~numpy.ndarray
        Particle-cell overlaps grouped by cell, see
        :func:`~martini._grid_search.find_grid_intersections`.

    cell_indices : ~numpy.ndarray
        The index of each cell with overlaps.

    strides : ~numpy.ndarray
        The range of ``intersections`` for each cell.

    coords : ~numpy.ndarray
        Particle positions, shape ``(N, 3)``.

    centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    sizes : ~numpy.ndarray
        Cell side lengths.

    h : ~numpy.ndarray
        Rescaled smoothing lengths of the particles.

    kernel_slot : ~numpy.ndarray
        For each particle, the entry of ``kernels`` and ``kernel_params`` to use.

    kernels : tuple
        Table of compiled kernel evaluations.

    kernel_params : ~numpy.ndarray
        Table of kernel parameters, shape ``(n_kernels, 2)``.
    """
    for c in range(cell_indices.size):
        cell = cell_indices[c]
        for m in range(strides[c, 0], strides[c, 1]):
            p = intersections[m]
            slot = kernel_slot[p]
            w = kernels[slot](
                coords[p, 0] - centres[cell, 0],
                coords[p, 1] - centres[cell, 1],
                coords[p, 2] - centres[cell, 2],
                sizes[cell],
                h[p],
                kernel_params[slot, 0],
                kernel_params[slot, 1],
            )
            if w != 0:
                volumes[p] += w * corrected_cell_volumes[cell] / total[cell]
                rectilinear_volumes[p] += cell_volumes[cell]


def _deposit_loop(
    out: np.ndarray,
    quantities: np.ndarray,
    scale: np.ndarray,
    intersections: np.ndarray,
    cell_indices: np.ndarray,
    strides: np.ndarray,
    coords: np.ndarray,
    centres: np.ndarray,
    sizes: np.ndarray,
    h: np.ndarray,
    kernel_slot: np.ndarray,
    kernels: tuple[ScalarKernel, ...],
    kernel_params: np.ndarray,
    block_edges: np.ndarray,
) -> None:
    """
    Add the kernel-weighted particle quantities into the cells that they overlap.

    Kernel weights are evaluated on the fly and are not stored. Each cell is visited by
    one thread only, so the writes are thread safe.

    Parameters
    ----------
    out : ~numpy.ndarray
        Weighted sums of the quantities for each cell, shape ``(n_cells, k)``, modified
        in place.

    quantities : ~numpy.ndarray
        Particle quantities, shape ``(N, k)``.

    scale : ~numpy.ndarray
        Factor multiplying the kernel weights of each particle.

    intersections : ~numpy.ndarray
        Particle-cell overlaps grouped by cell, see
        :func:`~martini._grid_search.find_grid_intersections`.

    cell_indices : ~numpy.ndarray
        The index of each cell with overlaps.

    strides : ~numpy.ndarray
        The range of ``intersections`` for each cell.

    coords : ~numpy.ndarray
        Particle positions, shape ``(N, 3)``.

    centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    sizes : ~numpy.ndarray
        Cell side lengths.

    h : ~numpy.ndarray
        Rescaled smoothing lengths of the particles.

    kernel_slot : ~numpy.ndarray
        For each particle, the entry of ``kernels`` and ``kernel_params`` to use.

    kernels : tuple
        Table of compiled kernel evaluations.

    kernel_params : ~numpy.ndarray
        Table of kernel parameters, shape ``(n_kernels, 2)``.

    block_edges : ~numpy.ndarray
        Cells are processed in parallel in blocks ``block_edges[b]:block_edges[b + 1]``
        (indices into ``cell_indices``).
    """
    n_quantities = quantities.shape[1]
    for b in prange(block_edges.size - 1):
        for c in range(block_edges[b], block_edges[b + 1]):
            cell = cell_indices[c]
            for m in range(strides[c, 0], strides[c, 1]):
                p = intersections[m]
                slot = kernel_slot[p]
                w = kernels[slot](
                    coords[p, 0] - centres[cell, 0],
                    coords[p, 1] - centres[cell, 1],
                    coords[p, 2] - centres[cell, 2],
                    sizes[cell],
                    h[p],
                    kernel_params[slot, 0],
                    kernel_params[slot, 1],
                )
                if w != 0:
                    w *= scale[p]
                    for k in range(n_quantities):
                        out[cell, k] += w * quantities[p, k]


if NUMBA_AVAILABLE:
    _cell_kernel_sums_loop = njit(parallel=True, cache=True)(_cell_kernel_sums_loop)
    _particle_volumes_loop = njit(cache=True)(_particle_volumes_loop)
    _deposit_loop = njit(parallel=True, cache=True)(_deposit_loop)


@contextlib.contextmanager
def _compiled_loop(ncpu: int) -> Generator[None, None, None]:
    """
    Set the number of threads for compiled loops, and silence numba warnings.

    The compiled kernels are passed to compiled code as numba "first-class functions",
    flagged as an experimental feature.

    Parameters
    ----------
    ncpu : int
        Number of threads to use.
    """
    if not NUMBA_AVAILABLE:  # pragma: no cover
        yield
        return
    with numba_threads(ncpu), warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)
        yield


def sph_loop(
    masses: np.ndarray,
    masses_HI: np.ndarray,
    velocities: np.ndarray,
    support_radii: np.ndarray,
    mask_out_of_bound: np.ndarray,
    gs: FindGridIntersectionsResult,
    cell_volumes: np.ndarray,
    cell_distortion_corrections: np.ndarray,
    kernel_weights: np.ndarray | _CompiledKernel,
    extra_fields: dict[str, np.ndarray],
    mfm: bool = False,
) -> dict[str, np.ndarray]:
    """
    Use SPH formalism to scatter particles onto the grid.

    The kernel weights of the particle-cell pairs are either given, or evaluated on the
    fly (without being stored) in compiled code.

    Parameters
    ----------
    masses : ~numpy.ndarray
        Particle masses as an array with implicit units.

    masses_HI : ~numpy.ndarray
        Particle HI masses as an array with implicit units.

    velocities : ~numpy.ndarray
        Particle line-of-sight velocities as an array with implicit units.

    support_radii : ~numpy.ndarray
        Particle radii of compact support as an array with implicit units.

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

    kernel_weights : ~numpy.ndarray or _CompiledKernel
        The kernel weight (per unit volume) of each particle-cell pair in ``gs``, or a
        description of how to evaluate them in compiled code.

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
    n_pos = len(cell_volumes)
    n_part = len(masses)
    # columns: masses, HI masses, momenta, then extra fields weighted by mass
    quantities = np.column_stack(
        (masses, masses_HI, velocities * masses)
        + tuple(v * masses for v in extra_fields.values())
    )
    compiled = isinstance(kernel_weights, _CompiledKernel)
    if compiled:
        ck = kernel_weights
        loop_args = (
            gs.intersections,
            gs.cell_indices,
            gs.strides,
            ck.coords,
            ck.centres,
            ck.sizes,
            ck.h,
            ck.kernel_slot,
            ck.kernels,
            ck.kernel_params,
        )
        hits_per_cell = gs.strides[:, 1] - gs.strides[:, 0]
        block_edges = _balanced_blocks(hits_per_cell, 8 * ck.ncpu)
    else:
        weights = kernel_weights
        hits_per_cell = np.diff(gs.strides, axis=1)[:, 0]
    total_kernel = np.zeros(n_pos)
    scale = np.ones(n_part)
    if mfm:
        # The total kernel contribution in each cell must be fully accumulated before
        # calculating volumes & applying weights (this matters when processing particles
        # in batches).
        volumes = np.zeros(n_part)
        rectilinear_volumes = np.zeros(n_part)
        corrected_cell_volumes = cell_volumes * cell_distortion_corrections
        if compiled:
            with _compiled_loop(ck.ncpu):
                _cell_kernel_sums_loop(total_kernel, *loop_args, block_edges)
                _particle_volumes_loop(
                    volumes,
                    rectilinear_volumes,
                    total_kernel,
                    corrected_cell_volumes,
                    cell_volumes,
                    *loop_args,
                )
        else:
            total_kernel[gs.cell_indices] += np.add.reduceat(weights, gs.strides[:, 0])
            # cells reached only with zero weights have no kernel total:
            share = np.divide(
                corrected_cell_volumes,
                total_kernel,
                out=np.zeros(n_pos),
                where=total_kernel != 0,
            )
            cells_of_pairs = np.repeat(gs.cell_indices, hits_per_cell)
            volumes = np.bincount(
                gs.intersections,
                weights=weights * share[cells_of_pairs],
                minlength=n_part,
            )
            rectilinear_volumes = np.bincount(
                gs.intersections,
                weights=np.where(weights, cell_volumes[cells_of_pairs], 0),
                minlength=n_part,
            )
        # Correct the volumes of particles whose kernels are cut off by the edge of the
        # grid (those reaching cells only with zero weights deposit nothing), using the
        # rectilinear volume to match the kernel volume in the numerator:
        correct = mask_out_of_bound & (rectilinear_volumes > 0)
        volumes[correct] *= (
            np.pi * 4 / 3 * support_radii[correct] ** 3 / rectilinear_volumes[correct]
        )
        np.divide(1, volumes, out=scale, where=volumes != 0)
    out = np.zeros((n_pos, quantities.shape[1]))
    if compiled:
        with _compiled_loop(ck.ncpu):
            _deposit_loop(out, quantities, scale, *loop_args, block_edges)
    elif gs.intersections.size > 0:
        out[gs.cell_indices] += np.add.reduceat(
            (weights * scale[gs.intersections])[:, np.newaxis]
            * quantities[gs.intersections],
            gs.strides[:, 0],
        )
    field_masses, field_masses_HI, field_momenta = out[:, 0], out[:, 1], out[:, 2]
    if mfm:
        kernel_slice = total_kernel != 0
        field_densities_HI = np.divide(
            field_masses_HI, total_kernel, out=np.zeros(n_pos), where=kernel_slice
        )
    else:
        kernel_slice = field_masses != 0
        # SPH kernels are normalized in pixel coordinates, so field_masses_HI is a density
        # per unit rectilinear volume. Convert to a density per unit corrected volume.
        field_densities_HI = field_masses_HI / cell_distortion_corrections
    return {
        "masses_HI": field_densities_HI,
        "velocities": np.divide(
            field_momenta, field_masses, out=np.zeros(n_pos), where=kernel_slice
        ),
        **{
            k: np.divide(
                out[:, 3 + i], field_masses, out=np.zeros(n_pos), where=kernel_slice
            )
            for i, k in enumerate(extra_fields.keys())
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


def _tree_grid_intersections(
    cell_centres: np.ndarray, coords: np.ndarray, radii: np.ndarray
) -> FindGridIntersectionsResult:
    """
    Search for cell centres within variable radii of coordinates, using a KDTree.

    Unlike :func:`~martini._grid_search.find_grid_intersections`, the cell centres can
    be arbitrary (e.g. an adaptive grid), at the cost of speed and memory.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Coordinates of cell centres, with one column per dimension.

    coords : ~numpy.ndarray
        Coordinates to search around for cells, with one column per dimension.

    radii : ~numpy.ndarray
        Search radii around the ``coords``.

    Returns
    -------
    ~martini._grid_search.FindGridIntersectionsResult
        The matches, in the same layout as returned by
        :func:`~martini._grid_search.find_grid_intersections`.
    """
    candidate_lists = KDTree(
        cell_centres, leafsize=16, compact_nodes=True, balanced_tree=True
    ).query_ball_point(x=coords, r=radii, p=2.0, return_sorted=False)
    data_counts = np.array([len(lst) for lst in candidate_lists], dtype=np.int64)
    if np.sum(data_counts) == 0:
        return FindGridIntersectionsResult(
            intersections=np.empty(0),
            distances=np.empty(0),
            cell_indices=np.empty(0),
            strides=np.empty((0, 2)),
        )
    flat_cell_indices = np.fromiter(
        itertools.chain.from_iterable(candidate_lists), dtype=np.int64
    )
    del candidate_lists
    flat_data_indices = np.repeat(np.arange(len(coords), dtype=np.int32), data_counts)
    del data_counts
    sort_idx = np.argsort(flat_cell_indices, kind="stable")
    sorted_cell_indices = flat_cell_indices[sort_idx]
    del flat_cell_indices
    cell_indices, split_indices, counts = np.unique(
        sorted_cell_indices, return_index=True, return_counts=True
    )
    strides = np.column_stack((split_indices, split_indices + counts))
    intersections = flat_data_indices[sort_idx]
    distances = coords[intersections] - cell_centres[sorted_cell_indices]  # vectors
    return FindGridIntersectionsResult(
        intersections=intersections,
        distances=distances,
        cell_indices=cell_indices,
        strides=strides,
    )


def particle_scatter(
    main_loop: Callable,  # fill in arg & return types
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: _BaseSPHKernel,
    cell_grid: "CellGrid",
    *,
    ncpu: int = 1,
    compiled: bool = NUMBA_AVAILABLE,
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
        The SPH kernel (with smoothing lengths initialized). Adaptive kernels evaluate
        their preferred kernel for all particles (see :func:`_preferred_kernel`), whose
        ``size_in_fwhm`` gives the radii of compact support in units of
        ``smoothing_lengths``. The kernel is evaluated at cell centres, except for a
        :class:`~martini.sph_kernels.DiracDeltaKernel`, which deposits each particle in
        the cell containing it.

    cell_grid : ~martini.mochi.cell_grid.CellGrid
        The grid of cells onto which to interpolate fields. Its ``field_positions``
        (cell centres, with units of pixels), ``field_volumes`` (rectilinear cell
        volumes, with units of pixels cubed), ``field_sizes`` (cell side lengths, with
        units of pixels) and ``field_distortion_corrections`` (dimensionless spherical
        distortion corrections to the volumes) are used. If its ``uniform_grid`` is not
        ``None`` (cells form a uniform grid), a fast search for the cells within reach
        of each particle is used.

    ncpu : int, optional
        Number of threads to use in compiled code.

    compiled : bool, optional
        If ``True`` (default if :mod:`numba` is available), kernel weights are evaluated
        on the fly in compiled code (see
        :meth:`~martini.sph_kernels._BaseSPHKernel._numba_spherical_kernel_arrays`),
        otherwise with :mod:`numpy` (see :func:`_numpy_kernel_weights`).

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    field_positions = cell_grid.field_positions
    d_volume = cell_grid.field_volumes
    distortion_corrections = cell_grid.field_distortion_corrections
    preferred = _preferred_kernel(kernel)
    assert preferred.size_in_fwhm is not None
    support_radii = smoothing_lengths * preferred.size_in_fwhm
    is_dirac_delta = isinstance(preferred, DiracDeltaKernel)
    kernel_arrays = None
    if compiled:
        kernel_arrays = kernel._numba_spherical_kernel_arrays()
        if kernel_arrays is None:
            raise ValueError(
                f"The SPH kernel {type(kernel).__name__} has no compiled kernel "
                "(`_scalar_kernel`), which is required when 'numba' is available. See "
                "the kernels in `martini.sph_kernels` for examples."
            )
    n_pos = len(field_positions)
    if not isinstance(d_volume, Iterable):
        # Is this required, or will it just broadcast if scalar?
        d_volume = np.ones(n_pos) * d_volume
    if not isinstance(distortion_corrections, Iterable):
        distortion_corrections = np.ones(n_pos) * distortion_corrections
    assert field_positions.unit == positions.unit
    assert field_positions.unit == smoothing_lengths.unit
    cell_volumes = d_volume.to_value(positions.unit**3)
    cell_sizes = cell_grid.field_sizes.to_value(positions.unit)
    # A Dirac-delta kernel deposits its particle in the cell containing it, which must be
    # found by the search (as must a cell of a kernel smaller than the cell, though its
    # weight there is zero), so the search reaches at least the furthest corner of the
    # cell containing each particle:
    if cell_grid.uniform_grid is not None:
        containing_cell_sizes = np.full(len(positions), cell_sizes[0])
    else:
        containing_cells = _find_containing_cells(
            positions.value,
            field_positions.value - cell_sizes[:, np.newaxis] / 2,
            cell_sizes,
        )
        containing_cell_sizes = np.where(
            containing_cells >= 0, cell_sizes[containing_cells], 0
        )
    search_radii = np.maximum(support_radii.value, _CELL_REACH * containing_cell_sizes)
    mask_out_of_bound = (
        np.zeros(len(positions), dtype=bool)
        if is_dirac_delta
        else _get_out_of_bound_particles(positions, support_radii, field_positions)
    )
    if cell_grid.uniform_grid is not None:
        # uniform grid: fast search, in grid units (cell centres at integer coordinates)
        grid_shape, cell_size = cell_grid.uniform_grid
        gs = find_grid_intersections(
            grid_shape,
            (positions.value + 0.5) / cell_size - 0.5,
            search_radii / cell_size,
            ncpu=ncpu,
            return_distances=not compiled,  # on the fly otherwise
        )
        if gs.distances is not None and gs.distances.size > 0:
            gs.distances[...] *= cell_size  # back to pixels
    else:
        gs = _tree_grid_intersections(
            field_positions.value, positions.value, search_radii
        )
    if kernel_arrays is not None:
        kernel_slot, kernels, kernel_params, h = kernel_arrays
        kernel_weights: np.ndarray | _CompiledKernel = _CompiledKernel(
            coords=np.ascontiguousarray(positions.value, dtype=np.float64),
            centres=np.ascontiguousarray(field_positions.value, dtype=np.float64),
            sizes=cell_sizes.astype(np.float64),
            kernel_slot=kernel_slot,
            kernels=kernels,
            kernel_params=kernel_params,
            h=h,
            ncpu=ncpu,
        )
    else:
        assert gs.distances is not None
        hits_per_cell = np.diff(gs.strides, axis=1)[:, 0]
        kernel_weights = _numpy_kernel_weights(
            gs.distances,
            np.repeat(cell_sizes[gs.cell_indices], hits_per_cell),
            smoothing_lengths.value[gs.intersections],
            kernel,
        )
    array_results = main_loop(
        masses.value,
        masses_HI.value,
        velocities.value,
        support_radii.value,
        mask_out_of_bound,
        gs,
        cell_volumes,
        distortion_corrections.to_value(U.dimensionless_unscaled),
        kernel_weights,
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
    np.add.at(
        field_quantity,
        missed_particle_cell_indices,
        particle_quantities[missed_particle_mask],
    )
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
    cell_grid: "CellGrid",
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

    cell_grid : ~martini.mochi.cell_grid.CellGrid
        The grid of cells onto which to interpolate fields. Its ``field_positions``
        (cell centres, with units of pixels), ``field_volumes`` (rectilinear cell
        volumes, with units of pixels cubed) and ``field_distortion_corrections``
        (dimensionless spherical distortion corrections to the volumes) are used.

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    field_positions = cell_grid.field_positions
    d_volume = cell_grid.field_volumes
    distortion_corrections = cell_grid.field_distortion_corrections
    masses *= U.dimensionless_unscaled
    if velocities.ndim != 1:
        # more than one dimension of velocity is given, use radial velocity
        velocities = velocities[:, 0]
    particle_indices = np.arange(len(positions))
    _, nearest_particle_indices = KDTree(positions).query(
        field_positions
    )  # nearest neighbor assignment of particles to field pos

    # construct a mask for inbound particles but not assigned to a cell, the grid
    # extends half a cell beyond the outermost cell centres and includes its lower
    # edges but not its upper edges (assume box shape for field pos)
    half_cell_sizes = (
        np.cbrt(d_volume * np.ones(len(field_positions)))[:, np.newaxis] / 2
    )
    inbound_particle_mask = np.all(
        positions >= (field_positions - half_cell_sizes).min(axis=0), axis=1
    ) & np.all(positions < (field_positions + half_cell_sizes).max(axis=0), axis=1)
    used_particle_mask = np.isin(particle_indices, nearest_particle_indices)
    missed_particle_mask = inbound_particle_mask & ~used_particle_mask
    missed_particle_indices = particle_indices[missed_particle_mask]
    _, missed_particle_cell_indices = KDTree(field_positions).query(
        positions[missed_particle_mask]
    )

    particle_masks = nearest_particle_indices == particle_indices[:, np.newaxis]
    particle_masks[missed_particle_indices, missed_particle_cell_indices] = True

    field_n_particle = np.ones(len(field_positions), dtype=np.uint64)
    np.add.at(field_n_particle, missed_particle_cell_indices, 1)

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
    cell_grid: "CellGrid",
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

    cell_grid : ~martini.mochi.cell_grid.CellGrid
        The grid of cells onto which to interpolate fields. Its ``field_positions``
        (cell centres, with units of pixels), ``field_volumes`` (rectilinear cell
        volumes, with units of pixels cubed) and ``field_distortion_corrections``
        (dimensionless spherical distortion corrections to the volumes) are used.

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    field_positions = cell_grid.field_positions
    d_volume = cell_grid.field_volumes
    distortion_corrections = cell_grid.field_distortion_corrections
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
    cell_grid: "CellGrid",
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

    cell_grid : ~martini.mochi.cell_grid.CellGrid
        The grid of cells onto which to interpolate fields. Its ``field_positions``
        (cell centres, with units of pixels), ``field_volumes`` (rectilinear cell
        volumes, with units of pixels cubed) and ``field_distortion_corrections``
        (dimensionless spherical distortion corrections to the volumes) are used.

    Returns
    -------
    dict
        Contains the interpolated fields.
    """
    field_positions = cell_grid.field_positions
    d_volume = cell_grid.field_volumes
    distortion_corrections = cell_grid.field_distortion_corrections
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
