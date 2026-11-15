"""Provide tools to search for particles touching uniform grid cell points."""

import math
import numpy as np
import contextlib
from typing import NamedTuple, TYPE_CHECKING
from martini._util import NUMBA_AVAILABLE, numba_threads

if NUMBA_AVAILABLE:
    from numba import njit
if TYPE_CHECKING or not NUMBA_AVAILABLE:
    prange = range  # pragma: no cover
else:
    from numba import prange


class FindGridIntersectionsResult(NamedTuple):
    """
    Provide a structured return type for grid search functions.

    Attributes
    ----------
    intersections : ~numpy.ndarray
        Each entry is the index of a search coordinate, representing a search coordinate
        matched to a cell. Each search coordinate may match to many cells and therefore
        appear multiple times. Results are grouped by the cell indices.

    distances : ~numpy.ndarray
        The vector offset from the cell centre to the search coordinate for each match.

    cell_indices : ~numpy.ndarray
        The unique cell indices. Each entry in this array corresponds to a range in
        ``strides``.

    strides : ~numpy.ndarray
        Each row can be used as a range to select rows from ``intersections`` that share
        a common cell index.
    """

    intersections: np.ndarray
    distances: np.ndarray
    cell_indices: np.ndarray
    strides: np.ndarray


# The kernels below are written as plain python and compiled with numba if available.
# Each loops over the cells in a particle's bounding box along the outer axes, and
# along each inner axis the range is narrowed to the chord through the sphere. Ranges
# are widened by one cell at each end as a guard against roundoff, and an exact
# squared-distance test (``<=``) decides membership.


def _count_hits_2d(
    coords: np.ndarray, radii: np.ndarray, nx: int, ny: int
) -> np.ndarray:
    """
    Count cells of a uniform 2D grid within the search radius of each coordinate.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 2)``, in units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    nx : int
        Number of cells along the first axis.

    ny : int
        Number of cells along the second axis.

    Returns
    -------
    ~numpy.ndarray
        Number of cell centres within the search radius of each coordinate.
    """
    n = coords.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for p in prange(n):
        x = coords[p, 0]
        y = coords[p, 1]
        r2 = radii[p] * radii[p]
        c = 0
        i0 = max(0, math.ceil(x - radii[p]) - 1)
        i1 = min(nx - 1, math.floor(x + radii[p]) + 1)
        for i in range(i0, i1 + 1):
            dx2 = (x - i) * (x - i)
            h = math.sqrt(max(r2 - dx2, 0.0))
            j0 = max(0, math.ceil(y - h) - 1)
            j1 = min(ny - 1, math.floor(y + h) + 1)
            for j in range(j0, j1 + 1):
                if dx2 + (y - j) * (y - j) <= r2:
                    c += 1
        counts[p] = c
    return counts


def _fill_hits_2d(
    coords: np.ndarray,
    radii: np.ndarray,
    nx: int,
    ny: int,
    offsets: np.ndarray,
    part_idx: np.ndarray,
    cell_idx: np.ndarray,
    dvec: np.ndarray,
) -> None:
    """
    Record cells of a uniform 2D grid within the search radius of each coordinate.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 2)``, in units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    nx : int
        Number of cells along the first axis.

    ny : int
        Number of cells along the second axis.

    offsets : ~numpy.ndarray
        Location in the output arrays where the matches for each coordinate start.

    part_idx : ~numpy.ndarray
        Output array for the index of the search coordinate of each match.

    cell_idx : ~numpy.ndarray
        Output array for the (flattened, C-order) index of the cell of each match.

    dvec : ~numpy.ndarray
        Output array for the offset from the cell centre to the search coordinate of
        each match, shape ``(M, 2)``.
    """
    n = coords.shape[0]
    for p in prange(n):
        x = coords[p, 0]
        y = coords[p, 1]
        r2 = radii[p] * radii[p]
        k = offsets[p]
        i0 = max(0, math.ceil(x - radii[p]) - 1)
        i1 = min(nx - 1, math.floor(x + radii[p]) + 1)
        for i in range(i0, i1 + 1):
            dx2 = (x - i) * (x - i)
            h = math.sqrt(max(r2 - dx2, 0.0))
            j0 = max(0, math.ceil(y - h) - 1)
            j1 = min(ny - 1, math.floor(y + h) + 1)
            for j in range(j0, j1 + 1):
                if dx2 + (y - j) * (y - j) <= r2:
                    part_idx[k] = p
                    cell_idx[k] = i * ny + j
                    dvec[k, 0] = x - i
                    dvec[k, 1] = y - j
                    k += 1


def _count_hits_3d(
    coords: np.ndarray, radii: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    """
    Count cells of a uniform 3D grid within the search radius of each coordinate.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``, in units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    nx : int
        Number of cells along the first axis.

    ny : int
        Number of cells along the second axis.

    nz : int
        Number of cells along the third axis.

    Returns
    -------
    ~numpy.ndarray
        Number of cell centres within the search radius of each coordinate.
    """
    n = coords.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for p in prange(n):
        x = coords[p, 0]
        y = coords[p, 1]
        z = coords[p, 2]
        r2 = radii[p] * radii[p]
        c = 0
        i0 = max(0, math.ceil(x - radii[p]) - 1)
        i1 = min(nx - 1, math.floor(x + radii[p]) + 1)
        for i in range(i0, i1 + 1):
            dx2 = (x - i) * (x - i)
            g = math.sqrt(max(r2 - dx2, 0.0))
            j0 = max(0, math.ceil(y - g) - 1)
            j1 = min(ny - 1, math.floor(y + g) + 1)
            for j in range(j0, j1 + 1):
                dxy2 = dx2 + (y - j) * (y - j)
                h = math.sqrt(max(r2 - dxy2, 0.0))
                k0 = max(0, math.ceil(z - h) - 1)
                k1 = min(nz - 1, math.floor(z + h) + 1)
                for kk in range(k0, k1 + 1):
                    if dxy2 + (z - kk) * (z - kk) <= r2:
                        c += 1
        counts[p] = c
    return counts


def _fill_hits_3d(
    coords: np.ndarray,
    radii: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    offsets: np.ndarray,
    part_idx: np.ndarray,
    cell_idx: np.ndarray,
    dvec: np.ndarray,
) -> None:
    """
    Record cells of a uniform 3D grid within the search radius of each coordinate.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``, in units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    nx : int
        Number of cells along the first axis.

    ny : int
        Number of cells along the second axis.

    nz : int
        Number of cells along the third axis.

    offsets : ~numpy.ndarray
        Location in the output arrays where the matches for each coordinate start.

    part_idx : ~numpy.ndarray
        Output array for the index of the search coordinate of each match.

    cell_idx : ~numpy.ndarray
        Output array for the (flattened, C-order) index of the cell of each match.

    dvec : ~numpy.ndarray
        Output array for the offset from the cell centre to the search coordinate of
        each match, shape ``(M, 3)``.
    """
    n = coords.shape[0]
    for p in prange(n):
        x = coords[p, 0]
        y = coords[p, 1]
        z = coords[p, 2]
        r2 = radii[p] * radii[p]
        k = offsets[p]
        i0 = max(0, math.ceil(x - radii[p]) - 1)
        i1 = min(nx - 1, math.floor(x + radii[p]) + 1)
        for i in range(i0, i1 + 1):
            dx2 = (x - i) * (x - i)
            g = math.sqrt(max(r2 - dx2, 0.0))
            j0 = max(0, math.ceil(y - g) - 1)
            j1 = min(ny - 1, math.floor(y + g) + 1)
            for j in range(j0, j1 + 1):
                dxy2 = dx2 + (y - j) * (y - j)
                h = math.sqrt(max(r2 - dxy2, 0.0))
                k0 = max(0, math.ceil(z - h) - 1)
                k1 = min(nz - 1, math.floor(z + h) + 1)
                for kk in range(k0, k1 + 1):
                    if dxy2 + (z - kk) * (z - kk) <= r2:
                        part_idx[k] = p
                        cell_idx[k] = (i * ny + j) * nz + kk
                        dvec[k, 0] = x - i
                        dvec[k, 1] = y - j
                        dvec[k, 2] = z - kk
                        k += 1


def _group_hits_by_cell(
    part_idx: np.ndarray,
    cell_idx: np.ndarray,
    dvec: np.ndarray,
    offsets: np.ndarray,
    intersections: np.ndarray,
    distances: np.ndarray,
) -> None:
    """
    Reorder matches so that they are grouped by cell.

    Parameters
    ----------
    part_idx : ~numpy.ndarray
        Index of the search coordinate of each match.

    cell_idx : ~numpy.ndarray
        Index of the cell of each match.

    dvec : ~numpy.ndarray
        Offset from cell centre to search coordinate of each match.

    offsets : ~numpy.ndarray
        On input, the location in the output arrays where each cell's group starts
        (exclusive cumulative sum of the per-cell counts). Modified in place: on output
        each entry is the location where the cell's group ends.

    intersections : ~numpy.ndarray
        Output array for ``part_idx`` grouped by cell.

    distances : ~numpy.ndarray
        Output array for ``dvec`` grouped by cell.
    """
    ndim = dvec.shape[1]
    for m in range(part_idx.size):
        c = cell_idx[m]
        dest = offsets[c]
        offsets[c] += 1
        intersections[dest] = part_idx[m]
        for d in range(ndim):
            distances[dest, d] = dvec[m, d]


if NUMBA_AVAILABLE:
    _count_hits_2d = njit(parallel=True, cache=True)(_count_hits_2d)
    _fill_hits_2d = njit(parallel=True, cache=True)(_fill_hits_2d)
    _count_hits_3d = njit(parallel=True, cache=True)(_count_hits_3d)
    _fill_hits_3d = njit(parallel=True, cache=True)(_fill_hits_3d)
    _group_hits_by_cell = njit(cache=True)(_group_hits_by_cell)


def _prepare_inputs(
    grid_shape: tuple[int, ...],
    coords: np.ndarray,
    radii: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """
    Validate and normalize inputs to the grid search functions.

    Parameters
    ----------
    grid_shape : tuple
        Number of cells along each axis (length 2 or 3).

    coords : ~numpy.ndarray
        Search coordinates, one column per dimension.

    radii : ~numpy.ndarray
        Search radii, one per row of ``coords``.

    Returns
    -------
    tuple
        The grid shape as a tuple of ints, and C-contiguous float64 copies (if needed)
        of ``coords`` and ``radii``.
    """
    grid_shape = tuple(int(n) for n in grid_shape)
    if len(grid_shape) not in (2, 3):
        raise ValueError("Grid search supports 2D or 3D grids only.")
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != len(grid_shape):
        raise ValueError(
            f"coords must have shape (N, {len(grid_shape)}) for grid_shape {grid_shape}."
        )
    if radii.shape != (coords.shape[0],):
        raise ValueError("radii must have one entry per row of coords.")
    # Radii can be infinite (e.g. for martini.GlobalProfile). Cap them at a distance that
    # reaches every cell so that cell index ranges are finite:
    reach_all = np.sqrt(np.sum((np.abs(coords) + np.array(grid_shape)) ** 2, axis=1))
    radii = np.minimum(radii, reach_all)
    return grid_shape, coords, radii


def _threads(ncpu: int) -> contextlib.AbstractContextManager:
    """
    Provide a context with the requested number of threads for parallel kernels.

    Parameters
    ----------
    ncpu : int
        Number of threads to use (ignored if :mod:`numba` is unavailable).

    Returns
    -------
    ~contextlib.AbstractContextManager
        The context manager.
    """
    return numba_threads(ncpu) if NUMBA_AVAILABLE else contextlib.nullcontext()


def count_grid_intersections(
    grid_shape: tuple[int, ...],
    coords: np.ndarray,
    radii: np.ndarray,
    ncpu: int = 1,
) -> np.ndarray:
    """
    Count the uniform grid cell centres within the search radius of each coordinate.

    This is much cheaper than :func:`find_grid_intersections` (no per-match output is
    stored), so it can be used to plan memory usage in advance. The counts can be passed
    to :func:`find_grid_intersections` to avoid repeating the calculation.

    Parameters
    ----------
    grid_shape : tuple
        Number of cells along each axis (length 2 or 3). In :mod:`martini` this is the
        spatial (2D) shape of the data cube (including any padding).

    coords : ~numpy.ndarray
        Coordinates to search around for cells, with one column per dimension. In
        :mod:`martini`, this means the ``coords`` of the particles (a particle at
        ``(0 * U.pix, 0 * U.pix)`` is centred in the first pixel).

    radii : ~numpy.ndarray
        Search radii around the ``coords``. Must be an array with the same length as the
        number of particles, a scalar is not accepted.

    ncpu : int
        Number of threads to use.

    Returns
    -------
    ~numpy.ndarray
        Number of cell centres within the search radius of each coordinate.
    """
    grid_shape, coords, radii = _prepare_inputs(grid_shape, coords, radii)
    with _threads(ncpu):
        ndim = len(grid_shape)
        # ndim validity checked in _prepare_inputs
        if ndim == 2:
            nx, ny = grid_shape
            return _count_hits_2d(coords, radii, nx, ny)
        else:
            nx, ny, nz = grid_shape
            return _count_hits_3d(coords, radii, nx, ny, nz)


def find_grid_intersections(
    grid_shape: tuple[int, ...],
    coords: np.ndarray,
    radii: np.ndarray,
    ncpu: int = 1,
    dist_dtype: type = np.float64,
    counts: np.ndarray | None = None,
) -> FindGridIntersectionsResult:
    r"""
    Search for coordinates that reach uniform grid locations within variable radii.

    The grid is assumed to be uniform with square (2D) or cubic (3D) cells of unit size.
    Cell centres are located at integer coordinates, and cells are indexed by their
    position in an array of shape ``grid_shape``.

    A search coordinate with a small radius can slip "between" cell centres if the sphere
    does not enclose any cell centres. This can be prevented by flooring the search radius
    at :math:`\sqrt{n_\mathrm{dim}}/2`.

    In the context of :mod:`martini` these particles will normally evaluate the
    Dirac-delta kernel which will ensure that all of their contribution is deposited in
    the grid cell containing them, even if their artificially increased search radius
    reaches multiple grid cell centres.

    Parameters
    ----------
    grid_shape : tuple
        Number of cells along each axis (length 2 or 3). In :mod:`martini` this is the
        spatial (2D) shape of the data cube (including any padding).

    coords : ~numpy.ndarray
        Coordinates to search around for cells, with one column per dimension. In
        :mod:`martini`, this means the ``coords`` of the particles (a particle at
        ``(0 * U.pix, 0 * U.pix)`` is centred in the first pixel).

    radii : ~numpy.ndarray
        Search radii around the ``coords``. Must be an array with the same length as the
        number of particles, a scalar is not accepted.

    ncpu : int
        Number of threads to use for the search.

    dist_dtype : type, optional
        Data type for returned distances array, can be used to manage memory usage.

    counts : ~numpy.ndarray, optional
        Number of matches for each coordinate, as returned by
        :func:`count_grid_intersections`. Calculated if not provided.

    Returns
    -------
    intersections : ~numpy.ndarray
        Each entry is the index of a search coordinate (index into ``coords``),
        representing a search coordinate matched to a cell. Each search coordinate may
        match to many cells and therefore appear multiple times. Results are grouped by
        the cell indices.

    distances : ~numpy.ndarray
        The vector offset from the cell centre to the search coordinate for each match,
        with one column per dimension.

    cell_indices : ~numpy.ndarray
        The unique (flattened) cell indices. Each entry in this array corresponds to a
        range in ``strides``.

    strides : ~numpy.ndarray
        Each row can be used as a range to select rows from ``intersections`` that share
        a common cell index.
    """
    grid_shape, coords, radii = _prepare_inputs(grid_shape, coords, radii)
    ndim = len(grid_shape)
    n_cells = math.prod(grid_shape)
    if counts is None:
        counts = count_grid_intersections(grid_shape, coords, radii, ncpu=ncpu)
    offsets = np.zeros(counts.size, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    total_intersections = int(offsets[-1] + counts[-1]) if counts.size > 0 else 0

    if total_intersections == 0:
        return FindGridIntersectionsResult(
            intersections=np.empty(0),
            distances=np.empty(0),
            cell_indices=np.empty(0),
            strides=np.empty((0, 2)),
        )

    part_idx = np.empty(total_intersections, dtype=np.int32)
    cell_idx = np.empty(
        total_intersections, dtype=np.int32 if n_cells < 2**31 else np.int64
    )
    dvec: np.ndarray = np.empty((total_intersections, ndim), dtype=dist_dtype)
    with _threads(ncpu):
        # ndim validity checked in _prepare_inputs
        if ndim == 2:
            nx, ny = grid_shape
            _fill_hits_2d(coords, radii, nx, ny, offsets, part_idx, cell_idx, dvec)
        else:
            nx, ny, nz = grid_shape
            _fill_hits_3d(coords, radii, nx, ny, nz, offsets, part_idx, cell_idx, dvec)
    del offsets

    cell_counts = np.bincount(cell_idx, minlength=n_cells)
    offsets = np.zeros(n_cells, dtype=np.int64)
    np.cumsum(cell_counts[:-1], out=offsets[1:])
    intersections = np.empty(total_intersections, dtype=np.int32)
    distances: np.ndarray = np.empty((total_intersections, ndim), dtype=dist_dtype)
    _group_hits_by_cell(part_idx, cell_idx, dvec, offsets, intersections, distances)
    del part_idx, cell_idx, dvec
    cell_indices = np.flatnonzero(cell_counts)
    ends = offsets[cell_indices]
    strides = np.column_stack((ends - cell_counts[cell_indices], ends))

    return FindGridIntersectionsResult(
        intersections=intersections,
        distances=distances,
        cell_indices=cell_indices,
        strides=strides,
    )
