"""Provide tools to search for particles touching uniform or adaptive (octree) grids."""

import math
import numpy as np
import contextlib
from collections.abc import Sequence
from typing import NamedTuple, TYPE_CHECKING
from martini._util import NUMBA_AVAILABLE, numba_threads

if NUMBA_AVAILABLE:
    import numba
    from numba import njit
if TYPE_CHECKING or not NUMBA_AVAILABLE:
    prange = range  # pragma: no cover
else:
    from numba import prange

# Define modes for _walk_subtree:
_WALK_COUNT = 0  # only count hits
_WALK_COUNT_CELLS = 1  # count the hits of each cell
_WALK_RECORD = 2  # record the hits, grouped by cell


class FindGridIntersectionsResult(NamedTuple):
    """
    Provide a structured return type for grid search functions.

    Attributes
    ----------
    intersections : ~numpy.ndarray
        Each entry is the index of a search coordinate, representing a search coordinate
        that "hit" a cell. Each search coordinate may match to many cells and therefore
        appear multiple times. Results are grouped by the cell indices.

    distances : ~numpy.ndarray or None
        The vector offset from the cell centre to the search coordinate for each match
        (``None`` if not requested).

    cell_indices : ~numpy.ndarray
        The unique cell indices. Each entry in this array corresponds to a range in
        ``strides``.

    strides : ~numpy.ndarray
        Each row can be used as a range to select rows from ``intersections`` that share
        a common cell index.
    """

    intersections: np.ndarray
    distances: np.ndarray | None
    cell_indices: np.ndarray
    strides: np.ndarray


def _line_interval(
    coord: float, r2: float, dperp2: float, line_cells: int
) -> tuple[int, int]:
    """
    Find the interval of cells along a line within a search radius.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    coord : float
        Coordinate of the search centre along the line.

    r2 : float
        Squared search radius.

    dperp2 : float
        Squared distance from the search centre to the line.

    line_cells : int
        Number of cells along the line.

    Returns
    -------
    tuple
        First cell and last cell (inclusive) within the search radius. The interval is
        empty if last is smaller than first.
    """
    half_chord = math.sqrt(max(r2 - dperp2, 0.0))
    lo = max(0, math.ceil(coord - half_chord) - 1)
    hi = min(line_cells - 1, math.floor(coord + half_chord) + 1)
    while lo <= hi and dperp2 + (coord - lo) * (coord - lo) > r2:
        lo += 1
    while hi >= lo and dperp2 + (coord - hi) * (coord - hi) > r2:
        hi -= 1
    return lo, hi


def _count_hits(coords: np.ndarray, radii: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    Count cells within the search radius of each coordinate.

    Works in 2 or 3 dimensions only.

    Can be compiled with numba for faster and parallel execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, ndim)``, in (implicit) units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``, same (implicit) units as ``coords``.

    shape : ~numpy.ndarray
        Number of cells along each axis (length 2 or 3).

    Returns
    -------
    ~numpy.ndarray
        Number of cell centres within the search radius of each coordinate.
    """
    ndim = shape.size
    n_line_cells = shape[ndim - 1]
    n_coords = coords.shape[0]
    counts = np.zeros(n_coords, dtype=np.int64)
    for p in prange(n_coords):
        x = coords[p, 0]
        last_coord = coords[p, ndim - 1]
        r2 = radii[p] * radii[p]
        hits = 0
        i0, i1 = _line_interval(x, r2, 0.0, shape[0])
        for i in range(i0, i1 + 1):
            dx2 = (x - i) * (x - i)
            if ndim == 2:
                lo, hi = _line_interval(last_coord, r2, dx2, n_line_cells)
                hits += max(hi - lo + 1, 0)
            else:
                y = coords[p, 1]
                j0, j1 = _line_interval(y, r2, dx2, shape[1])
                for j in range(j0, j1 + 1):
                    lo, hi = _line_interval(
                        last_coord, r2, dx2 + (y - j) * (y - j), n_line_cells
                    )
                    hits += max(hi - lo + 1, 0)
        counts[p] = hits
    return counts


def _candidate_rows(
    p: int, coords: np.ndarray, radii: np.ndarray, shape: np.ndarray
) -> tuple[int, int]:
    """
    Find the range of the first pixel or cell grid axis that a search radius may touch.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    p : int
        Index of the particle.

    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, ndim)``, in (implicit) units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``, same (implicit) units as ``coords``.

    shape : ~numpy.ndarray
        Number of cells along each axis (length 2 or 3).

    Returns
    -------
    tuple
        First and last index (inclusive) along the first axis. Empty if the search
        sphere's bounding box misses the grid.
    """
    r2 = radii[p] * radii[p]
    for d in range(1, shape.size):
        lo, hi = _line_interval(coords[p, d], r2, 0.0, shape[d])
        if hi < lo:
            return 0, -1
    return _line_interval(coords[p, 0], r2, 0.0, shape[0])


def _build_line_candidates(
    coords: np.ndarray, radii: np.ndarray, shape: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    List the candidate particles for each pixel or cell grid line.

    Candidates are particles whose search radius may intersect the line; the exact
    intersection (possibly empty) is found later.

    Works in 2 or 3 dimensions only.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, ndim)``, in (implicit) units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``, same (implicit) units as ``coords``.

    shape : ~numpy.ndarray
        Number of cells along each axis (length 2 or 3).

    Returns
    -------
    tuple
        The candidates of line ``L`` are ``line_particles[line_ptr[L]:line_ptr[L + 1]]``,
        in ascending order.
    """
    ndim = shape.size
    n_lines = shape[0] if ndim == 2 else shape[0] * shape[1]
    # A particle's candidate lines are contiguous in the flattened line index (in 3D,
    # along each row of the first axis), so counts are accumulated as differences.
    candidate_diff = np.zeros(n_lines + 1, dtype=np.int64)
    for p in range(coords.shape[0]):
        i0, i1 = _candidate_rows(p, coords, radii, shape)
        if ndim == 2:
            if i1 >= i0:
                candidate_diff[i0] += 1
                candidate_diff[i1 + 1] -= 1
        else:
            x = coords[p, 0]
            r2 = radii[p] * radii[p]
            for i in range(i0, i1 + 1):
                j0, j1 = _line_interval(coords[p, 1], r2, (x - i) * (x - i), shape[1])
                if j1 >= j0:
                    candidate_diff[i * shape[1] + j0] += 1
                    candidate_diff[i * shape[1] + j1 + 1] -= 1
    line_ptr = np.zeros(n_lines + 1, dtype=np.int64)
    line_ptr[1:] = np.cumsum(candidate_diff[:-1])
    line_ptr[1:] = np.cumsum(line_ptr[1:])
    line_particles = np.empty(line_ptr[-1], dtype=np.int32)
    next_slot = line_ptr[:-1].copy()
    for p in range(coords.shape[0]):
        i0, i1 = _candidate_rows(p, coords, radii, shape)
        if ndim == 2:
            for i in range(i0, i1 + 1):
                line_particles[next_slot[i]] = p
                next_slot[i] += 1
        else:
            x = coords[p, 0]
            r2 = radii[p] * radii[p]
            for i in range(i0, i1 + 1):
                j0, j1 = _line_interval(coords[p, 1], r2, (x - i) * (x - i), shape[1])
                for line in range(i * shape[1] + j0, i * shape[1] + j1 + 1):
                    line_particles[next_slot[line]] = p
                    next_slot[line] += 1
    return line_ptr, line_particles


def _line_perp(
    line_index: int, particle_index: int, coords: np.ndarray, shape: np.ndarray
) -> tuple[float, float, float]:
    """
    Find the offset of a particle from a pixel or cell grid line.

    Works in 2 or 3 dimensions only.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    line_index : int
        Index of the line.

    particle_index : int
        Index of the particle.

    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, ndim)``, in (implicit) units of the cell spacing.

    shape : ~numpy.ndarray
        Number of cells along each axis (length 2 or 3).

    Returns
    -------
    tuple
        Offsets along the first and (in 3D) second axes, and the squared distance from
        the line.
    """
    ndim = shape.size
    if ndim == 2:
        dx = coords[particle_index, 0] - line_index
        return dx, 0.0, dx * dx
    dx = coords[particle_index, 0] - line_index // shape[1]
    dy = coords[particle_index, 1] - line_index % shape[1]
    return dx, dy, dx * dx + dy * dy


def _count_line_hits(
    coords: np.ndarray,
    radii: np.ndarray,
    shape: np.ndarray,
    line_ptr: np.ndarray,
    line_particles: np.ndarray,
    block_edges: np.ndarray,
    line_hits: np.ndarray,
    line_cells_hit: np.ndarray,
) -> None:
    """
    Count the "hits" and the number of distinct cells hit on each pixel or cell grid line.

    Can be compiled with numba for faster and parallel execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, ndim)``, in (implicit) units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``, same (implicit) units as ``coords``.

    shape : ~numpy.ndarray
        Number of cells along each axis (length 2 or 3).

    line_ptr : ~numpy.ndarray
        Candidate list offsets, see :func:`_build_line_candidates`.

    line_particles : ~numpy.ndarray
        Candidate lists, see :func:`_build_line_candidates`.

    block_edges : ~numpy.ndarray
        Lines are processed in parallel in blocks ``block_edges[b]:block_edges[b + 1]``.

    line_hits : ~numpy.ndarray
        Output array for the number of hits on each line.

    line_cells_hit : ~numpy.ndarray
        Output array for the number of distinct cells hit on each line.
    """
    n_line_cells = shape[shape.size - 1]
    for b in prange(block_edges.size - 1):
        coverage_diff = np.zeros(n_line_cells + 1, dtype=np.int64)
        for line in range(block_edges[b], block_edges[b + 1]):
            hits = 0
            kmin = n_line_cells
            kmax = -1
            for m in range(line_ptr[line], line_ptr[line + 1]):
                p = line_particles[m]
                _, _, dperp2 = _line_perp(line, p, coords, shape)
                r2 = radii[p] * radii[p]
                lo, hi = _line_interval(
                    coords[p, shape.size - 1], r2, dperp2, n_line_cells
                )
                if hi >= lo:
                    hits += hi - lo + 1
                    coverage_diff[lo] += 1
                    coverage_diff[hi + 1] -= 1
                    kmin = min(kmin, lo)
                    kmax = max(kmax, hi)
            cells_hit = 0
            coverage = 0
            for k in range(kmin, kmax + 1):
                coverage += coverage_diff[k]
                if coverage > 0:
                    cells_hit += 1
                coverage_diff[k] = 0
            if kmax >= 0:
                coverage_diff[kmax + 1] = 0
            line_hits[line] = hits
            line_cells_hit[line] = cells_hit


def _fill_lines(
    coords: np.ndarray,
    radii: np.ndarray,
    shape: np.ndarray,
    line_ptr: np.ndarray,
    line_particles: np.ndarray,
    block_edges: np.ndarray,
    hit_ptr: np.ndarray,
    cell_ptr: np.ndarray,
    intersections: np.ndarray,
    cell_indices: np.ndarray,
    strides: np.ndarray,
    distances: np.ndarray,
    fill_distances: bool,
) -> None:
    """
    Record cell "hits" on each pixel or cell grid line, grouped by cell.

    Works in 2 or 3 dimensions only.

    Can be compiled with numba for faster and parallel execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, ndim)``, in (implicit) units of the cell spacing.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``, same (implicit) units as ``coords``.

    shape : ~numpy.ndarray
        Number of cells along each axis (length 2 or 3).

    line_ptr : ~numpy.ndarray
        Candidate list offsets, see :func:`_build_line_candidates`.

    line_particles : ~numpy.ndarray
        Candidate lists, see :func:`_build_line_candidates`.

    block_edges : ~numpy.ndarray
        In parallel execution, lines are processed in load-balanced blocks
        ``block_edges[b]:block_edges[b + 1]``.

    hit_ptr : ~numpy.ndarray
        Location in ``intersections`` where the hits for each line start.

    cell_ptr : ~numpy.ndarray
        Location in ``cell_indices`` and ``strides`` where the cells of each line start.

    intersections : ~numpy.ndarray
        Output array for the index of the search coordinate of each cell hit.

    cell_indices : ~numpy.ndarray
        Output array for the (flattened) index of each cell that is hit.

    strides : ~numpy.ndarray
        Output array for the range of ``intersections`` matched to each cell.

    distances : ~numpy.ndarray
        Output array for the offset from the cell centre to the search coordinate of
        each match, shape ``(M, ndim)`` (ignored unless ``fill_distances``).

    fill_distances : bool
        If ``True``, fill ``distances``.
    """
    ndim = shape.size
    n_line_cells = shape[ndim - 1]
    for b in prange(block_edges.size - 1):
        coverage_diff = np.zeros(n_line_cells + 1, dtype=np.int64)
        cursor = np.empty(n_line_cells, dtype=np.int64)
        for line in range(block_edges[b], block_edges[b + 1]):
            kmin = n_line_cells
            kmax = -1
            for m in range(line_ptr[line], line_ptr[line + 1]):
                p = line_particles[m]
                _, _, dperp2 = _line_perp(line, p, coords, shape)
                r2 = radii[p] * radii[p]
                lo, hi = _line_interval(coords[p, ndim - 1], r2, dperp2, n_line_cells)
                if hi >= lo:
                    coverage_diff[lo] += 1
                    coverage_diff[hi + 1] -= 1
                    kmin = min(kmin, lo)
                    kmax = max(kmax, hi)
            hit_slot = hit_ptr[line]
            cell_slot = cell_ptr[line]
            coverage = 0
            for k in range(kmin, kmax + 1):
                coverage += coverage_diff[k]
                coverage_diff[k] = 0
                if coverage > 0:
                    cursor[k] = hit_slot
                    cell_indices[cell_slot] = line * n_line_cells + k
                    strides[cell_slot, 0] = hit_slot
                    strides[cell_slot, 1] = hit_slot + coverage
                    cell_slot += 1
                    hit_slot += coverage
            if kmax >= 0:
                coverage_diff[kmax + 1] = 0
            for m in range(line_ptr[line], line_ptr[line + 1]):
                p = line_particles[m]
                dx, dy, dperp2 = _line_perp(line, p, coords, shape)
                r2 = radii[p] * radii[p]
                last_coord = coords[p, ndim - 1]
                lo, hi = _line_interval(last_coord, r2, dperp2, n_line_cells)
                for k in range(lo, hi + 1):
                    dest = cursor[k]
                    cursor[k] += 1
                    intersections[dest] = p
                    if fill_distances:
                        distances[dest, 0] = dx
                        if ndim == 2:
                            distances[dest, 1] = last_coord - k
                        else:
                            distances[dest, 1] = dy
                            distances[dest, 2] = last_coord - k


# The functions for tree search:
# _box_distance2
# _root_range
# _walk_subtree
# _walk_from_roots
# _count_from_roots
# _fill_work_unit_candidates
# _group_by_work_unit
# _work_unit_hits
# _cells_below
# logically belong to Octree. They are placed at module level so that numba can compile
# and cache them, and they can call each other from compiled code.


def _box_distance2(
    x: float, y: float, z: float, corner: np.ndarray, size: float
) -> float:
    """
    Find the squared distance from a point to a cube (zero inside the cube).

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    x : float
        First coordinate of the point.

    y : float
        Second coordinate of the point.

    z : float
        Third coordinate of the point.

    corner : ~numpy.ndarray
        Lower corner of the cube, shape ``(3,)``.

    size : float
        Side length of the cube.

    Returns
    -------
    float
        The squared distance.
    """
    d2 = 0.0
    for d, coord in enumerate((x, y, z)):
        lo = corner[d]
        hi = corner[d] + size
        if coord < lo:
            d2 += (lo - coord) * (lo - coord)
        elif coord > hi:
            d2 += (coord - hi) * (coord - hi)
    return d2


def _root_range(
    coord: float, r: float, origin: float, root_size: float, n_roots: int
) -> tuple[int, int]:
    """
    Find the range of root nodes along an axis that a search radius may touch.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    coord : float
        Coordinate of the search centre along the axis.

    r : float
        Search radius.

    origin : float
        Lower edge of the root lattice along the axis.

    root_size : float
        Side length of the root nodes.

    n_roots : int
        Number of root nodes along the axis.

    Returns
    -------
    tuple
        First and last (inclusive) root node index. Empty if last is smaller than first.
    """
    lo = max(math.floor((coord - r - origin) / root_size), 0.0)
    hi = min(math.floor((coord + r - origin) / root_size), n_roots - 1.0)
    return int(lo), int(hi)


def _walk_subtree(
    start: int,
    p: int,
    x: float,
    y: float,
    z: float,
    r2: float,
    cell_centres: np.ndarray,
    first_child: np.ndarray,
    node_cell: np.ndarray,
    node_corner: np.ndarray,
    node_size: np.ndarray,
    stack: np.ndarray,
    mode: int,
    cell_counts: np.ndarray,
    cursor: np.ndarray,
    intersections: np.ndarray,
    distances: np.ndarray,
    fill_distances: bool,
) -> int:
    """
    Find the cells below a node of a tree with centres within a search radius.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    start : int
        Node to search below.

    p : int
        Index of the search coordinate.

    x : float
        First search coordinate.

    y : float
        Second search coordinate.

    z : float
        Third search coordinate.

    r2 : float
        Squared search radius.

    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1``.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1``.

    node_corner : ~numpy.ndarray
        Lower corner of each node, shape ``(n_nodes, 3)``.

    node_size : ~numpy.ndarray
        Side length of each node.

    stack : ~numpy.ndarray
        Work space for the tree traversal, at least ``8 * (depth + 1)`` long.

    mode : int
        ``_WALK_COUNT`` to only count the cells hit, ``_WALK_COUNT_CELLS`` to also add
        them to ``cell_counts``, ``_WALK_RECORD`` to also record the hits in
        ``intersections`` (and ``distances`` if ``fill_distances``).

    cell_counts : ~numpy.ndarray
        Number of hits of each cell (incremented if ``mode`` is ``_WALK_COUNT_CELLS``).

    cursor : ~numpy.ndarray
        Location in ``intersections`` of the next hit of each cell (incremented if
        ``mode`` is ``_WALK_RECORD``).

    intersections : ~numpy.ndarray
        Output array for the index of the search coordinate of each hit (if ``mode`` is
        ``_WALK_RECORD``).

    distances : ~numpy.ndarray
        Output array for the offset from the cell centre to the search coordinate of
        each hit (if ``mode`` is ``_WALK_RECORD`` and ``fill_distances``).

    fill_distances : bool
        If ``True``, fill ``distances``. Ignored if ``mode`` is not ``_WALK_RECORD``.

    Returns
    -------
    int
        Number of cells hit.
    """
    stack[0] = start
    stack_size = 1
    hits = 0
    while stack_size > 0:
        stack_size -= 1
        node = stack[stack_size]
        cell = node_cell[node]
        if cell >= 0:
            dx = x - cell_centres[cell, 0]
            dy = y - cell_centres[cell, 1]
            dz = z - cell_centres[cell, 2]
            if dx * dx + dy * dy + dz * dz <= r2:
                hits += 1
                if mode == _WALK_COUNT_CELLS:
                    cell_counts[cell] += 1
                elif mode == _WALK_RECORD:
                    dest = cursor[cell]
                    cursor[cell] += 1
                    intersections[dest] = p
                    if fill_distances:
                        distances[dest, 0] = dx
                        distances[dest, 1] = dy
                        distances[dest, 2] = dz
            continue
        if _box_distance2(x, y, z, node_corner[node], node_size[node]) <= r2:
            for child in range(8):
                stack[stack_size] = first_child[node] + child
                stack_size += 1
    return hits


def _walk_from_roots(
    p: int,
    coords: np.ndarray,
    radii: np.ndarray,
    cell_centres: np.ndarray,
    origin: np.ndarray,
    root_size: float,
    root_shape: np.ndarray,
    root_nodes: np.ndarray,
    first_child: np.ndarray,
    node_cell: np.ndarray,
    node_corner: np.ndarray,
    node_size: np.ndarray,
    work_unit_of_node: np.ndarray,
    stack: np.ndarray,
    work_units: np.ndarray,
    start: int,
    record: bool,
) -> int:
    """
    Search a tree from its root nodes for one coordinate.

    Counts the cells with centres within the search radius or, if ``work_unit_of_node`` is
    not empty, records the work units (see :meth:`Octree._work_units`) that the search
    radius may reach without searching below them.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    p : int
        Index of the search coordinate.

    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    origin : ~numpy.ndarray
        Lower corner of the root lattice, see :meth:`Octree._build_octree`.

    root_size : float
        Side length of the root nodes.

    root_shape : ~numpy.ndarray
        Number of root nodes along each axis.

    root_nodes : ~numpy.ndarray
        Node at each (flattened) position of the root lattice, or ``-1``.

    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1``.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1``.

    node_corner : ~numpy.ndarray
        Lower corner of each node, shape ``(n_nodes, 3)``.

    node_size : ~numpy.ndarray
        Side length of each node.

    work_unit_of_node : ~numpy.ndarray
        Index of the work unit of each node, or ``-1``. Empty to count cells instead.

    stack : ~numpy.ndarray
        Work space for the tree traversal, at least ``8 * (depth + 1)`` long.

    work_units : ~numpy.ndarray
        Output array for the work units reached (ignored unless ``record``).

    start : int
        Location in ``work_units`` of the first work unit reached by this coordinate.

    record : bool
        If ``True``, record ``work_units``.

    Returns
    -------
    int
        Number of cells hit, or of work units reached.
    """
    x = coords[p, 0]
    y = coords[p, 1]
    z = coords[p, 2]
    r = radii[p]
    r2 = r * r
    i0, i1 = _root_range(x, r, origin[0], root_size, root_shape[0])
    j0, j1 = _root_range(y, r, origin[1], root_size, root_shape[1])
    k0, k1 = _root_range(z, r, origin[2], root_size, root_shape[2])
    by_work_unit = work_unit_of_node.size > 0
    placeholder_int = np.empty(0, dtype=np.int64)
    placeholder_float = np.empty((0, 3), dtype=np.float64)
    n_found = 0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                root = root_nodes[(i * root_shape[1] + j) * root_shape[2] + k]
                if root < 0:
                    continue
                if not by_work_unit:
                    n_found += _walk_subtree(
                        root,
                        p,
                        x,
                        y,
                        z,
                        r2,
                        cell_centres,
                        first_child,
                        node_cell,
                        node_corner,
                        node_size,
                        stack,
                        _WALK_COUNT,
                        placeholder_int,
                        placeholder_int,
                        placeholder_int,
                        placeholder_float,
                        False,
                    )
                    continue
                stack[0] = root
                stack_size = 1
                while stack_size > 0:
                    stack_size -= 1
                    node = stack[stack_size]
                    corner = node_corner[node]
                    if _box_distance2(x, y, z, corner, node_size[node]) > r2:
                        continue
                    if work_unit_of_node[node] >= 0:
                        if record:
                            work_units[start + n_found] = work_unit_of_node[node]
                        n_found += 1
                        continue
                    # nodes above the work units are never leaves:
                    for child in range(8):
                        stack[stack_size] = first_child[node] + child
                        stack_size += 1
    return n_found


def _count_from_roots(
    coords: np.ndarray,
    radii: np.ndarray,
    cell_centres: np.ndarray,
    origin: np.ndarray,
    root_size: float,
    root_shape: np.ndarray,
    root_nodes: np.ndarray,
    first_child: np.ndarray,
    node_cell: np.ndarray,
    node_corner: np.ndarray,
    node_size: np.ndarray,
    depth: int,
    work_unit_of_node: np.ndarray,
    block_edges: np.ndarray,
    counts: np.ndarray,
) -> None:
    """
    Count cells hit (or work units reached) from the root nodes for each coordinate.

    Can be compiled with numba for faster and parallel execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    origin : ~numpy.ndarray
        Lower corner of the root lattice, see :meth:`Octree._build_octree`.

    root_size : float
        Side length of the root nodes.

    root_shape : ~numpy.ndarray
        Number of root nodes along each axis.

    root_nodes : ~numpy.ndarray
        Node at each (flattened) position of the root lattice, or ``-1``.

    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1``.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1``.

    node_corner : ~numpy.ndarray
        Lower corner of each node, shape ``(n_nodes, 3)``.

    node_size : ~numpy.ndarray
        Side length of each node.

    depth : int
        Number of levels of the tree below the root nodes.

    work_unit_of_node : ~numpy.ndarray
        Index of the work unit of each node, or ``-1``. Empty to count cells instead.

    block_edges : ~numpy.ndarray
        Coordinates are processed in parallel in blocks
        ``block_edges[b]:block_edges[b + 1]``.

    counts : ~numpy.ndarray
        Output array for the number of cells hit (or work units reached) by each
        coordinate.
    """
    placeholder = np.empty(0, dtype=np.int64)
    for b in prange(block_edges.size - 1):
        stack = np.empty(8 * (depth + 1), dtype=np.int64)
        for p in range(block_edges[b], block_edges[b + 1]):
            counts[p] = _walk_from_roots(
                p,
                coords,
                radii,
                cell_centres,
                origin,
                root_size,
                root_shape,
                root_nodes,
                first_child,
                node_cell,
                node_corner,
                node_size,
                work_unit_of_node,
                stack,
                placeholder,
                0,
                False,
            )


def _fill_work_unit_candidates(
    coords: np.ndarray,
    radii: np.ndarray,
    cell_centres: np.ndarray,
    origin: np.ndarray,
    root_size: float,
    root_shape: np.ndarray,
    root_nodes: np.ndarray,
    first_child: np.ndarray,
    node_cell: np.ndarray,
    node_corner: np.ndarray,
    node_size: np.ndarray,
    depth: int,
    work_unit_of_node: np.ndarray,
    block_edges: np.ndarray,
    candidate_ptr: np.ndarray,
    candidate_work_units: np.ndarray,
) -> None:
    """
    Record the work units that the search radius of each coordinate may reach.

    Can be compiled with numba for faster and parallel execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    origin : ~numpy.ndarray
        Lower corner of the root lattice, see :meth:`Octree._build_octree`.

    root_size : float
        Side length of the root nodes.

    root_shape : ~numpy.ndarray
        Number of root nodes along each axis.

    root_nodes : ~numpy.ndarray
        Node at each (flattened) position of the root lattice, or ``-1``.

    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1``.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1``.

    node_corner : ~numpy.ndarray
        Lower corner of each node, shape ``(n_nodes, 3)``.

    node_size : ~numpy.ndarray
        Side length of each node.

    depth : int
        Number of levels of the tree below the root nodes.

    work_unit_of_node : ~numpy.ndarray
        Index of the work unit of each node, or ``-1``.

    block_edges : ~numpy.ndarray
        In parallel execution, coordinates are processed in load-balanced blocks
        ``block_edges[b]:block_edges[b + 1]``.

    candidate_ptr : ~numpy.ndarray
        Location in ``candidate_work_units`` where the work units of each coordinate
        start.

    candidate_work_units : ~numpy.ndarray
        Output array for the work units reached, grouped by coordinate.
    """
    for b in prange(block_edges.size - 1):
        stack = np.empty(8 * (depth + 1), dtype=np.int64)
        for p in range(block_edges[b], block_edges[b + 1]):
            _walk_from_roots(
                p,
                coords,
                radii,
                cell_centres,
                origin,
                root_size,
                root_shape,
                root_nodes,
                first_child,
                node_cell,
                node_corner,
                node_size,
                work_unit_of_node,
                stack,
                candidate_work_units,
                candidate_ptr[p],
                True,
            )


def _group_by_work_unit(
    candidate_ptr: np.ndarray,
    candidate_work_units: np.ndarray,
    cursor: np.ndarray,
    work_unit_particles: np.ndarray,
) -> None:
    """
    Reorder candidate work units grouped by search coordinate into groups by work unit.

    Within each work unit, candidate search coordinates are in ascending order.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    candidate_ptr : ~numpy.ndarray
        Location in ``candidate_work_units`` where the candidate work units of each
        search coordinate start.

    candidate_work_units : ~numpy.ndarray
        The candidate work units, grouped by search coordinate.

    cursor : ~numpy.ndarray
        Location in ``work_unit_particles`` where the candidates of each work unit start
        (modified).

    work_unit_particles : ~numpy.ndarray
        Output array for the candidate search coordinates, grouped by work unit.
    """
    for p in range(candidate_ptr.size - 1):
        for m in range(candidate_ptr[p], candidate_ptr[p + 1]):
            work_unit = candidate_work_units[m]
            work_unit_particles[cursor[work_unit]] = p
            cursor[work_unit] += 1


def _work_unit_hits(
    coords: np.ndarray,
    radii: np.ndarray,
    cell_centres: np.ndarray,
    first_child: np.ndarray,
    node_cell: np.ndarray,
    node_corner: np.ndarray,
    node_size: np.ndarray,
    depth: int,
    work_units: np.ndarray,
    work_unit_ptr: np.ndarray,
    work_unit_particles: np.ndarray,
    block_edges: np.ndarray,
    record: bool,
    work_unit_hits: np.ndarray,
    cell_counts: np.ndarray,
    cursor: np.ndarray,
    intersections: np.ndarray,
    distances: np.ndarray,
    fill_distances: bool,
) -> None:
    """
    Count or record the cell "hits" below each work unit, grouped by cell.

    Each cell is below exactly one work unit, so work units can be processed in parallel
    without write collisions. The candidate coordinates of each work unit are processed
    in ascending order, so that the hits of each cell are recorded in ascending order.

    Can be compiled with numba for faster and parallel execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1``.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1``.

    node_corner : ~numpy.ndarray
        Lower corner of each node, shape ``(n_nodes, 3)``.

    node_size : ~numpy.ndarray
        Side length of each node.

    depth : int
        Number of levels of the tree below the root nodes.

    work_units : ~numpy.ndarray
        Node of each work unit.

    work_unit_ptr : ~numpy.ndarray
        Location in ``work_unit_particles`` where the candidates of each work unit start.

    work_unit_particles : ~numpy.ndarray
        Candidate search coordinates of the ``work_units``, in ascending order for each.

    block_edges : ~numpy.ndarray
        In parallel execution, ``work_units`` are processed in load-balanced blocks
        ``block_edges[b]:block_edges[b + 1]``.

    record : bool
        If ``True``, record the hits, otherwise count them.

    work_unit_hits : ~numpy.ndarray
        Output array for the number of hits below each work unit (if not ``record``).

    cell_counts : ~numpy.ndarray
        Output array for the number of hits of each cell (if not ``record``).

    cursor : ~numpy.ndarray
        Location in ``intersections`` where the hits of each cell start (if ``record``,
        modified).

    intersections : ~numpy.ndarray
        Output array for the index of the search coordinate of each hit (if ``record``).

    distances : ~numpy.ndarray
        Output array for the offset from the cell centre to the search coordinate of
        each hit, shape ``(M, 3)`` (if ``record`` and ``fill_distances``).

    fill_distances : bool
        If ``True``, fill ``distances``.
    """
    mode = _WALK_RECORD if record else _WALK_COUNT_CELLS
    for b in prange(block_edges.size - 1):
        stack = np.empty(8 * (depth + 1), dtype=np.int64)
        for u in range(block_edges[b], block_edges[b + 1]):
            hits = 0
            for m in range(work_unit_ptr[u], work_unit_ptr[u + 1]):
                p = work_unit_particles[m]
                hits += _walk_subtree(
                    work_units[u],
                    p,
                    coords[p, 0],
                    coords[p, 1],
                    coords[p, 2],
                    radii[p] * radii[p],
                    cell_centres,
                    first_child,
                    node_cell,
                    node_corner,
                    node_size,
                    stack,
                    mode,
                    cell_counts,
                    cursor,
                    intersections,
                    distances,
                    fill_distances,
                )
            if not record:
                work_unit_hits[u] = hits


def _cells_below(first_child: np.ndarray, node_cell: np.ndarray) -> np.ndarray:
    """
    Count the cells below each node of a tree (``1`` for leaves).

    Children are stored after their parent (see :meth:`Octree._build_octree`), so nodes
    are processed in reverse order.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1``.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1``.

    Returns
    -------
    ~numpy.ndarray
        The number of cells below each node.
    """
    n_below = np.zeros(first_child.size, dtype=np.int64)
    for node in range(first_child.size - 1, -1, -1):
        if node_cell[node] >= 0:
            n_below[node] = 1
        else:
            for child in range(8):
                n_below[node] += n_below[first_child[node] + child]
    return n_below


def _containing_cells_loop(
    coords: np.ndarray,
    origin: np.ndarray,
    root_size: float,
    root_shape: np.ndarray,
    root_nodes: np.ndarray,
    first_child: np.ndarray,
    node_cell: np.ndarray,
    node_corner: np.ndarray,
    node_size: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Find the cell containing each coordinate by walking down the tree.

    The root node containing a coordinate is found on the root lattice, then the tree is
    descended to the leaf containing the coordinate, choosing the child octant by
    comparing the coordinate with the centre of each node. Cells are half-open (they
    include their lower faces), as are the octants of the nodes.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Coordinates to place in cells, shape ``(N, 3)``.

    origin : ~numpy.ndarray
        Lower corner of the lattice of root nodes.

    root_size : float
        Side length of the root nodes.

    root_shape : ~numpy.ndarray
        Number of root nodes along each axis.

    root_nodes : ~numpy.ndarray
        Node at each (flattened, C-order) position of the root lattice, or ``-1``.

    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1``.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1``.

    node_corner : ~numpy.ndarray
        Lower corner of each node, shape ``(n_nodes, 3)``.

    node_size : ~numpy.ndarray
        Side length of each node.

    out : ~numpy.ndarray
        Output, the index of the cell containing each coordinate or ``-1`` for
        coordinates outside of all cells, shape ``(N,)``.
    """
    for p in prange(coords.shape[0]):
        flat = 0
        inside = True
        for d in range(3):
            # compare as floats before casting to catch coordinates far outside (or nan):
            f = (coords[p, d] - origin[d]) / root_size
            if not (f >= 0 and f < root_shape[d]):
                inside = False
                break
            flat = flat * root_shape[d] + int(math.floor(f))
        node = root_nodes[flat] if inside else -1
        if node < 0:
            out[p] = -1
            continue
        while first_child[node] >= 0:
            half = node_size[node] / 2
            octant = 0
            for d in range(3):
                octant = 2 * octant + (coords[p, d] >= node_corner[node, d] + half)
            node = first_child[node] + octant
        out[p] = node_cell[node]


if NUMBA_AVAILABLE:
    _line_interval = njit(cache=True, inline="always")(_line_interval)
    _candidate_rows = njit(cache=True, inline="always")(_candidate_rows)
    _line_perp = njit(cache=True, inline="always")(_line_perp)
    _count_hits = njit(parallel=True, cache=True)(_count_hits)
    _build_line_candidates = njit(cache=True)(_build_line_candidates)
    _count_line_hits = njit(parallel=True, cache=True)(_count_line_hits)
    _fill_lines = njit(parallel=True, cache=True)(_fill_lines)
    _box_distance2 = njit(cache=True, inline="always")(_box_distance2)
    _root_range = njit(cache=True, inline="always")(_root_range)
    _walk_subtree = njit(cache=True)(_walk_subtree)
    _walk_from_roots = njit(cache=True)(_walk_from_roots)
    _count_from_roots = njit(parallel=True, cache=True)(_count_from_roots)
    _fill_work_unit_candidates = njit(parallel=True, cache=True)(
        _fill_work_unit_candidates
    )
    _group_by_work_unit = njit(cache=True)(_group_by_work_unit)
    _work_unit_hits = njit(parallel=True, cache=True)(_work_unit_hits)
    _cells_below = njit(cache=True)(_cells_below)
    _containing_cells_loop = njit(parallel=True, cache=True)(_containing_cells_loop)


def _balanced_blocks(weights: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Divide a set of weights into a number of blocks of approximately equal total weight.

    Parameters
    ----------
    weights : ~numpy.ndarray
        Weight of each item.

    n_blocks : int
        Target number of blocks.

    Returns
    -------
    ~numpy.ndarray
        Block edges: block ``b`` contains items ``edges[b]:edges[b + 1]``.
    """
    cumulative = np.cumsum(weights)
    total = cumulative[-1] if cumulative.size > 0 else 0
    targets = np.linspace(0, total, n_blocks + 1)[1:-1]
    inner = np.searchsorted(cumulative, targets, side="left") + 1
    return np.unique(np.concatenate(([0], inner, [weights.size]))).astype(np.int64)


def _prepare_grid_inputs(
    grid_shape: Sequence[int],
    coords: np.ndarray,
    radii: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """
    Validate and normalize inputs to the grid search functions.

    Parameters
    ----------
    grid_shape : tuple or list
        Number of pixels or cells along each axis (length 2 or 3).

    coords : ~numpy.ndarray
        Search coordinates with implicit units of pixels, one column per dimension.

    radii : ~numpy.ndarray
        Search radii with implicit units of pixels, one per row of ``coords``. A constant
        (e.g. scalar or broadcastable array) is not accepted.

    Returns
    -------
    tuple
        The pixel or cell grid shape as a tuple of ints, and C-contiguous, ``float64``
        copies (if needed) of ``coords`` and ``radii``.
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
    grid_shape: Sequence[int],
    coords: np.ndarray,
    radii: np.ndarray,
    ncpu: int = 1,
) -> np.ndarray:
    r"""
    Count the uniform pixel or cell grid cell centres within the search radii.

    The grid is assumed to be uniform with square pixels (2D) or cubic cells (3D) of unit
    size. Cell centres are located at integer coordinates, and cells are indexed by their
    position in an array of shape ``grid_shape``.

    A search coordinate with a small radius can slip "between" cell centres if the sphere
    does not enclose any cell centres. This can be prevented by flooring the search radius
    at :math:`\sqrt{n_\mathrm{dim}}/2`.

    In the context of :mod:`martini` these particles will normally evaluate the
    Dirac-delta kernel which will ensure that all of their contribution is deposited in
    the grid cell containing them, even if their artificially increased search radius
    reaches multiple grid cell centres.

    This function to count cell "hits" is much cheaper than recording them with
    :func:`find_grid_intersections` (the cost scales with the number of grid lines
    crossed, not the number of cell "hits", and no per-match output is stored), so it can
    be used to plan memory usage in advance. Knowing how many hits will be found also
    enables parallel recording of the hits without write collisions.

    Parameters
    ----------
    grid_shape : tuple or list
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

    See Also
    --------
    find_grid_intersections, Octree
    """
    grid_shape, coords, radii = _prepare_grid_inputs(grid_shape, coords, radii)
    with _threads(ncpu):
        return _count_hits(coords, radii, np.array(grid_shape, dtype=np.int64))


def _empty_result(
    ndim: int, dist_dtype: type, return_distances: bool
) -> FindGridIntersectionsResult:
    """
    Make the result of a search without matches.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the coordinates.

    dist_dtype : type
        Data type of the ``distances``.

    return_distances : bool
        If ``False``, ``distances`` is ``None``.

    Returns
    -------
    ~martini._grid_search.FindGridIntersectionsResult
        Empty arrays with the data types and number of columns of a search with matches.
    """
    return FindGridIntersectionsResult(
        intersections=np.empty(0, dtype=np.int32),
        distances=np.empty((0, ndim), dtype=dist_dtype) if return_distances else None,
        cell_indices=np.empty(0, dtype=np.int64),
        strides=np.empty((0, 2), dtype=np.int64),
    )


def find_grid_intersections(
    grid_shape: Sequence[int],
    coords: np.ndarray,
    radii: np.ndarray,
    ncpu: int = 1,
    dist_dtype: type = np.float64,
    return_distances: bool = True,
) -> FindGridIntersectionsResult:
    r"""
    Search for coordinates that reach uniform grid locations within variable radii.

    The grid is assumed to be uniform with square pixels (2D) or cubic cells (3D) of unit
    size. Cell centres are located at integer coordinates, and cells are indexed by their
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
    grid_shape : tuple or list
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
        Data type for returned ``distances`` array, can be used to manage memory usage.

    return_distances : bool, optional
        If ``False``, the offset vectors between cell centres and search coordinates are
        not computed (``distances`` is ``None``), saving memory. They can be recomputed
        from ``coords``, ``intersections`` and ``cell_indices`` when needed.

    Returns
    -------
    intersections : ~numpy.ndarray
        Each entry is the index of a search coordinate (index into ``coords``),
        representing a search coordinate that "hit" a cell. Each search coordinate may
        match to many cells and therefore appear multiple times. Results are grouped by
        the cell indices, and in ascending particle (search coordinate) index order within
        each cell.

    distances : ~numpy.ndarray or None
        The vector offset from the cell centre to the search coordinate for each match,
        with one column per dimension. ``None`` if ``return_distances`` is ``False``.

    cell_indices : ~numpy.ndarray
        The unique (flattened) cell indices, in ascending order. Each entry in this array
        corresponds to a range in ``strides``.

    strides : ~numpy.ndarray
        Each row can be used as a range to select rows from ``intersections`` that share
        a common cell index.

    See Also
    --------
    count_grid_intersections, Octree

    Notes
    -----
    The pixel/voxel grid is treated as a set of "lines" (rows in 2D, columns along the
    third axis in 3D) that we iterate over. A sphere intersects a line in a contiguous
    interval of grid cells, found directly from the chord half-width. The interval is
    widened by one cell at each end as a guard against roundoff and then
    trimmed with exact distance calculations to decide whether the cell is "hit".

    Since cell "hits" are counted in advance, their locations in the output arrays are
    known and we can safely process "lines" in parallel without risking write collisions.
    Output is grouped by cell, and within each cell matches are ordered by ascending
    particle index.
    """
    grid_shape, coords, radii = _prepare_grid_inputs(grid_shape, coords, radii)
    ndim = len(grid_shape)
    shape = np.array(grid_shape, dtype=np.int64)
    n_lines = math.prod(grid_shape[:-1])
    with _threads(ncpu):
        n_blocks = 8 * (numba.get_num_threads() if NUMBA_AVAILABLE else 1)
        line_ptr, line_particles = _build_line_candidates(coords, radii, shape)
        line_hits = np.empty(n_lines, dtype=np.int64)
        line_cells_hit = np.empty(n_lines, dtype=np.int64)
        _count_line_hits(
            coords,
            radii,
            shape,
            line_ptr,
            line_particles,
            _balanced_blocks(np.diff(line_ptr) + grid_shape[-1], n_blocks),
            line_hits,
            line_cells_hit,
        )
        hit_ptr = np.zeros(n_lines + 1, dtype=np.int64)
        np.cumsum(line_hits, out=hit_ptr[1:])
        cell_ptr = np.zeros(n_lines + 1, dtype=np.int64)
        np.cumsum(line_cells_hit, out=cell_ptr[1:])
        total_intersections = int(hit_ptr[-1])

        if total_intersections == 0:
            return _empty_result(ndim, dist_dtype, return_distances)

        intersections = np.empty(total_intersections, dtype=np.int32)
        cell_indices = np.empty(int(cell_ptr[-1]), dtype=np.int64)
        strides = np.empty((int(cell_ptr[-1]), 2), dtype=np.int64)
        distances: np.ndarray = np.empty(
            (total_intersections if return_distances else 1, ndim), dtype=dist_dtype
        )
        _fill_lines(
            coords,
            radii,
            shape,
            line_ptr,
            line_particles,
            _balanced_blocks(line_hits + 1, n_blocks),
            hit_ptr,
            cell_ptr,
            intersections,
            cell_indices,
            strides,
            distances,
            return_distances,
        )

    return FindGridIntersectionsResult(
        intersections=intersections,
        distances=distances if return_distances else None,
        cell_indices=cell_indices,
        strides=strides,
    )


class _OctreeArrays(NamedTuple):
    """
    Provide the arrays describing an octree of cells, see :meth:`Octree._build_octree`.

    Attributes
    ----------
    origin : ~numpy.ndarray
        Lower corner of the lattice of root nodes.

    root_size : float
        Side length of the root nodes (the largest cell size).

    root_shape : ~numpy.ndarray
        Number of root nodes along each axis.

    root_nodes : ~numpy.ndarray
        Node at each (flattened) position of the root lattice, or ``-1`` where there is
        no cell.

    first_child : ~numpy.ndarray
        Index of the first of the 8 consecutive children of each node, or ``-1`` for
        leaves.

    node_cell : ~numpy.ndarray
        Index of the cell of each leaf node, or ``-1`` for internal nodes.

    node_corner : ~numpy.ndarray
        Lower corner of each node, shape ``(n_nodes, 3)``.

    node_size : ~numpy.ndarray
        Side length of each node.

    depth : int
        Number of levels of the tree below the root nodes.
    """

    origin: np.ndarray
    root_size: float
    root_shape: np.ndarray
    root_nodes: np.ndarray
    first_child: np.ndarray
    node_cell: np.ndarray
    node_corner: np.ndarray
    node_size: np.ndarray
    depth: int


class Octree:
    """
    The tree of the cells of an adaptive (octree) grid, to search for cells near points.

    The cells are cubes obtained by recursively bisecting a uniform grid of cubes: the
    largest cells have size ``S``, every cell has a size ``S / 2**k`` and its lower corner
    on a lattice of its own size starting from the lowest cell corner, and the cells tile
    a region without overlapping. The tree is built once, and can then be searched around
    several sets of coordinates (e.g. batches of particles) with :meth:`count` and
    :meth:`find`, and the cell containing each of a set of points found with
    :meth:`containing_cells`.

    For the simpler case of a one-level tree (a uniform cell grid), see
    :func:`find_grid_intersections` and :func:`count_grid_intersections`.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Coordinates of the cell centres, shape ``(n_cells, 3)``.

    cell_sizes : ~numpy.ndarray
        Side lengths of the cells (same implicit units as ``cell_centres``).

    Attributes
    ----------
    cell_centres : ~numpy.ndarray
        C-contiguous, ``float64`` copy (if needed) of the cell centres.

    cell_sizes : ~numpy.ndarray
        ``float64`` copy (if needed) of the cell sizes.

    Raises
    ------
    ValueError
        If the cells do not form an octree as described above.

    See Also
    --------
    find_grid_intersections, count_grid_intersections

    Notes
    -----
    The cells are divided into work units that each own the cells below a node of the
    tree (busy nodes are split, so that no work unit holds a large share of the cells),
    as each grid line owns its cells in :func:`find_grid_intersections`. The tree is
    traversed from the largest cells down for each coordinate, skipping nodes whose
    cube is beyond the search radius, to list the candidate coordinates of each work
    unit. The work units then count the hits of their cells, and record them once their
    locations in the output arrays are known, in parallel without risking write
    collisions. Output is grouped by cell, and within each cell matches are ordered by
    ascending search coordinate index.

    Work units are never larger than the largest cells, so many large cells that are not
    bisected (e.g. a uniform grid, better searched with
    :func:`find_grid_intersections`) make the candidate lists nearly as long as the list
    of matches, which costs time and memory.
    """

    def __init__(self, cell_centres: np.ndarray, cell_sizes: np.ndarray) -> None:
        cell_centres = np.ascontiguousarray(cell_centres, dtype=np.float64)
        cell_sizes = np.ascontiguousarray(cell_sizes, dtype=np.float64)
        if cell_centres.ndim != 2 or cell_centres.shape[1] != 3:
            raise ValueError("cell_centres must have shape (n_cells, 3).")
        if cell_sizes.shape != (cell_centres.shape[0],):
            raise ValueError("cell_sizes must have one entry per row of cell_centres.")
        self.cell_centres = cell_centres
        self.cell_sizes = cell_sizes
        self._octree = Octree._build_octree(cell_centres, cell_sizes)
        # number of cells below each node:
        self._n_below = _cells_below(self._octree.first_child, self._octree.node_cell)

    @classmethod
    def _from_arrays(
        cls, octree: _OctreeArrays, cell_centres: np.ndarray, cell_sizes: np.ndarray
    ) -> "Octree":
        """
        Create the tree from arrays that are already built, without validation.

        Used when the tree is constructed along with the cells (e.g. by grid
        refinement), instead of rebuilding it from the cells.

        Parameters
        ----------
        octree : _OctreeArrays
            The tree, in the layout of :meth:`_build_octree`.

        cell_centres : ~numpy.ndarray
            Coordinates of the cell centres (the leaves of the tree, ordered by their
            ``node_cell``), shape ``(n_cells, 3)``.

        cell_sizes : ~numpy.ndarray
            Side lengths of the cells.

        Returns
        -------
        Octree
            The tree.
        """
        tree = cls.__new__(cls)
        tree.cell_centres = np.ascontiguousarray(cell_centres, dtype=np.float64)
        tree.cell_sizes = np.ascontiguousarray(cell_sizes, dtype=np.float64)
        tree._octree = octree
        tree._n_below = _cells_below(octree.first_child, octree.node_cell)
        return tree

    def count(self, coords: np.ndarray, radii: np.ndarray, ncpu: int = 1) -> np.ndarray:
        """
        Count the cell centres within the search radii.

        This is cheaper than recording the hits with :meth:`find` (no per-match output is
        stored), so it can be used to plan memory usage in advance.

        Parameters
        ----------
        coords : ~numpy.ndarray
            Coordinates to search around for cells, shape ``(N, 3)`` (same implicit units
            as the cell centres).

        radii : ~numpy.ndarray
            Search radii around the ``coords``. Must be an array with the same length as
            the number of coordinates, a scalar is not accepted.

        ncpu : int
            Number of threads to use.

        Returns
        -------
        ~numpy.ndarray
            Number of cell centres within the search radius of each coordinate.
        """
        coords, radii = self._prepare_coords(coords, radii)
        with _threads(ncpu):
            return self._counts_per_coordinate(coords, radii)

    def containing_cells(self, coords: np.ndarray, ncpu: int = 1) -> np.ndarray:
        """
        Find the cell containing each coordinate.

        Cells are half-open: they contain their lower faces but not their upper faces.

        Parameters
        ----------
        coords : ~numpy.ndarray
            Coordinates to place in cells, shape ``(N, 3)`` (same implicit units as the
            cell centres).

        ncpu : int
            Number of threads to use.

        Returns
        -------
        ~numpy.ndarray
            Index of the cell containing each coordinate, or ``-1`` for coordinates
            outside of all cells.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3).")
        octree = self._octree
        out = np.empty(coords.shape[0], dtype=np.intp)
        with _threads(ncpu):
            _containing_cells_loop(
                coords,
                octree.origin,
                octree.root_size,
                octree.root_shape,
                octree.root_nodes,
                octree.first_child,
                octree.node_cell,
                octree.node_corner,
                octree.node_size,
                out,
            )
        return out

    def find(
        self,
        coords: np.ndarray,
        radii: np.ndarray,
        ncpu: int = 1,
        dist_dtype: type = np.float64,
        return_distances: bool = True,
    ) -> FindGridIntersectionsResult:
        """
        Search for coordinates that reach the cell centres.

        A cell is "hit" if its centre is within the search radius (inclusive) of a
        coordinate.

        Parameters
        ----------
        coords : ~numpy.ndarray
            Coordinates to search around for cells, shape ``(N, 3)`` (same implicit units
            as the cell centres).

        radii : ~numpy.ndarray
            Search radii around the ``coords``. Must be an array with the same length as
            the number of coordinates, a scalar is not accepted.

        ncpu : int
            Number of threads to use for the search.

        dist_dtype : type, optional
            Data type for returned ``distances`` array, can be used to manage memory
            usage.

        return_distances : bool, optional
            If ``False``, the offset vectors between cell centres and search coordinates
            are not computed (``distances`` is ``None``), saving memory.

        Returns
        -------
        intersections : ~numpy.ndarray
            Each entry is the index of a search coordinate (index into ``coords``),
            representing a search coordinate that "hit" a cell. Results are grouped by
            the cell indices, and in ascending search coordinate index order within each
            cell.

        distances : ~numpy.ndarray or None
            The vector offset from the cell centre to the search coordinate for each
            match. ``None`` if ``return_distances`` is ``False``.

        cell_indices : ~numpy.ndarray
            The unique cell indices (index into the cells), in ascending order. Each
            entry in this array corresponds to a range in ``strides``.

        strides : ~numpy.ndarray
            Each row can be used as a range to select rows from ``intersections`` that
            share a common cell index.
        """
        octree, cell_centres = self._octree, self.cell_centres
        coords, radii = self._prepare_coords(coords, radii)
        n_cells = cell_centres.shape[0]
        n_coords = coords.shape[0]
        with _threads(ncpu):
            n_threads = numba.get_num_threads() if NUMBA_AVAILABLE else 1
            n_blocks = 8 * n_threads
            n_below = self._n_below
            work_units = Octree._work_units(
                octree, n_below, max(1, n_cells // (64 * n_threads))
            )
            work_unit_of_node = np.full(octree.first_child.size, -1, dtype=np.int64)
            work_unit_of_node[work_units] = np.arange(work_units.size)
            # candidate coordinates of each work unit, in ascending order:
            candidate_counts = self._counts_per_coordinate(
                coords, radii, work_unit_of_node
            )
            candidate_ptr = np.zeros(n_coords + 1, dtype=np.int64)
            np.cumsum(candidate_counts, out=candidate_ptr[1:])
            candidate_work_units = np.empty(int(candidate_ptr[-1]), dtype=np.int32)
            if n_coords > 0:
                _fill_work_unit_candidates(
                    coords,
                    radii,
                    cell_centres,
                    *octree,
                    work_unit_of_node,
                    _balanced_blocks(candidate_counts + 1, n_blocks),
                    candidate_ptr,
                    candidate_work_units,
                )
            work_unit_ptr = np.zeros(work_units.size + 1, dtype=np.int64)
            np.cumsum(
                np.bincount(candidate_work_units, minlength=work_units.size),
                out=work_unit_ptr[1:],
            )
            work_unit_particles = np.empty(candidate_work_units.size, dtype=np.int32)
            _group_by_work_unit(
                candidate_ptr,
                candidate_work_units,
                work_unit_ptr[:-1].copy(),
                work_unit_particles,
            )
            del candidate_work_units
            work_unit_args = (
                coords,
                radii,
                cell_centres,
                octree.first_child,
                octree.node_cell,
                octree.node_corner,
                octree.node_size,
                octree.depth,
                work_units,
                work_unit_ptr,
                work_unit_particles,
            )
            work_unit_hits = np.zeros(work_units.size, dtype=np.int64)
            cell_ptr = np.zeros(n_cells + 1, dtype=np.int64)
            cell_counts = cell_ptr[1:]  # counted in place, then accumulated
            placeholder_int = np.empty(0, dtype=np.int32)
            _work_unit_hits(
                *work_unit_args,
                _balanced_blocks(
                    np.diff(work_unit_ptr) + n_below[work_units], n_blocks
                ),
                False,
                work_unit_hits,
                cell_counts,
                cell_ptr,
                placeholder_int,
                np.empty((0, 3), dtype=dist_dtype),
                False,
            )
            cell_indices = np.flatnonzero(cell_counts).astype(np.int64)
            np.cumsum(cell_counts, out=cell_counts)
            total_intersections = int(cell_ptr[-1])
            if total_intersections == 0:
                return _empty_result(3, dist_dtype, return_distances)
            strides = np.column_stack(
                (cell_ptr[cell_indices], cell_ptr[cell_indices + 1])
            )
            intersections = np.empty(total_intersections, dtype=np.int32)
            distances: np.ndarray = np.empty(
                (total_intersections if return_distances else 1, 3), dtype=dist_dtype
            )
            _work_unit_hits(
                *work_unit_args,
                _balanced_blocks(work_unit_hits + 1, n_blocks),
                True,
                work_unit_hits,
                cell_counts,
                cell_ptr[:-1].copy(),
                intersections,
                distances,
                return_distances,
            )
        return FindGridIntersectionsResult(
            intersections=intersections,
            distances=distances if return_distances else None,
            cell_indices=cell_indices,
            strides=strides,
        )

    def _prepare_coords(
        self, coords: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and normalize the coordinates and radii of a search.

        Parameters
        ----------
        coords : ~numpy.ndarray
            Search coordinates, shape ``(N, 3)``.

        radii : ~numpy.ndarray
            Search radii, one per row of ``coords``.

        Returns
        -------
        tuple
            C-contiguous, ``float64`` copies (if needed) of ``coords`` and ``radii``, with
            infinite radii capped.
        """
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3).")
        if radii.shape != (coords.shape[0],):
            raise ValueError("radii must have one entry per row of coords.")
        # Radii can be infinite, cap them at a distance that reaches every cell so that
        # the range of root nodes is finite:
        octree = self._octree
        extent = octree.root_shape * octree.root_size
        reach_all = np.sqrt(
            np.sum((np.abs(coords - octree.origin) + extent) ** 2, axis=1)
        )
        return coords, np.minimum(radii, reach_all)

    def _counts_per_coordinate(
        self,
        coords: np.ndarray,
        radii: np.ndarray,
        work_unit_of_node: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Count the cells (or work units) within the search radius of each coordinate.

        Must be called within :func:`_threads`.

        Parameters
        ----------
        coords : ~numpy.ndarray
            Search coordinates, shape ``(N, 3)``.

        radii : ~numpy.ndarray
            Search radii, shape ``(N,)``.

        work_unit_of_node : ~numpy.ndarray, optional
            Index of the work unit of each node, or ``-1`` (see :meth:`_work_units`). If
            provided, the work units that each search radius may reach are counted instead
            of the cells.

        Returns
        -------
        ~numpy.ndarray
            Number of cell centres (or work units) within the search radius of each
            coordinate.
        """
        if work_unit_of_node is None:
            work_unit_of_node = np.empty(0, dtype=np.int64)
        counts = np.zeros(coords.shape[0], dtype=np.int64)
        if coords.shape[0] > 0:
            n_blocks = 8 * (numba.get_num_threads() if NUMBA_AVAILABLE else 1)
            _count_from_roots(
                coords,
                radii,
                self.cell_centres,
                *self._octree,
                work_unit_of_node,
                _balanced_blocks(np.ones(coords.shape[0]), n_blocks),
                counts,
            )
        return counts

    @staticmethod
    def _build_octree(
        cell_centres: np.ndarray, cell_sizes: np.ndarray
    ) -> _OctreeArrays:
        """
        Build the octree of a grid of cubic cells obtained by recursive bisection.

        The largest cells (size ``S``) and the cells of each size ``S / 2**k`` are
        aligned with lattices of their own size, starting from the lowest cell corner,
        and the cells tile a region without overlapping. Nodes of the tree are the cells
        (leaves) and the cubes that were bisected to obtain them (internal nodes). The
        tree is built top-down, one level at a time, with the 8 children of a node stored
        consecutively.

        Parameters
        ----------
        cell_centres : ~numpy.ndarray
            Cell centres, shape ``(n_cells, 3)``.

        cell_sizes : ~numpy.ndarray
            Cell side lengths, shape ``(n_cells,)``.

        Returns
        -------
        _OctreeArrays
            The tree.

        Raises
        ------
        ValueError
            If the cells do not form an octree.
        """
        if cell_sizes.size == 0:
            raise ValueError("Cannot build a tree without cells.")
        root_size = float(cell_sizes.max())
        cell_levels = np.rint(np.log2(root_size / cell_sizes)).astype(np.int64)
        if np.any(root_size / 2.0**cell_levels != cell_sizes):
            raise ValueError(
                "Cell sizes must be the largest cell size divided by powers of two."
            )
        corners = cell_centres - cell_sizes[:, np.newaxis] / 2
        origin = corners.min(axis=0)
        lattice_coords = (corners - origin) / cell_sizes[:, np.newaxis]
        cell_ijk = np.rint(lattice_coords).astype(np.int64)
        if np.any(np.abs(lattice_coords - cell_ijk) > 1e-6):
            raise ValueError(
                "Cell corners must lie on a lattice of the cell size, starting from the "
                "lowest cell corner."
            )
        depth = int(cell_levels.max())
        root_shape = (cell_ijk >> cell_levels[:, np.newaxis]).max(axis=0) + 1
        if np.log2(np.prod(root_shape.astype(float))) + 3 * depth > 62:
            raise ValueError("Too many levels of refinement to index the cells.")
        child_offsets = np.array(
            [(dx, dy, dz) for dx in range(2) for dy in range(2) for dz in range(2)]
        )
        first_child, node_cell, node_corner, node_size = [], [], [], []
        n_nodes = 0
        node_ijk = np.empty((0, 3), dtype=np.int64)
        is_internal = np.empty(0, dtype=bool)
        root_nodes = np.empty(0, dtype=np.int64)
        for level in range(depth + 1):
            level_shape = root_shape << level
            level_size = root_size / 2.0**level
            level_cells = np.flatnonzero(cell_levels == level)
            # positions of the nodes of this level on its lattice (flattened):
            leaf_positions = Octree._flat_lattice_indices(
                cell_ijk[level_cells], level_shape
            )
            leaf_order = np.argsort(leaf_positions)
            leaf_positions = leaf_positions[leaf_order]
            if np.any(np.diff(leaf_positions) == 0):
                raise ValueError("Cells must not be duplicated.")
            deeper_cells = cell_levels > level
            internal_positions = np.unique(
                Octree._flat_lattice_indices(
                    cell_ijk[deeper_cells]
                    >> (cell_levels[deeper_cells] - level)[:, np.newaxis],
                    level_shape,
                )
            )
            if level == 0:
                root_positions = np.union1d(leaf_positions, internal_positions)
                node_ijk = np.column_stack(
                    np.unravel_index(root_positions, tuple(level_shape))
                )
            else:
                node_ijk = (
                    2 * node_ijk[is_internal, np.newaxis, :] + child_offsets[np.newaxis]
                ).reshape(-1, 3)
            node_positions = Octree._flat_lattice_indices(node_ijk, level_shape)
            leaf_locations, is_leaf = Octree._sorted_lookup(
                leaf_positions, node_positions
            )
            _, is_internal = Octree._sorted_lookup(internal_positions, node_positions)
            if np.any(is_leaf & is_internal):
                raise ValueError("Cells must not overlap.")
            if np.any(~is_leaf & ~is_internal):
                raise ValueError(
                    "Cells must tile the region covered by the largest cells (each "
                    "bisected cell must be replaced by all 8 of its octants)."
                )
            n_level = node_positions.size
            next_offset = n_nodes + n_level
            level_first_child = np.full(n_level, -1, dtype=np.int64)
            level_first_child[is_internal] = next_offset + 8 * np.arange(
                np.count_nonzero(is_internal)
            )
            level_cell = np.full(n_level, -1, dtype=np.int64)
            level_cell[is_leaf] = level_cells[leaf_order[leaf_locations[is_leaf]]]
            first_child.append(level_first_child)
            node_cell.append(level_cell)
            node_corner.append(origin + node_ijk * level_size)
            node_size.append(np.full(n_level, level_size))
            if level == 0:
                root_nodes = np.full(np.prod(root_shape), -1, dtype=np.int64)
                root_nodes[node_positions] = np.arange(n_level)
            n_nodes = next_offset
        return _OctreeArrays(
            origin=origin,
            root_size=root_size,
            root_shape=root_shape,
            root_nodes=root_nodes,
            first_child=np.concatenate(first_child),
            node_cell=np.concatenate(node_cell),
            node_corner=np.ascontiguousarray(np.concatenate(node_corner)),
            node_size=np.concatenate(node_size),
            depth=depth,
        )

    @staticmethod
    def _work_units(
        octree: _OctreeArrays, n_below: np.ndarray, max_cells: int
    ) -> np.ndarray:
        """
        Divide the cells of a tree into work units that each own the cells below a node.

        Starting from the root nodes, nodes with more than ``max_cells`` cells below them
        are replaced by their children, so that work is shared out evenly even where cells
        are concentrated in a few root nodes. Every cell is below exactly one work unit.

        Parameters
        ----------
        octree : _OctreeArrays
            The tree of cells.

        n_below : ~numpy.ndarray
            Number of cells below each node, see :func:`_cells_below`.

        max_cells : int
            Largest number of cells below a work unit, unless it is a leaf.

        Returns
        -------
        ~numpy.ndarray
            The node of each work unit, in ascending order.
        """
        work_units = octree.root_nodes[octree.root_nodes >= 0]
        while True:
            to_split = (octree.first_child[work_units] >= 0) & (
                n_below[work_units] > max_cells
            )
            if not np.any(to_split):
                return np.sort(work_units)
            child_nodes = octree.first_child[
                work_units[to_split], np.newaxis
            ] + np.arange(8)
            work_units = np.concatenate((work_units[~to_split], child_nodes.ravel()))

    @staticmethod
    def _flat_lattice_indices(ijk: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """
        Flatten 3D lattice indices into positions on the lattice.

        Parameters
        ----------
        ijk : ~numpy.ndarray
            Lattice indices, shape ``(N, 3)``.

        shape : ~numpy.ndarray
            Shape of the lattice.

        Returns
        -------
        ~numpy.ndarray
            The flattened lattice positions.
        """
        return np.ravel_multi_index(tuple(ijk.T), tuple(shape))

    @staticmethod
    def _sorted_lookup(
        sorted_values: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Look up values in a sorted array of values.

        Parameters
        ----------
        sorted_values : ~numpy.ndarray
            Values to search, in ascending order.

        values : ~numpy.ndarray
            Values to look up.

        Returns
        -------
        tuple
            The location in ``sorted_values`` of each value (meaningful only where found),
            and whether each value was found.
        """
        if sorted_values.size == 0:
            return np.zeros(values.size, dtype=np.intp), np.zeros(
                values.size, dtype=bool
            )
        locations = np.minimum(
            np.searchsorted(sorted_values, values), sorted_values.size - 1
        )
        return locations, sorted_values[locations] == values
