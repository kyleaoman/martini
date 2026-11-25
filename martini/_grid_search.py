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
    h = math.sqrt(max(r2 - dperp2, 0.0))
    lo = max(0, math.ceil(coord - h) - 1)
    hi = min(line_cells - 1, math.floor(coord + h) + 1)
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
    n_last = shape[ndim - 1]
    n = coords.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for p in prange(n):
        x = coords[p, 0]
        c = coords[p, ndim - 1]
        r2 = radii[p] * radii[p]
        hits = 0
        i0, i1 = _line_interval(x, r2, 0.0, shape[0])
        for i in range(i0, i1 + 1):
            dx2 = (x - i) * (x - i)
            if ndim == 2:
                lo, hi = _line_interval(c, r2, dx2, n_last)
                hits += max(hi - lo + 1, 0)
            else:
                y = coords[p, 1]
                j0, j1 = _line_interval(y, r2, dx2, shape[1])
                for j in range(j0, j1 + 1):
                    lo, hi = _line_interval(c, r2, dx2 + (y - j) * (y - j), n_last)
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
    diff = np.zeros(n_lines + 1, dtype=np.int64)
    for p in range(coords.shape[0]):
        i0, i1 = _candidate_rows(p, coords, radii, shape)
        if ndim == 2:
            if i1 >= i0:
                diff[i0] += 1
                diff[i1 + 1] -= 1
        else:
            x = coords[p, 0]
            r2 = radii[p] * radii[p]
            for i in range(i0, i1 + 1):
                j0, j1 = _line_interval(coords[p, 1], r2, (x - i) * (x - i), shape[1])
                if j1 >= j0:
                    diff[i * shape[1] + j0] += 1
                    diff[i * shape[1] + j1 + 1] -= 1
    line_ptr = np.zeros(n_lines + 1, dtype=np.int64)
    line_ptr[1:] = np.cumsum(diff[:-1])
    line_ptr[1:] = np.cumsum(line_ptr[1:])
    line_particles = np.empty(line_ptr[-1], dtype=np.int32)
    offset = line_ptr[:-1].copy()
    for p in range(coords.shape[0]):
        i0, i1 = _candidate_rows(p, coords, radii, shape)
        if ndim == 2:
            for i in range(i0, i1 + 1):
                line_particles[offset[i]] = p
                offset[i] += 1
        else:
            x = coords[p, 0]
            r2 = radii[p] * radii[p]
            for i in range(i0, i1 + 1):
                j0, j1 = _line_interval(coords[p, 1], r2, (x - i) * (x - i), shape[1])
                for line in range(i * shape[1] + j0, i * shape[1] + j1 + 1):
                    line_particles[offset[line]] = p
                    offset[line] += 1
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
    n_last = shape[shape.size - 1]
    for b in prange(block_edges.size - 1):
        diff = np.zeros(n_last + 1, dtype=np.int64)
        for line in range(block_edges[b], block_edges[b + 1]):
            hits = 0
            kmin = n_last
            kmax = -1
            for m in range(line_ptr[line], line_ptr[line + 1]):
                p = line_particles[m]
                _, _, dperp2 = _line_perp(line, p, coords, shape)
                r2 = radii[p] * radii[p]
                lo, hi = _line_interval(coords[p, shape.size - 1], r2, dperp2, n_last)
                if hi >= lo:
                    hits += hi - lo + 1
                    diff[lo] += 1
                    diff[hi + 1] -= 1
                    kmin = min(kmin, lo)
                    kmax = max(kmax, hi)
            cells_hit = 0
            cover = 0
            for k in range(kmin, kmax + 1):
                cover += diff[k]
                if cover > 0:
                    cells_hit += 1
                diff[k] = 0
            if kmax >= 0:
                diff[kmax + 1] = 0
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
    n_last = shape[ndim - 1]
    for b in prange(block_edges.size - 1):
        diff = np.zeros(n_last + 1, dtype=np.int64)
        cursor = np.empty(n_last, dtype=np.int64)
        for line in range(block_edges[b], block_edges[b + 1]):
            kmin = n_last
            kmax = -1
            for m in range(line_ptr[line], line_ptr[line + 1]):
                p = line_particles[m]
                _, _, dperp2 = _line_perp(line, p, coords, shape)
                r2 = radii[p] * radii[p]
                lo, hi = _line_interval(coords[p, ndim - 1], r2, dperp2, n_last)
                if hi >= lo:
                    diff[lo] += 1
                    diff[hi + 1] -= 1
                    kmin = min(kmin, lo)
                    kmax = max(kmax, hi)
            pos = hit_ptr[line]
            c = cell_ptr[line]
            cover = 0
            for k in range(kmin, kmax + 1):
                cover += diff[k]
                diff[k] = 0
                if cover > 0:
                    cursor[k] = pos
                    cell_indices[c] = line * n_last + k
                    strides[c, 0] = pos
                    strides[c, 1] = pos + cover
                    c += 1
                    pos += cover
            if kmax >= 0:
                diff[kmax + 1] = 0
            for m in range(line_ptr[line], line_ptr[line + 1]):
                p = line_particles[m]
                dx, dy, dperp2 = _line_perp(line, p, coords, shape)
                r2 = radii[p] * radii[p]
                cp = coords[p, ndim - 1]
                lo, hi = _line_interval(cp, r2, dperp2, n_last)
                for k in range(lo, hi + 1):
                    dest = cursor[k]
                    cursor[k] += 1
                    intersections[dest] = p
                    if fill_distances:
                        distances[dest, 0] = dx
                        if ndim == 2:
                            distances[dest, 1] = cp - k
                        else:
                            distances[dest, 1] = dy
                            distances[dest, 2] = cp - k


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
    for d, v in enumerate((x, y, z)):
        lo = corner[d]
        hi = corner[d] + size
        if v < lo:
            d2 += (lo - v) * (lo - v)
        elif v > hi:
            d2 += (v - hi) * (v - hi)
    return d2


def _root_range(
    v: float, r: float, origin: float, root_size: float, n_roots: int
) -> tuple[int, int]:
    """
    Find the range of root nodes along an axis that a search radius may touch.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    v : float
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
    lo = max(math.floor((v - r - origin) / root_size), 0.0)
    hi = min(math.floor((v + r - origin) / root_size), n_roots - 1.0)
    return int(lo), int(hi)


def _tree_walk(
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
    stack: np.ndarray,
    pair_cells: np.ndarray,
    start: int,
    fill: bool,
) -> int:
    """
    Find the cells of a tree with centres within the search radius of one coordinate.

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
        Lower corner of the root lattice, see :func:`_build_cell_tree`.

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

    stack : ~numpy.ndarray
        Work space for the tree traversal, at least ``8 * (depth + 1)`` long.

    pair_cells : ~numpy.ndarray
        Output array for the cell of each hit (ignored unless ``fill``).

    start : int
        Location in ``pair_cells`` of the first hit of this coordinate.

    fill : bool
        If ``True``, fill ``pair_cells``.

    Returns
    -------
    int
        Number of cells hit.
    """
    x = coords[p, 0]
    y = coords[p, 1]
    z = coords[p, 2]
    r = radii[p]
    r2 = r * r
    i0, i1 = _root_range(x, r, origin[0], root_size, root_shape[0])
    j0, j1 = _root_range(y, r, origin[1], root_size, root_shape[1])
    k0, k1 = _root_range(z, r, origin[2], root_size, root_shape[2])
    hits = 0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                root = root_nodes[(i * root_shape[1] + j) * root_shape[2] + k]
                if root < 0:
                    continue
                stack[0] = root
                top = 1
                while top > 0:
                    top -= 1
                    node = stack[top]
                    cell = node_cell[node]
                    if cell >= 0:
                        dx = x - cell_centres[cell, 0]
                        dy = y - cell_centres[cell, 1]
                        dz = z - cell_centres[cell, 2]
                        if dx * dx + dy * dy + dz * dz <= r2:
                            if fill:
                                pair_cells[start + hits] = cell
                            hits += 1
                        continue
                    corner = node_corner[node]
                    if _box_distance2(x, y, z, corner, node_size[node]) <= r2:
                        for c in range(8):
                            stack[top] = first_child[node] + c
                            top += 1
    return hits


def _count_tree_hits(
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
    block_edges: np.ndarray,
    counts: np.ndarray,
) -> None:
    """
    Count the cells of a tree with centres within the search radius of each coordinate.

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
        Lower corner of the root lattice, see :func:`_build_cell_tree`.

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

    block_edges : ~numpy.ndarray
        Coordinates are processed in parallel in blocks
        ``block_edges[b]:block_edges[b + 1]``.

    counts : ~numpy.ndarray
        Output array for the number of cells hit by each coordinate.
    """
    unused = np.empty(0, dtype=np.int64)
    for b in prange(block_edges.size - 1):
        stack = np.empty(8 * (depth + 1), dtype=np.int64)
        for p in range(block_edges[b], block_edges[b + 1]):
            counts[p] = _tree_walk(
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
                stack,
                unused,
                0,
                False,
            )


def _fill_tree_hits(
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
    block_edges: np.ndarray,
    hit_ptr: np.ndarray,
    pair_cells: np.ndarray,
) -> None:
    """
    Record the cells of a tree within the search radius of each coordinate.

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
        Lower corner of the root lattice, see :func:`_build_cell_tree`.

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

    block_edges : ~numpy.ndarray
        In parallel execution, coordinates are processed in load-balanced blocks
        ``block_edges[b]:block_edges[b + 1]``.

    hit_ptr : ~numpy.ndarray
        Location in ``pair_cells`` where the hits of each coordinate start.

    pair_cells : ~numpy.ndarray
        Output array for the cell of each hit, grouped by coordinate.
    """
    for b in prange(block_edges.size - 1):
        stack = np.empty(8 * (depth + 1), dtype=np.int64)
        for p in range(block_edges[b], block_edges[b + 1]):
            _tree_walk(
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
                stack,
                pair_cells,
                hit_ptr[p],
                True,
            )


def _group_by_cell(
    coords: np.ndarray,
    cell_centres: np.ndarray,
    hit_ptr: np.ndarray,
    pair_cells: np.ndarray,
    cursor: np.ndarray,
    intersections: np.ndarray,
    distances: np.ndarray,
    fill_distances: bool,
) -> None:
    """
    Reorder hits grouped by search coordinate into hits grouped by cell.

    Within each cell, hits remain in ascending order of the search coordinates.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``.

    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    hit_ptr : ~numpy.ndarray
        Location in ``pair_cells`` where the hits of each coordinate start.

    pair_cells : ~numpy.ndarray
        The cell of each hit, grouped by coordinate.

    cursor : ~numpy.ndarray
        Location in ``intersections`` where the hits of each cell start (modified).

    intersections : ~numpy.ndarray
        Output array for the index of the search coordinate of each hit.

    distances : ~numpy.ndarray
        Output array for the offset from the cell centre to the search coordinate of
        each hit, shape ``(M, 3)`` (ignored unless ``fill_distances``).

    fill_distances : bool
        If ``True``, fill ``distances``.
    """
    for p in range(hit_ptr.size - 1):
        for m in range(hit_ptr[p], hit_ptr[p + 1]):
            cell = pair_cells[m]
            dest = cursor[cell]
            cursor[cell] += 1
            intersections[dest] = p
            if fill_distances:
                for d in range(3):
                    distances[dest, d] = coords[p, d] - cell_centres[cell, d]


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
    _tree_walk = njit(cache=True)(_tree_walk)
    _count_tree_hits = njit(parallel=True, cache=True)(_count_tree_hits)
    _fill_tree_hits = njit(parallel=True, cache=True)(_fill_tree_hits)
    _group_by_cell = njit(cache=True)(_group_by_cell)


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


def _prepare_inputs(
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
    find_grid_intersections, find_cell_tree_intersections, count_cell_tree_intersections
    """
    grid_shape, coords, radii = _prepare_inputs(grid_shape, coords, radii)
    with _threads(ncpu):
        return _count_hits(coords, radii, np.array(grid_shape, dtype=np.int64))


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
    count_grid_intersections, find_cell_tree_intersections, count_cell_tree_intersections

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
    grid_shape, coords, radii = _prepare_inputs(grid_shape, coords, radii)
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
            return FindGridIntersectionsResult(
                intersections=np.empty(0),
                distances=np.empty(0) if return_distances else None,
                cell_indices=np.empty(0),
                strides=np.empty((0, 2)),
            )

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


class _CellTree(NamedTuple):
    """
    Provide the arrays describing an octree of cells, see :func:`_build_cell_tree`.

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


def _lattice_keys(ijk: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    Flatten 3D lattice indices.

    Parameters
    ----------
    ijk : ~numpy.ndarray
        Lattice indices, shape ``(N, 3)``.

    shape : ~numpy.ndarray
        Shape of the lattice.

    Returns
    -------
    ~numpy.ndarray
        The flattened indices.
    """
    return np.ravel_multi_index(tuple(ijk.T), tuple(shape))


def _sorted_lookup(
    sorted_keys: np.ndarray, keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Look up keys in a sorted array of keys.

    Parameters
    ----------
    sorted_keys : ~numpy.ndarray
        Keys to search, in ascending order.

    keys : ~numpy.ndarray
        Keys to look up.

    Returns
    -------
    tuple
        The location in ``sorted_keys`` of each key (meaningful only where found), and
        whether each key was found.
    """
    if sorted_keys.size == 0:
        return np.zeros(keys.size, dtype=np.intp), np.zeros(keys.size, dtype=bool)
    found = np.minimum(np.searchsorted(sorted_keys, keys), sorted_keys.size - 1)
    return found, sorted_keys[found] == keys


def _build_cell_tree(cell_centres: np.ndarray, cell_sizes: np.ndarray) -> _CellTree:
    """
    Build the octree of a grid of cubic cells obtained by recursive bisection.

    The largest cells (size ``S``) and the cells of each size ``S / 2**k`` are aligned
    with lattices of their own size, starting from the lowest cell corner, and the cells
    tile a region without overlapping (as for
    :class:`~martini.mochi.cell_grid.AdaptiveCellGrid`). Nodes of the tree are the cells
    (leaves) and the cubes that were bisected to obtain them (internal nodes). The tree
    is built top-down, one level at a time, with the 8 children of a node stored
    consecutively.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    cell_sizes : ~numpy.ndarray
        Cell side lengths, shape ``(n_cells,)``.

    Returns
    -------
    _CellTree
        The tree.

    Raises
    ------
    ValueError
        If the cells do not form an octree.
    """
    if cell_sizes.size == 0:
        raise ValueError("Cannot build a tree without cells.")
    root_size = float(cell_sizes.max())
    levels = np.rint(np.log2(root_size / cell_sizes)).astype(np.int64)
    if np.any(root_size / 2.0**levels != cell_sizes):
        raise ValueError(
            "Cell sizes must be the largest cell size divided by powers of two."
        )
    corners = cell_centres - cell_sizes[:, np.newaxis] / 2
    origin = corners.min(axis=0)
    lattice = (corners - origin) / cell_sizes[:, np.newaxis]
    ijk = np.rint(lattice).astype(np.int64)
    if np.any(np.abs(lattice - ijk) > 1e-6):
        raise ValueError(
            "Cell corners must lie on a lattice of the cell size, starting from the "
            "lowest cell corner."
        )
    depth = int(levels.max())
    root_shape = (ijk >> levels[:, np.newaxis]).max(axis=0) + 1
    if np.log2(np.prod(root_shape.astype(float))) + 3 * depth > 62:
        raise ValueError("Too many levels of refinement to index the cells.")
    children = np.array(
        [(dx, dy, dz) for dx in range(2) for dy in range(2) for dz in range(2)]
    )
    first_child, node_cell, node_corner, node_size = [], [], [], []
    n_nodes = 0
    node_ijk = np.empty((0, 3), dtype=np.int64)
    is_internal = np.empty(0, dtype=bool)
    root_nodes = np.empty(0, dtype=np.int64)
    for level in range(depth + 1):
        shape = root_shape << level
        size = root_size / 2.0**level
        leaves = np.flatnonzero(levels == level)
        leaf_keys = _lattice_keys(ijk[leaves], shape)
        leaf_order = np.argsort(leaf_keys)
        leaf_keys = leaf_keys[leaf_order]
        if np.any(np.diff(leaf_keys) == 0):
            raise ValueError("Cells must not be duplicated.")
        deeper = levels > level
        internal_keys = np.unique(
            _lattice_keys(ijk[deeper] >> (levels[deeper] - level)[:, np.newaxis], shape)
        )
        if level == 0:
            root_keys = np.union1d(leaf_keys, internal_keys)
            node_ijk = np.column_stack(np.unravel_index(root_keys, tuple(shape)))
        else:
            node_ijk = (
                2 * node_ijk[is_internal, np.newaxis, :] + children[np.newaxis]
            ).reshape(-1, 3)
        keys = _lattice_keys(node_ijk, shape)
        leaf_found, is_leaf = _sorted_lookup(leaf_keys, keys)
        _, is_internal = _sorted_lookup(internal_keys, keys)
        if np.any(is_leaf & is_internal):
            raise ValueError("Cells must not overlap.")
        if np.any(~is_leaf & ~is_internal):
            raise ValueError(
                "Cells must tile the region covered by the largest cells (each bisected "
                "cell must be replaced by all 8 of its octants)."
            )
        n_level = keys.size
        next_offset = n_nodes + n_level
        level_first_child = np.full(n_level, -1, dtype=np.int64)
        level_first_child[is_internal] = next_offset + 8 * np.arange(
            np.count_nonzero(is_internal)
        )
        level_cell = np.full(n_level, -1, dtype=np.int64)
        level_cell[is_leaf] = leaves[leaf_order[leaf_found[is_leaf]]]
        first_child.append(level_first_child)
        node_cell.append(level_cell)
        node_corner.append(origin + node_ijk * size)
        node_size.append(np.full(n_level, size))
        if level == 0:
            root_nodes = np.full(np.prod(root_shape), -1, dtype=np.int64)
            root_nodes[keys] = np.arange(n_level)
        n_nodes = next_offset
    return _CellTree(
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


def _prepare_tree_inputs(
    cell_centres: np.ndarray,
    cell_sizes: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
) -> tuple[_CellTree, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and normalize inputs to the cell tree search functions.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    cell_sizes : ~numpy.ndarray
        Cell side lengths, shape ``(n_cells,)``.

    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Search radii, one per row of ``coords``.

    Returns
    -------
    tuple
        The tree of cells, and C-contiguous, ``float64`` copies (if needed) of
        ``cell_centres``, ``coords`` and ``radii``.
    """
    cell_centres = np.ascontiguousarray(cell_centres, dtype=np.float64)
    cell_sizes = np.ascontiguousarray(cell_sizes, dtype=np.float64)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    if cell_centres.ndim != 2 or cell_centres.shape[1] != 3:
        raise ValueError("cell_centres must have shape (n_cells, 3).")
    if cell_sizes.shape != (cell_centres.shape[0],):
        raise ValueError("cell_sizes must have one entry per row of cell_centres.")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3).")
    if radii.shape != (coords.shape[0],):
        raise ValueError("radii must have one entry per row of coords.")
    tree = _build_cell_tree(cell_centres, cell_sizes)
    # Radii can be infinite, cap them at a distance that reaches every cell so that the
    # range of root nodes is finite:
    extent = tree.root_shape * tree.root_size
    reach_all = np.sqrt(np.sum((np.abs(coords - tree.origin) + extent) ** 2, axis=1))
    radii = np.minimum(radii, reach_all)
    return tree, cell_centres, coords, radii


def _tree_counts(
    tree: _CellTree,
    cell_centres: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """
    Count the cells within the search radius of each coordinate.

    Must be called within :func:`_threads`.

    Parameters
    ----------
    tree : _CellTree
        The tree of cells.

    cell_centres : ~numpy.ndarray
        Cell centres, shape ``(n_cells, 3)``.

    coords : ~numpy.ndarray
        Search coordinates, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Search radii, shape ``(N,)``.

    Returns
    -------
    ~numpy.ndarray
        Number of cell centres within the search radius of each coordinate.
    """
    counts = np.zeros(coords.shape[0], dtype=np.int64)
    if coords.shape[0] > 0:
        n_blocks = 8 * (numba.get_num_threads() if NUMBA_AVAILABLE else 1)
        _count_tree_hits(
            coords,
            radii,
            cell_centres,
            *tree,
            _balanced_blocks(np.ones(coords.shape[0]), n_blocks),
            counts,
        )
    return counts


def count_cell_tree_intersections(
    cell_centres: np.ndarray,
    cell_sizes: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    ncpu: int = 1,
) -> np.ndarray:
    """
    Count the cell centres of an adaptive (octree) cell grid within the search radii.

    The cells are cubes obtained by recursively bisecting a uniform grid of cubes (e.g.
    :class:`~martini.mochi.cell_grid.AdaptiveCellGrid`), see
    :func:`find_cell_tree_intersections`.

    This is cheaper than recording the hits with :func:`find_cell_tree_intersections`
    (no per-match output is stored), so it can be used to plan memory usage in advance.

    For the simpler case of a one-level tree (a uniform cell grid), see
    :func:`count_grid_intersections`.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Coordinates of the cell centres, shape ``(n_cells, 3)``.

    cell_sizes : ~numpy.ndarray
        Side lengths of the cells (same implicit units as ``cell_centres``).

    coords : ~numpy.ndarray
        Coordinates to search around for cells, shape ``(N, 3)`` (same implicit units as
        ``cell_centres``).

    radii : ~numpy.ndarray
        Search radii around the ``coords``. Must be an array with the same length as the
        number of coordinates, a scalar is not accepted.

    ncpu : int
        Number of threads to use.

    Returns
    -------
    ~numpy.ndarray
        Number of cell centres within the search radius of each coordinate.

    See Also
    --------
    find_cell_tree_intersections, find_grid_intersections, count_grid_intersections
    """
    tree, cell_centres, coords, radii = _prepare_tree_inputs(
        cell_centres, cell_sizes, coords, radii
    )
    with _threads(ncpu):
        return _tree_counts(tree, cell_centres, coords, radii)


def find_cell_tree_intersections(
    cell_centres: np.ndarray,
    cell_sizes: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    ncpu: int = 1,
    dist_dtype: type = np.float64,
    return_distances: bool = True,
) -> FindGridIntersectionsResult:
    """
    Search for coordinates that reach the cell centres of an adaptive (octree) grid.

    The cells are cubes obtained by recursively bisecting a uniform grid of cubes, as
    in :class:`~martini.mochi.cell_grid.AdaptiveCellGrid`: the largest cells have size
    ``S``, every cell has a size ``S / 2**k`` and its lower corner on a lattice of its
    own size starting from the lowest cell corner, and the cells tile a region without
    overlapping. A cell is "hit" if its centre is within the search radius (inclusive)
    of a coordinate.

    For the simpler case of a one-level tree (a uniform cell grid), see
    :func:`find_grid_intersections`.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Coordinates of the cell centres, shape ``(n_cells, 3)``.

    cell_sizes : ~numpy.ndarray
        Side lengths of the cells (same implicit units as ``cell_centres``).

    coords : ~numpy.ndarray
        Coordinates to search around for cells, shape ``(N, 3)`` (same implicit units as
        ``cell_centres``).

    radii : ~numpy.ndarray
        Search radii around the ``coords``. Must be an array with the same length as the
        number of coordinates, a scalar is not accepted.

    ncpu : int
        Number of threads to use for the search.

    dist_dtype : type, optional
        Data type for returned ``distances`` array, can be used to manage memory usage.

    return_distances : bool, optional
        If ``False``, the offset vectors between cell centres and search coordinates are
        not computed (``distances`` is ``None``), saving memory.

    Returns
    -------
    intersections : ~numpy.ndarray
        Each entry is the index of a search coordinate (index into ``coords``),
        representing a search coordinate that "hit" a cell. Results are grouped by the
        cell indices, and in ascending search coordinate index order within each cell.

    distances : ~numpy.ndarray or None
        The vector offset from the cell centre to the search coordinate for each match.
        ``None`` if ``return_distances`` is ``False``.

    cell_indices : ~numpy.ndarray
        The unique cell indices (index into ``cell_centres``), in ascending order. Each
        entry in this array corresponds to a range in ``strides``.

    strides : ~numpy.ndarray
        Each row can be used as a range to select rows from ``intersections`` that share
        a common cell index.

    Raises
    ------
    ValueError
        If the cells do not form an octree as described above.

    See Also
    --------
    count_cell_tree_intersections, find_grid_intersections, count_grid_intersections

    Notes
    -----
    The tree is traversed from the largest cells down for each coordinate, skipping
    nodes (bisected cells) whose cube is beyond the search radius, to count the hits.
    Since the hits of each coordinate are counted in advance, coordinates can be
    processed in parallel to record the hits without risking write collisions. The hits
    are then regrouped by cell.
    """
    tree, cell_centres, coords, radii = _prepare_tree_inputs(
        cell_centres, cell_sizes, coords, radii
    )
    n_cells = cell_centres.shape[0]
    with _threads(ncpu):
        counts = _tree_counts(tree, cell_centres, coords, radii)
        hit_ptr = np.zeros(coords.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=hit_ptr[1:])
        total_intersections = int(hit_ptr[-1])
        if total_intersections == 0:
            return FindGridIntersectionsResult(
                intersections=np.empty(0),
                distances=np.empty(0) if return_distances else None,
                cell_indices=np.empty(0),
                strides=np.empty((0, 2)),
            )
        n_blocks = 8 * (numba.get_num_threads() if NUMBA_AVAILABLE else 1)
        pair_cells: np.ndarray = np.empty(
            total_intersections,
            dtype=np.int32 if n_cells <= np.iinfo(np.int32).max else np.int64,
        )
        _fill_tree_hits(
            coords,
            radii,
            cell_centres,
            *tree,
            _balanced_blocks(counts + 1, n_blocks),
            hit_ptr,
            pair_cells,
        )
    cell_ptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(pair_cells, minlength=n_cells), out=cell_ptr[1:])
    cell_indices = np.flatnonzero(np.diff(cell_ptr)).astype(np.int64)
    strides = np.column_stack((cell_ptr[cell_indices], cell_ptr[cell_indices + 1]))
    intersections = np.empty(total_intersections, dtype=np.int32)
    distances: np.ndarray = np.empty(
        (total_intersections if return_distances else 1, 3), dtype=dist_dtype
    )
    _group_by_cell(
        coords,
        cell_centres,
        hit_ptr,
        pair_cells,
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
