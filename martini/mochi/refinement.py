"""Provide tools for recursively refining a cell grid."""

import numpy as np
from typing import TYPE_CHECKING
from martini._util import NUMBA_AVAILABLE, resolve_ncpu
from martini._grid_search import (
    Octree,
    _OctreeArrays,
    _threads,
    find_grid_intersections,
)
from martini.mochi._dtypes import CELL_DTYPE as _CELL_DTYPE

if NUMBA_AVAILABLE:
    from numba import njit
if TYPE_CHECKING or not NUMBA_AVAILABLE:
    prange = range  # pragma: no cover
else:
    from numba import prange

_RF = np.sqrt(3) / 2  # ratio between radius and cell length

# Refinement criteria (see _keep_particle):
_HALF_PARTICLE_SCALE = 0
_OCCUPANCY = 1

# Offsets of the 8 children of a node, in the order of Octree._build_octree:
_CHILD_OFFSETS = np.array(
    [(dx, dy, dz) for dx in range(2) for dy in range(2) for dz in range(2)], dtype=float
)


def _keep_particle(
    criterion: int,
    x: float,
    y: float,
    z: float,
    radius: float,
    corner_x: float,
    corner_y: float,
    corner_z: float,
    size: float,
) -> bool:
    """
    Decide whether a particle counts towards bisecting a cell.

    For ``_HALF_PARTICLE_SCALE``, particles count if they are small compared to the cell
    (half of their radius is less than the cell size) and they intersect the cell. The
    intersection is approximated (conservatively) by the sphere enclosing the cell: the
    distance from the particle to the cell centre is less than the particle radius plus
    half of the cell diagonal.

    For ``_OCCUPANCY``, particles count if their (L1, or "taxicab") distance from the cell
    centre is less than twice the cell size. Particle radii are ignored.

    Neither criterion depends on the parent of the cell, and any particle that counts for
    a cell also counts for its parent, so the candidates of a cell can be restricted to
    the particles that count for its parent.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    criterion : int
        ``_HALF_PARTICLE_SCALE`` or ``_OCCUPANCY``.

    x, y, z : float
        Particle position, in pixels.

    radius : float
        Particle radius (smoothing range), in pixels.

    corner_x, corner_y, corner_z : float
        Lower corner of the cell, in pixels.

    size : float
        Side length of the cell, in pixels.

    Returns
    -------
    bool
        ``True`` if the particle counts towards bisecting the cell.
    """
    half = size / 2
    dx = (x - corner_x) - half
    dy = (y - corner_y) - half
    dz = (z - corner_z) - half
    if criterion == _HALF_PARTICLE_SCALE:
        return radius * 0.5 < size and np.sqrt(dx * dx + dy * dy + dz * dz) < (
            radius + size * _RF
        )
    return abs(dx) + abs(dy) + abs(dz) < size * 2


def _count_kept(
    criterion: int,
    positions: np.ndarray,
    radii: np.ndarray,
    candidates: np.ndarray,
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
    corners: np.ndarray,
    size: float,
    counts: np.ndarray,
) -> None:
    """
    Count the particles that count towards bisecting each cell of a level of the tree.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    criterion : int
        ``_HALF_PARTICLE_SCALE`` or ``_OCCUPANCY``, see :func:`_keep_particle`.

    positions : ~numpy.ndarray
        Particle positions, in pixels, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Particle radii (smoothing ranges), in pixels.

    candidates : ~numpy.ndarray
        Indices of candidate particles, see ``candidate_start``.

    candidate_start : ~numpy.ndarray
        Start of the range of ``candidates`` of each cell. Ranges of different cells may
        overlap (the 8 children of a node share the candidates of the node).

    candidate_end : ~numpy.ndarray
        End of the range of ``candidates`` of each cell.

    corners : ~numpy.ndarray
        Lower corners of the cells, in pixels, shape ``(n_cells, 3)``.

    size : float
        Side length of the cells, in pixels.

    counts : ~numpy.ndarray
        Output, the number of particles that count for each cell.
    """
    for n in prange(corners.shape[0]):
        count = 0
        for j in range(candidate_start[n], candidate_end[n]):
            p = candidates[j]
            if _keep_particle(
                criterion,
                positions[p, 0],
                positions[p, 1],
                positions[p, 2],
                radii[p],
                corners[n, 0],
                corners[n, 1],
                corners[n, 2],
                size,
            ):
                count += 1
        counts[n] = count


def _fill_kept(
    criterion: int,
    positions: np.ndarray,
    radii: np.ndarray,
    candidates: np.ndarray,
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
    corners: np.ndarray,
    size: float,
    bisect: np.ndarray,
    offsets: np.ndarray,
    kept: np.ndarray,
) -> None:
    """
    List the particles that count for each bisected cell, as candidates of its children.

    Can be compiled with numba for faster execution.

    Parameters
    ----------
    criterion : int
        ``_HALF_PARTICLE_SCALE`` or ``_OCCUPANCY``, see :func:`_keep_particle`.

    positions : ~numpy.ndarray
        Particle positions, in pixels, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Particle radii (smoothing ranges), in pixels.

    candidates : ~numpy.ndarray
        Indices of candidate particles, see :func:`_count_kept`.

    candidate_start : ~numpy.ndarray
        Start of the range of ``candidates`` of each cell.

    candidate_end : ~numpy.ndarray
        End of the range of ``candidates`` of each cell.

    corners : ~numpy.ndarray
        Lower corners of the cells, in pixels, shape ``(n_cells, 3)``.

    size : float
        Side length of the cells, in pixels.

    bisect : ~numpy.ndarray
        Whether each cell is bisected (only bisected cells are listed).

    offsets : ~numpy.ndarray
        Start of the output range of each cell in ``kept``.

    kept : ~numpy.ndarray
        Output, the indices of the particles that count for each bisected cell.
    """
    for n in prange(corners.shape[0]):
        if not bisect[n]:
            continue
        k = offsets[n]
        for j in range(candidate_start[n], candidate_end[n]):
            p = candidates[j]
            if _keep_particle(
                criterion,
                positions[p, 0],
                positions[p, 1],
                positions[p, 2],
                radii[p],
                corners[n, 0],
                corners[n, 1],
                corners[n, 2],
                size,
            ):
                kept[k] = p
                k += 1


if NUMBA_AVAILABLE:
    _keep_particle = njit(cache=True, inline="always")(_keep_particle)
    _count_kept = njit(parallel=True, cache=True)(_count_kept)
    _fill_kept = njit(parallel=True, cache=True)(_fill_kept)


def _root_lattice(
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Place the initial cells on a lattice, as the root nodes of the tree.

    Parameters
    ----------
    cells : ~numpy.ndarray
        Lower corners (``cells["x"]``, etc.) and sizes (``cells["size"]``) of the initial
        cells.

    Returns
    -------
    corners : ~numpy.ndarray
        Lower corners of the initial cells, shape ``(n_cells, 3)``.

    origin : ~numpy.ndarray
        Lower corner of the lattice.

    root_size : float
        Side length of the initial cells.

    root_shape : ~numpy.ndarray
        Number of lattice positions along each axis.

    root_nodes : ~numpy.ndarray
        Initial cell at each (flattened, C-order) lattice position, or ``-1``.

    Raises
    ------
    ValueError
        If the initial cells do not all have the same size, are not on a lattice or are
        duplicated.
    """
    sizes = np.asarray(cells["size"], dtype=np.float64)
    if sizes.size == 0:
        raise ValueError("Cannot refine a grid without cells.")
    root_size = float(sizes[0])
    if np.any(sizes != root_size):
        raise ValueError("Initial cells must all have the same size.")
    corners = np.column_stack([cells[axis] for axis in "xyz"]).astype(np.float64)
    origin = corners.min(axis=0)
    lattice_coords = (corners - origin) / root_size
    ijk = np.rint(lattice_coords).astype(np.int64)
    if np.any(np.abs(lattice_coords - ijk) > 1e-6):
        raise ValueError("Initial cells must lie on a lattice of the cell size.")
    root_shape = ijk.max(axis=0) + 1
    flat = np.ravel_multi_index(tuple(ijk.T), tuple(root_shape))
    root_nodes = np.full(np.prod(root_shape), -1, dtype=np.int64)
    root_nodes[flat] = np.arange(sizes.size)
    if np.unique(flat).size != flat.size:
        raise ValueError("Initial cells must not be duplicated.")
    return corners, origin, root_size, root_shape, root_nodes


def _refinement_mem_estimate(
    n_nodes: int, n_candidates: int, n_particles: int
) -> float:
    """
    Estimate the peak memory needed to refine a cell grid to a number of tree nodes.

    Estimated as:

    - Per node of the tree, the arrays describing the tree and the cells (corners,
      sizes, centres), and the arrays that
      :class:`~martini.mochi.cell_grid.AdaptiveCellGrid` derives from them (volumes,
      distances, distortion corrections), with their temporaries (measured ~250B, allow
      300B).
    - Per candidate particle of the cells being refined, its index (8B).
    - Per particle, copies of its position and radius and the search for the cells near
      it (allow 48B).

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the tree.

    n_candidates : int
        Number of candidate particles held for the cells being refined.

    n_particles : int
        Number of particles.

    Returns
    -------
    float
        The estimated memory, in GB.
    """
    return (300 * n_nodes + 8 * n_candidates + 48 * n_particles) / 1024**3


def _refine(
    criterion: int,
    max_count: int,
    cells: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    stop_iter: int,
    ncpu: int,
    mem_allowance_GB: float,
) -> Octree:
    """
    Start from a uniform grid, bisect cells until none meet a criterion.

    A cell is bisected if more than ``max_count`` particles count towards bisecting it
    (see :func:`_keep_particle`), unless it is ``stop_iter`` bisections below the
    initial cells. The tree is built level by level, from the initial cells down, in the
    layout of :meth:`~martini._grid_search.Octree._build_octree`: the 8 children of each
    node are stored consecutively, after all nodes of the level of their parent. The
    candidate particles of each cell are those that count for its parent, found for the
    initial cells with :func:`~martini._grid_search.find_grid_intersections`.

    Parameters
    ----------
    criterion : int
        ``_HALF_PARTICLE_SCALE`` or ``_OCCUPANCY``, see :func:`_keep_particle`.

    max_count : int
        Cells where more particles than this count are bisected.

    cells : ~numpy.ndarray
        Lower corners (``cells["x"]``, etc.) and sizes (``cells["size"]``) of the initial
        cells, which must form a (possibly incomplete) uniform grid.

    positions : ~numpy.ndarray
        Particle positions, in pixels, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Particle radii (smoothing ranges), in pixels.

    stop_iter : int
        Maximum recursive splitting depth.

    ncpu : int
        Number of threads to use.

    mem_allowance_GB : float
        Memory available to refine the grid, in GB. Before building each level of the
        tree, its memory is estimated (see :func:`_refinement_mem_estimate`), and the
        refinement stops with an error if it would exceed the allowance.

    Returns
    -------
    ~martini._grid_search.Octree
        The tree of the refined cells.

    Raises
    ------
    RuntimeError
        If refining further would exceed the memory allowance.
    """
    corners, origin, root_size, root_shape, root_nodes = _root_lattice(cells)
    positions = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    # candidates of the initial cells, from a search for cell centres within a radius
    # (slightly enlarged against rounding) enclosing the region where particles count:
    if criterion == _HALF_PARTICLE_SCALE:
        eligible = np.flatnonzero(radii * 0.5 < root_size)
        search_radii = radii[eligible] + root_size * _RF
    else:
        eligible = np.arange(radii.size)
        search_radii = np.full(eligible.size, 2 * root_size)
    search = find_grid_intersections(
        tuple(root_shape),
        (positions[eligible] - origin) / root_size - 0.5,
        search_radii / root_size * (1 + 1e-9) + 1e-9,
        ncpu=ncpu,
        return_distances=False,
    )
    candidates = eligible[search.intersections]
    search_nodes = root_nodes[search.cell_indices]
    in_grid = search_nodes >= 0  # the lattice may have holes
    candidate_start = np.zeros(root_nodes.max() + 1, dtype=np.int64)
    candidate_end = np.zeros_like(candidate_start)
    candidate_start[search_nodes[in_grid]] = search.strides[in_grid, 0]
    candidate_end[search_nodes[in_grid]] = search.strides[in_grid, 1]

    first_child, node_cell, node_corner, node_size = [], [], [], []
    n_nodes = 0
    n_cells = 0
    size = root_size
    depth = 0
    with _threads(ncpu):
        for level in range(stop_iter + 1):
            depth = level
            n_level = corners.shape[0]
            counts = np.empty(n_level, dtype=np.int64)
            _count_kept(
                criterion,
                positions,
                radii,
                candidates,
                candidate_start,
                candidate_end,
                corners,
                size,
                counts,
            )
            bisect = (counts > max_count) & (level < stop_iter)
            n_bisect = np.count_nonzero(bisect)
            next_offset = n_nodes + n_level
            level_first_child = np.full(n_level, -1, dtype=np.int64)
            level_first_child[bisect] = next_offset + 8 * np.arange(n_bisect)
            level_cell = np.full(n_level, -1, dtype=np.int64)
            level_cell[~bisect] = n_cells + np.arange(n_level - n_bisect)
            first_child.append(level_first_child)
            node_cell.append(level_cell)
            node_corner.append(corners)
            node_size.append(np.full(n_level, size))
            n_nodes = next_offset
            n_cells += n_level - n_bisect
            if n_bisect == 0:
                break
            offsets = np.zeros(n_level + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(np.where(bisect, counts, 0))
            mem_estimate = _refinement_mem_estimate(
                n_nodes + 8 * n_bisect, candidates.size + offsets[-1], radii.size
            )
            if mem_estimate > mem_allowance_GB:
                raise RuntimeError(
                    "Memory allowance for refining the cell grid "
                    f"({mem_allowance_GB:.2f} GB) is insufficient to refine to depth "
                    f"{level + 1} (estimated memory required: {mem_estimate:.2f} GB, "
                    f"with {n_nodes + 8 * n_bisect} tree nodes). Increase `mem_lim_GB` "
                    "(the allowance is what remains of it after the estimated baseline "
                    "memory and the cell grid), or limit the refinement (e.g. its "
                    "`stop_iter`)."
                )
            kept = np.empty(offsets[-1], dtype=candidates.dtype)
            _fill_kept(
                criterion,
                positions,
                radii,
                candidates,
                candidate_start,
                candidate_end,
                corners,
                size,
                bisect,
                offsets,
                kept,
            )
            parents = np.flatnonzero(bisect)
            size = size / 2
            corners = np.ascontiguousarray(
                (corners[parents, np.newaxis, :] + _CHILD_OFFSETS * size).reshape(-1, 3)
            )
            candidates = kept
            candidate_start = np.repeat(offsets[parents], 8)
            candidate_end = np.repeat(offsets[parents] + counts[parents], 8)

    node_cell_array = np.concatenate(node_cell)
    node_corner_array = np.ascontiguousarray(np.concatenate(node_corner))
    node_size_array = np.concatenate(node_size)
    leaves = node_cell_array >= 0  # cells are numbered in node order
    cell_sizes = node_size_array[leaves]
    return Octree._from_arrays(
        _OctreeArrays(
            origin=origin,
            root_size=root_size,
            root_shape=root_shape,
            root_nodes=root_nodes,
            first_child=np.concatenate(first_child),
            node_cell=node_cell_array,
            node_corner=node_corner_array,
            node_size=node_size_array,
            depth=depth,
        ),
        node_corner_array[leaves] + cell_sizes[:, np.newaxis] / 2,
        cell_sizes,
    )


def _octree_cells(octree: Octree) -> np.ndarray:
    """
    Get the cells of a tree as an array of lower corners and sizes.

    Parameters
    ----------
    octree : ~martini._grid_search.Octree
        The tree of cells.

    Returns
    -------
    ~numpy.ndarray
        Contains entries with the 3D position of the lower corner (accessible as
        ``cells["x"]``, etc.) and sizes (accessible as ``cells["size"]``) of the cells, in
        the order of the cells of the tree.
    """
    cells = np.empty(octree.cell_sizes.size, dtype=_CELL_DTYPE)
    corners = octree.cell_centres - octree.cell_sizes[:, np.newaxis] / 2
    for i, axis in enumerate("xyz"):
        cells[axis] = corners[:, i]
    cells["size"] = octree.cell_sizes
    return cells


def refine_grid_to_single_occupancy(
    cells: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray | None = None,
    stop_iter: int = 8,
    ncpu: int = 1,
    mem_allowance_GB: float = np.inf,
) -> Octree:
    """
    From a coarse grid, refine until no cells contain more than two nearby particles.

    A cell is bisected if more than two particles are closer to its centre than twice
    its size, measured as the L1 ("taxicab") distance. Particle radii are ignored.
    Compiled with :mod:`numba` if it is available.

    Parameters
    ----------
    cells : ~numpy.ndarray
        Contains entries with the 3D position of the lower corner (accessible as
        ``cells["x"]``, etc.) and sizes (accessible as ``cells["size"]``) of the initial
        cells, which must form a uniform grid.

    positions : ~numpy.ndarray
        Array of particle positions in pixels, shape ``(N, 3)``.

    radii : ~numpy.ndarray, optional
        Unused.

    stop_iter : int
        Maximum recursive splitting depth.

    ncpu : int, optional
        Number of threads to use (if :mod:`numba` is available). A value of ``-1`` means
        as many threads as available cores.

    mem_allowance_GB : float, optional
        Memory available to refine the grid, in GB. This is a "soft" limit.

    Returns
    -------
    ~martini._grid_search.Octree
        The tree of the refined cell grid.

    Raises
    ------
    RuntimeError
        If refining further would exceed the memory allowance.
    """
    positions = np.asarray(positions).reshape(-1, 3)
    return _refine(
        _OCCUPANCY,
        2,
        cells,
        positions,
        np.zeros(positions.shape[0]),
        stop_iter,
        resolve_ncpu(ncpu),
        mem_allowance_GB,
    )


def refine_grid_to_half_particle_scale(
    cells: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    stop_iter: int = 8,
    ncpu: int = 1,
    mem_allowance_GB: float = np.inf,
) -> Octree:
    """
    Refine until cells are smaller than the half radius of intersecting particles.

    A cell is bisected if any particle intersects it while half of the particle's
    radius is smaller than the cell. The intersection is approximated (conservatively)
    by the sphere enclosing the cell. Compiled with :mod:`numba` if it is available.

    Parameters
    ----------
    cells : ~numpy.ndarray
        Contains entries with the 3D position of the lower corner (accessible as
        ``cells["x"]``, etc.) and sizes (accessible as ``cells["size"]``) of the initial
        cells, which must form a uniform grid.

    positions : ~numpy.ndarray
        Array of particle positions in pixels, shape ``(N, 3)``.

    radii : ~numpy.ndarray
        Array of particle radii (smoothing ranges) in pixels.

    stop_iter : int
        Maximum recursive splitting depth.

    ncpu : int, optional
        Number of threads to use (if :mod:`numba` is available). A value of ``-1`` means
        as many threads as available cores.

    mem_allowance_GB : float, optional
        Memory available to refine the grid, in GB. This is a "soft" limit.

    Returns
    -------
    ~martini._grid_search.Octree
        The tree of the refined cell grid.

    Raises
    ------
    RuntimeError
        If refining further would exceed the memory allowance.
    """
    return _refine(
        _HALF_PARTICLE_SCALE,
        0,
        cells,
        positions,
        radii,
        stop_iter,
        resolve_ncpu(ncpu),
        mem_allowance_GB,
    )
