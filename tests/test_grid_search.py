"""Tests of the optimized grid search methods."""

import pytest
import numpy as np
from martini import _grid_search
from martini._grid_search import (
    Octree,
    count_grid_intersections,
    find_grid_intersections,
    FindGridIntersectionsResult,
)


def _brute_force(
    grid_shape: tuple[int, ...], coords: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all matches between coordinates and cells by brute force.

    Parameters
    ----------
    grid_shape : tuple
        Number of cells along each axis.

    coords : ~numpy.ndarray
        Search coordinates, one column per dimension.

    radii : ~numpy.ndarray
        Search radii.

    Returns
    -------
    tuple
        Arrays of particle indices, cell indices and offset vectors of matches, sorted
        by particle then cell index.
    """
    cell_centres = np.indices(grid_shape).reshape(len(grid_shape), -1).T
    return _brute_force_centres(cell_centres, coords, radii)


def _brute_force_centres(
    cell_centres: np.ndarray, coords: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all matches between coordinates and arbitrary cell centres by brute force.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Cell centres, one column per dimension.

    coords : ~numpy.ndarray
        Search coordinates, one column per dimension.

    radii : ~numpy.ndarray
        Search radii.

    Returns
    -------
    tuple
        Arrays of particle indices, cell indices and offset vectors of matches, sorted
        by particle then cell index.
    """
    part, cell, dvec = [], [], []
    for p, (coord, radius) in enumerate(zip(coords, radii)):
        offsets = coord - cell_centres
        (hits,) = np.nonzero(np.sum(offsets**2, axis=1) <= radius**2)
        part.append(np.full(hits.size, p))
        cell.append(hits)
        dvec.append(offsets[hits])
    return np.concatenate(part), np.concatenate(cell), np.concatenate(dvec)


def _unpack(
    gs: FindGridIntersectionsResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand grid search result into per-match arrays sorted by particle then cell index.

    Parameters
    ----------
    gs : ~martini._grid_search.FindGridIntersectionsResult
        The grid search result.

    Returns
    -------
    tuple
        Arrays of particle indices, cell indices and offset vectors of matches.
    """
    cell = np.repeat(gs.cell_indices, gs.strides[:, 1] - gs.strides[:, 0])
    order = np.lexsort((cell, gs.intersections))
    return gs.intersections[order], cell[order], gs.distances[order]


def _random_input(
    grid_shape: tuple[int, ...], n: int, rmax: float, margin: float = 0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random search coordinates and radii.

    Parameters
    ----------
    grid_shape : tuple
        Number of cells along each axis.

    n : int
        Number of coordinates.

    rmax : float
        Maximum search radius (minimum is 0.25).

    margin : float
        Coordinates extend this far beyond the grid on each side.

    seed : int
        Random seed.

    Returns
    -------
    tuple
        The coordinates and radii.
    """
    rng = np.random.default_rng(seed)
    ndim = len(grid_shape)
    coords = rng.random((n, ndim)) * (np.array(grid_shape) + 2 * margin) - (
        0.5 + margin
    )
    radii = rng.random(n) * (rmax - 0.25) + 0.25
    return coords, radii


class TestUniformGridSearch:
    """Tests of the optimized grid search methods for uniform grids."""

    @pytest.mark.parametrize(
        "grid_shape, n, rmax, margin",
        (
            ((128, 128), 1000, 5.0, 0),
            ((37, 64), 300, 4.0, 3),  # non-square, some coords off grid
            ((24, 20, 16), 200, 3.0, 2),  # 3D
        ),
    )
    def test_matches_brute_force(
        self, grid_shape: tuple[int, ...], n: int, rmax: float, margin: int
    ) -> None:
        """Check that optimized search gives same result as a brute force search."""
        coords, radii = _random_input(grid_shape, n, rmax, margin=margin)
        bf_part, bf_cell, bf_dvec = _brute_force(grid_shape, coords, radii)
        gs = find_grid_intersections(grid_shape, coords, radii)
        part, cell, dvec = _unpack(gs)
        assert np.array_equal(part, bf_part)
        assert np.array_equal(cell, bf_cell)
        assert np.allclose(dvec, bf_dvec)
        # cell_indices unique and sorted, strides contiguous and covering all matches:
        assert np.all(np.diff(gs.cell_indices) > 0)
        assert gs.strides[0, 0] == 0
        assert np.array_equal(gs.strides[1:, 0], gs.strides[:-1, 1])
        assert gs.strides[-1, 1] == gs.intersections.size
        assert np.all(gs.strides[:, 1] > gs.strides[:, 0])
        # within each cell, matches are in ascending order of search coordinate index:
        for start, end in gs.strides:
            assert np.all(np.diff(gs.intersections[start:end]) > 0)
        # We expect some particles to be small and miss touching any cell centres:
        assert (radii < 0.5 * np.sqrt(len(grid_shape))).any()
        assert np.unique(part).size < n  # strictly less

    def test_exact_boundary(self) -> None:
        """Check that cells exactly at the search radius are included."""
        coords = np.array([[5.0, 5.0], [0.0, 0.0]])
        radii = np.array([2.0, 1.0])
        gs = find_grid_intersections((10, 10), coords, radii)
        part, cell, _ = _unpack(gs)
        assert np.sum(part == 0) == 13  # 3x3 square + 4 cells at distance exactly 2
        assert set(cell[part == 0]) >= {3 * 10 + 5, 7 * 10 + 5, 5 * 10 + 3, 5 * 10 + 7}
        # in the corner, cells at negative indices are not counted:
        assert set(cell[part == 1]) == {0, 1, 10}

    def test_infinite_radius(self) -> None:
        """Check that an infinite search radius reaches every cell."""
        coords = np.array([[0.0, 0.0], [-3.0, 20.0]])
        radii = np.array([np.inf, np.inf])
        assert np.array_equal(count_grid_intersections((5, 4), coords, radii), [20, 20])
        gs = find_grid_intersections((5, 4), coords, radii)
        assert np.array_equal(gs.cell_indices, np.arange(20))

    def test_off_grid(self) -> None:
        """Check that coordinates far from the grid find nothing."""
        coords = np.array([[-10.0, 5.0], [5.0, 30.0]])
        radii = np.array([3.0, 3.0])
        gs = find_grid_intersections((10, 10), coords, radii)
        assert gs.intersections.size == 0

    @pytest.mark.parametrize("grid_shape", ((128, 128), (24, 20, 16)))
    def test_all_coords_matched(self, grid_shape: tuple[int, ...]) -> None:
        """Check that with floored radii we never miss any particles."""
        n = 1000 if len(grid_shape) == 2 else 200
        coords, radii = _random_input(grid_shape, n, 3.0)
        gs = find_grid_intersections(grid_shape, coords, radii)  # not floored
        assert np.unique(gs.intersections).size < n
        gs = find_grid_intersections(
            grid_shape,
            coords,
            np.clip(radii, 0.5 * np.sqrt(len(grid_shape)), np.inf),  # floored
        )
        assert np.unique(gs.intersections).size == n

    def test_empty_input_safe(self) -> None:
        """Check that we get empty results for empty input."""
        coords = np.empty((0, 2))
        radii = np.empty((0,))
        gs = find_grid_intersections((128, 128), coords, radii)
        assert gs.intersections.shape == (0,)
        assert gs.distances.shape == (0, 2)
        assert gs.cell_indices.shape == (0,)
        assert gs.strides.shape == (0, 2)
        # same data types as results with matches:
        assert gs.intersections.dtype == np.int32
        assert gs.cell_indices.dtype == gs.strides.dtype == np.int64

    @pytest.mark.parametrize("grid_shape", ((37, 64), (24, 20, 16)))
    def test_counts(self, grid_shape: tuple[int, ...]) -> None:
        """Check that counts agree with the number of matches found for each particle."""
        coords, radii = _random_input(grid_shape, 100, 3.0, margin=2)
        counts = count_grid_intersections(grid_shape, coords, radii)
        gs = find_grid_intersections(grid_shape, coords, radii)
        assert np.array_equal(
            counts, np.bincount(gs.intersections, minlength=coords.shape[0])
        )

    @pytest.mark.parametrize("grid_shape", ((37, 64), (24, 20, 16)))
    def test_without_distances(self, grid_shape: tuple[int, ...]) -> None:
        """Check that skipping the distances leaves the other results unchanged."""
        coords, radii = _random_input(grid_shape, 100, 3.0, margin=2)
        gs = find_grid_intersections(grid_shape, coords, radii)
        gs_nodist = find_grid_intersections(
            grid_shape, coords, radii, return_distances=False
        )
        assert gs_nodist.distances is None
        for field in ("intersections", "cell_indices", "strides"):
            assert np.array_equal(getattr(gs, field), getattr(gs_nodist, field))
        empty = find_grid_intersections(
            grid_shape, coords[:0], radii[:0], return_distances=False
        )
        assert empty.distances is None

    @pytest.mark.parametrize("ncpu", (1, 3))
    def test_unbalanced_load(self, ncpu: int) -> None:
        """Check a mix of one huge and many tiny search radii (uneven work per line)."""
        grid_shape = (40, 30)
        coords, radii = _random_input(grid_shape, 200, 1.0)
        coords[0] = (10.0, 12.0)
        radii[0] = 25.0
        bf_part, bf_cell, bf_dvec = _brute_force(grid_shape, coords, radii)
        part, cell, dvec = _unpack(
            find_grid_intersections(grid_shape, coords, radii, ncpu=ncpu)
        )
        assert np.array_equal(part, bf_part)
        assert np.array_equal(cell, bf_cell)
        assert np.allclose(dvec, bf_dvec)

    def test_dist_dtype(self) -> None:
        """Check that distances have the requested dtype."""
        grid_size = (37, 64)
        coords, radii = _random_input(grid_size, 50, 3.0)
        gs = find_grid_intersections(grid_size, coords, radii, dist_dtype=np.float32)
        assert gs.distances.dtype == np.float32

    def test_input_validation(self) -> None:
        """Check that inconsistent inputs are rejected."""
        coords, radii = _random_input((10, 10), 5, 2.0)
        with pytest.raises(ValueError, match="2D or 3D"):
            find_grid_intersections((10,), coords[:, :1], radii)
        with pytest.raises(ValueError, match="coords must have shape"):
            find_grid_intersections((10, 10, 10), coords, radii)
        with pytest.raises(ValueError, match="radii must have"):
            find_grid_intersections((10, 10), coords, radii[:-1])
        with pytest.raises(ValueError, match="2D or 3D"):
            count_grid_intersections((10,), coords[:, :1], radii)
        with pytest.raises(ValueError, match="coords must have shape"):
            count_grid_intersections((10, 10, 10), coords, radii)
        with pytest.raises(ValueError, match="radii must have"):
            count_grid_intersections((10, 10), coords, radii[:-1])

    def test_without_numba_threads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Check that the search runs without numba thread control."""
        coords, radii = _random_input((37, 64), 50, 3.0)
        expected = find_grid_intersections((37, 64), coords, radii)
        monkeypatch.setattr(_grid_search, "NUMBA_AVAILABLE", False)
        gs = find_grid_intersections((37, 64), coords, radii)
        for a, b in zip(gs, expected):
            assert np.array_equal(a, b)


def _bisected_grid(
    root_shape: tuple[int, int, int] = (3, 2, 2),
    root_size: float = 2.0,
    depth: int = 3,
    p_bisect: float = 0.4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an adaptive grid by randomly bisecting a uniform grid of cubic cells.

    The grid starts at ``-0.5`` along each axis, as for
    :class:`~martini.mochi.cell_grid.AdaptiveCellGrid`, and the cells are shuffled.

    Parameters
    ----------
    root_shape : tuple
        Number of initial cells along each axis.

    root_size : float
        Side length of the initial cells.

    depth : int
        Maximum number of bisections.

    p_bisect : float
        Probability of bisecting each cell.

    seed : int
        Random seed.

    Returns
    -------
    tuple
        The cell centres and sizes.
    """
    rng = np.random.default_rng(seed)
    todo = [
        (np.array(ijk, dtype=float) * root_size - 0.5, root_size, 0)
        for ijk in np.ndindex(root_shape)
    ]
    corners, sizes = [], []
    while todo:
        corner, size, level = todo.pop()
        if level < depth and rng.random() < p_bisect:
            for offset in np.ndindex((2, 2, 2)):
                todo.append((corner + np.array(offset) * size / 2, size / 2, level + 1))
        else:
            corners.append(corner)
            sizes.append(size)
    order = rng.permutation(len(sizes))
    sizes_array = np.array(sizes)[order]
    return np.array(corners)[order] + sizes_array[:, np.newaxis] / 2, sizes_array


def _small_tree_grid() -> tuple[np.ndarray, np.ndarray]:
    """
    Make a small adaptive grid by hand.

    2x2x2 cells of size 2, one bisected octant, and one octant further bisected, starting
    at ``-0.5``.

    Returns
    -------
    tuple
        The cell centres and sizes.
    """
    corners, sizes = [], []
    for corner in np.ndindex((2, 2, 2)):
        if corner != (0, 0, 0):
            corners.append(np.array(corner) * 2.0)
            sizes.append(2.0)
    for corner in np.ndindex((2, 2, 2)):
        if corner != (1, 1, 1):
            corners.append(np.array(corner, dtype=float))
            sizes.append(1.0)
    for corner in np.ndindex((2, 2, 2)):
        corners.append(1 + np.array(corner) * 0.5)
        sizes.append(0.5)
    sizes_array = np.array(sizes)
    return np.array(corners) + sizes_array[:, np.newaxis] / 2 - 0.5, sizes_array


def _tree_input(
    cell_centres: np.ndarray, n: int, rmax: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random search coordinates and radii around an adaptive grid.

    Coordinates extend 2 beyond the grid on each side, radii range from tiny to
    ``rmax``, and one radius is infinite.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Cell centres of the grid.

    n : int
        Number of coordinates.

    rmax : float
        Maximum (finite) search radius.

    seed : int
        Random seed.

    Returns
    -------
    tuple
        The coordinates and radii.
    """
    rng = np.random.default_rng(seed)
    lo = cell_centres.min(axis=0) - 2
    hi = cell_centres.max(axis=0) + 2
    coords = rng.uniform(lo, hi, (n, 3))
    radii = rng.uniform(0, 1, n) ** 2 * rmax
    radii[0] = np.inf
    return coords, radii


class TestOctreeSearch:
    """Tests of the grid search methods for adaptive (octree) cell grids."""

    @pytest.mark.parametrize(
        "grid",
        (
            _small_tree_grid(),
            # without the last size-2 cell, so the region is not a box:
            tuple(a[:6].tolist() + a[7:].tolist() for a in _small_tree_grid()),
            _bisected_grid(),
            _bisected_grid(root_shape=(1, 2, 3), root_size=1.0, depth=5, seed=1),
        ),
    )
    def test_matches_brute_force(self, grid: tuple[np.ndarray, np.ndarray]) -> None:
        """Check that the tree search gives the same result as a brute force search."""
        cell_centres, cell_sizes = map(np.array, grid)
        coords, radii = _tree_input(cell_centres, 300, 3.0)
        bf_part, bf_cell, bf_dvec = _brute_force_centres(cell_centres, coords, radii)
        gs = Octree(cell_centres, cell_sizes).find(coords, radii)
        part, cell, dvec = _unpack(gs)
        assert np.array_equal(part, bf_part)
        assert np.array_equal(cell, bf_cell)
        assert np.allclose(dvec, bf_dvec)
        # cell_indices unique and sorted, strides contiguous and covering all matches:
        assert np.all(np.diff(gs.cell_indices) > 0)
        assert gs.strides[0, 0] == 0
        assert np.array_equal(gs.strides[1:, 0], gs.strides[:-1, 1])
        assert gs.strides[-1, 1] == gs.intersections.size
        # within each cell, matches are in ascending order of search coordinate index:
        for start, end in gs.strides:
            assert np.all(np.diff(gs.intersections[start:end]) > 0)
        # the infinite radius reaches every cell, some coordinates reach none:
        assert np.sum(part == 0) == cell_sizes.size
        assert np.unique(part).size < coords.shape[0]

    def test_prebuilt_tree_batches(self) -> None:
        """Check that searching one tree for batches of coordinates matches one search."""
        cell_centres, cell_sizes = map(np.array, _bisected_grid())
        coords, radii = _tree_input(cell_centres, 300, 3.0)
        octree = Octree(cell_centres, cell_sizes)
        full_part, full_cell, full_dvec = _unpack(octree.find(coords, radii))
        full_counts = octree.count(coords, radii)
        batches = []
        for start, stop in ((0, 100), (100, 101), (101, 300)):
            batch = slice(start, stop)
            gs = octree.find(coords[batch], radii[batch], ncpu=2)
            part, cell, dvec = _unpack(gs)
            batches.append((part + start, cell, dvec))
            assert np.array_equal(
                octree.count(coords[batch], radii[batch]), full_counts[batch]
            )
        # sorted by coordinate (as batches are contiguous), then cell:
        part, cell, dvec = map(np.concatenate, zip(*batches))
        assert np.array_equal(part, full_part)
        assert np.array_equal(cell, full_cell)
        assert np.array_equal(dvec, full_dvec)

    def test_exact_boundary(self) -> None:
        """Check that cells exactly at the search radius are included."""
        cell_centres, cell_sizes = _small_tree_grid()
        # exactly one unit below the centre of the first size-0.5 cell:
        target = 7 + 7
        coord = cell_centres[target] - [0.0, 0.0, 1.0]
        radius = 1.0
        for r, included in ((radius, True), (np.nextafter(radius, 0), False)):
            gs = Octree(cell_centres, cell_sizes).find(coord[np.newaxis], np.array([r]))
            assert (target in gs.cell_indices) == included

    def test_uniform_matches_grid_search(self) -> None:
        """Check that equal cells give the same result as the uniform grid search."""
        grid_shape = (7, 5, 6)
        cell_centres = np.indices(grid_shape).reshape(3, -1).T.astype(float)
        coords, radii = _random_input(grid_shape, 200, 3.0, margin=2)
        expected = find_grid_intersections(grid_shape, coords, radii)
        gs = Octree(cell_centres, np.ones(len(cell_centres))).find(coords, radii)
        for field in ("intersections", "cell_indices", "strides"):
            assert np.array_equal(getattr(gs, field), getattr(expected, field))
        assert np.allclose(gs.distances, expected.distances)

    def test_empty(self) -> None:
        """Check that we get empty results without coordinates or hits."""
        cell_centres, cell_sizes = _small_tree_grid()
        far = (np.array([[20.0, 0.0, 0.0]]), np.array([3.0]))
        for coords, radii in ((np.empty((0, 3)), np.empty(0)), far):
            gs = Octree(cell_centres, cell_sizes).find(coords, radii)
            assert gs.intersections.shape == (0,)
            assert gs.distances.shape == (0, 3)
            assert gs.cell_indices.shape == (0,)
            assert gs.strides.shape == (0, 2)
            # same data types as results with matches:
            assert gs.intersections.dtype == np.int32
            assert gs.cell_indices.dtype == gs.strides.dtype == np.int64
            assert np.array_equal(
                Octree(cell_centres, cell_sizes).count(coords, radii),
                np.zeros(coords.shape[0]),
            )
            gs = Octree(cell_centres, cell_sizes).find(
                coords, radii, return_distances=False
            )
            assert gs.distances is None

    def test_counts(self) -> None:
        """Check that counts agree with the number of matches found for each particle."""
        cell_centres, cell_sizes = _bisected_grid()
        coords, radii = _tree_input(cell_centres, 200, 3.0)
        counts = Octree(cell_centres, cell_sizes).count(coords, radii)
        gs = Octree(cell_centres, cell_sizes).find(coords, radii)
        assert np.array_equal(
            counts, np.bincount(gs.intersections, minlength=coords.shape[0])
        )

    def test_without_distances(self) -> None:
        """Check that skipping the distances leaves the other results unchanged."""
        cell_centres, cell_sizes = _bisected_grid()
        coords, radii = _tree_input(cell_centres, 100, 3.0)
        gs = Octree(cell_centres, cell_sizes).find(coords, radii)
        gs_nodist = Octree(cell_centres, cell_sizes).find(
            coords, radii, return_distances=False
        )
        assert gs_nodist.distances is None
        for field in ("intersections", "cell_indices", "strides"):
            assert np.array_equal(getattr(gs, field), getattr(gs_nodist, field))

    def test_threads(self) -> None:
        """Check that the result does not depend on the number of threads."""
        cell_centres, cell_sizes = _bisected_grid(depth=4)
        coords, radii = _tree_input(cell_centres, 300, 3.0)
        results = [
            Octree(cell_centres, cell_sizes).find(coords, radii, ncpu=ncpu)
            for ncpu in (1, 3)
        ]
        for a, b in zip(*results):
            assert np.array_equal(a, b)
        # with 3 threads the grid is split into many small work units:
        bf_part, bf_cell, bf_dvec = _brute_force_centres(cell_centres, coords, radii)
        part, cell, dvec = _unpack(results[1])
        assert np.array_equal(part, bf_part)
        assert np.array_equal(cell, bf_cell)
        assert np.allclose(dvec, bf_dvec)

    @pytest.mark.parametrize("max_cells", (1, 5, 40, 10**6))
    def test_work_units(self, max_cells: int) -> None:
        """
        Check that every cell is below exactly one work unit, and work units are small.

        Parallelization is first split across top-level tree cells. If these give a poor
        load balance, they are (recursively) unpacked into octants to provide finer
        division of work.
        """
        cell_centres, cell_sizes = _bisected_grid(depth=4)
        tree = Octree._build_octree(cell_centres, cell_sizes)
        n_below = _grid_search._cells_below(tree.first_child, tree.node_cell)
        assert n_below[tree.root_nodes[tree.root_nodes >= 0]].sum() == cell_sizes.size
        work_units = Octree._work_units(tree, n_below, max_cells)
        assert np.all(np.diff(work_units) > 0)
        cells = []
        for work_unit in work_units:
            assert tree.first_child[work_unit] < 0 or n_below[work_unit] <= max_cells
            todo = [work_unit]
            while todo:
                node = todo.pop()
                if tree.node_cell[node] >= 0:
                    cells.append(tree.node_cell[node])
                else:
                    todo.extend(tree.first_child[node] + np.arange(8))
        assert np.array_equal(np.sort(cells), np.arange(cell_sizes.size))

    @pytest.mark.parametrize("dist_dtype", (np.float64, np.float32))
    def test_dist_dtype(self, dist_dtype) -> None:
        """Check that distances have the requested dtype."""
        cell_centres, cell_sizes = _small_tree_grid()
        coords, radii = _tree_input(cell_centres, 50, 3.0)
        gs = Octree(cell_centres, cell_sizes).find(coords, radii, dist_dtype=dist_dtype)
        assert gs.distances.dtype == dist_dtype

    @pytest.mark.parametrize(
        "change, message",
        (
            ("no cells", "without cells"),
            ("size", "powers of two"),
            ("corner", "lattice"),
            ("missing", "tile"),
            ("duplicate", "duplicated"),
            ("overlap", "overlap"),
            ("deep", "Too many levels"),
        ),
    )
    def test_not_octree(self, change: str, message: str) -> None:
        """Check that grids that are not an octree are rejected."""
        cell_centres, cell_sizes = _small_tree_grid()
        if change == "no cells":
            cell_centres, cell_sizes = cell_centres[:0], cell_sizes[:0]
        elif change == "size":
            cell_sizes[-1] = 0.75
        elif change == "corner":
            cell_centres[0] += 0.25
        elif change == "missing":
            cell_centres, cell_sizes = cell_centres[:-1], cell_sizes[:-1]
        elif change == "duplicate":
            cell_centres = np.vstack((cell_centres, cell_centres[-1]))
            cell_sizes = np.r_[cell_sizes, cell_sizes[-1]]
        elif change == "overlap":  # the bisected size-1 cell is also kept
            cell_centres = np.vstack((cell_centres, [1.0, 1.0, 1.0]))
            cell_sizes = np.r_[cell_sizes, 1.0]
        elif change == "deep":  # a tiny cell at the corner of a size-0.5 cell
            cell_centres[-1] += 2.0**-41 - 0.25
            cell_sizes[-1] = 2.0**-40
        with pytest.raises(ValueError, match=message):
            Octree(cell_centres, cell_sizes)

    def test_input_validation(self) -> None:
        """Check that inconsistent inputs are rejected."""
        cell_centres, cell_sizes = _small_tree_grid()
        coords, radii = _tree_input(cell_centres, 5, 2.0)
        bad_cells = (
            ((cell_centres[:, :2], cell_sizes), "cell_centres must"),
            ((cell_centres, cell_sizes[:-1]), "cell_sizes must"),
        )
        for args, message in bad_cells:
            with pytest.raises(ValueError, match=message):
                Octree(*args)
        octree = Octree(cell_centres, cell_sizes)
        bad_searches = (
            ((coords[:, :2], radii), "coords must"),
            ((coords, radii[:-1]), "radii must"),
        )
        for args, message in bad_searches:
            with pytest.raises(ValueError, match=message):
                octree.find(*args)
            with pytest.raises(ValueError, match=message):
                octree.count(*args)

    def test_without_numba_threads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Check that the search runs without numba thread control."""
        cell_centres, cell_sizes = _bisected_grid()
        coords, radii = _tree_input(cell_centres, 50, 3.0)
        expected = Octree(cell_centres, cell_sizes).find(coords, radii)
        monkeypatch.setattr(_grid_search, "NUMBA_AVAILABLE", False)
        gs = Octree(cell_centres, cell_sizes).find(coords, radii)
        for a, b in zip(gs, expected):
            assert np.array_equal(a, b)
