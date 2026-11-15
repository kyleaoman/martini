"""Tests of the optimized grid search methods."""

import pytest
import numpy as np
from martini import _grid_search
from martini._grid_search import (
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
        assert gs.distances.shape == (0,)
        assert gs.cell_indices.shape == (0,)
        assert gs.strides.shape == (0, 2)

    @pytest.mark.parametrize("grid_shape", ((37, 64), (24, 20, 16)))
    def test_counts(self, grid_shape: tuple[int, ...]) -> None:
        """Check that counts agree with the number of matches found for each particle."""
        coords, radii = _random_input(grid_shape, 100, 3.0, margin=2)
        counts = count_grid_intersections(grid_shape, coords, radii)
        gs = find_grid_intersections(grid_shape, coords, radii)
        assert np.array_equal(
            counts, np.bincount(gs.intersections, minlength=coords.shape[0])
        )
        # passing counts in gives the same result:
        gs_counts = find_grid_intersections(grid_shape, coords, radii, counts=counts)
        for a, b in zip(gs, gs_counts):
            assert np.array_equal(a, b)

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

    def test_without_numba_threads(self, monkeypatch: pytest.Monkeypatch) -> None:
        """Check that the search runs without numba thread control."""
        coords, radii = _random_input((37, 64), 50, 3.0)
        expected = find_grid_intersections((37, 64), coords, radii)
        monkeypatch.setattr(_grid_search, "NUMBA_AVAILABLE", False)
        gs = find_grid_intersections((37, 64), coords, radii)
        for a, b in zip(gs, expected):
            assert np.array_equal(a, b)
