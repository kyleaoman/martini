"""Tests of the optimized grid search methods."""

import numpy as np
from martini._grid_search import find_grid_intersections


class TestGridSearch:
    """Tests of the optimized grid search methods."""

    def test_matches_brute_force(self):
        """Check that optimized search gives same result as a brute force search."""
        n_px = 128
        n = 1000
        i_grid, j_grid = np.meshgrid(np.arange(n_px), np.arange(n_px))
        cell_centres = np.array((i_grid.flatten(), j_grid.flatten())).T
        np.random.seed(0)
        coords = np.column_stack((np.random.rand(n), np.random.rand(n))) * n_px
        radii = np.random.rand(n) * 4.75 + 0.25
        # Record results of a brute-force search as a benchmark:
        bf_cell_hits = []
        bf_distances = []
        for idx, ij_px in enumerate(cell_centres):
            dists = np.sqrt(np.sum((ij_px - coords) ** 2, axis=1))
            mask = dists <= radii
            for m, d in zip(np.argwhere(mask.flatten()), dists[mask]):
                bf_cell_hits.append(m)
                bf_distances.append(d)
        bf_cell_hits = np.array(bf_cell_hits).squeeze()
        bf_distances = np.array(bf_distances).squeeze()
        # Evaluate through optimized routine:
        gs = find_grid_intersections(
            cell_centres,
            coords,
            radii,
        )
        cell_hits = gs.intersections
        # Check equality:
        assert (np.sort(cell_hits) == np.sort(bf_cell_hits)).all()
        # We expect some particles to be small and miss touching any cell centres:
        assert (radii < 0.5 * np.sqrt(2)).any()
        assert np.unique(cell_hits).size < n  # strictly less

    def test_all_coords_matched(self):
        """
        Check that with floored radii we never miss any particles.

        Not clear how to handle adaptive grid here, as flooring at the coarse grid scale
        will catch too many fine cells, but flooring at the fine scale can miss particles
        in coarse zones.
        """
        n_px = 128
        n = 1000
        i_grid, j_grid = np.meshgrid(np.arange(n_px), np.arange(n_px))
        cell_centres = np.array((i_grid.flatten(), j_grid.flatten())).T
        np.random.seed(0)
        coords = np.column_stack((np.random.rand(n), np.random.rand(n))) * n_px - 0.5
        radii = np.random.rand(n) * 4.75 + 0.25  # some particles small enough to "miss"
        gs = find_grid_intersections(
            cell_centres,
            coords,
            np.clip(radii, 0.5 * np.sqrt(2), np.inf),  # floored
        )
        cell_hits = gs.intersections
        assert np.unique(cell_hits).size == n
