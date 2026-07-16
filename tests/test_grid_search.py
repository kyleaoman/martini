"""Tests of the optimized grid search methods."""

import pytest
import numpy as np
from martini._grid_search import find_grid_intersections


class TestGridSearch:
    """Tests of the optimized grid search methods."""

    @pytest.mark.parametrize("non_uniform", (True, False))
    def test_matches_brute_force(self, non_uniform):
        """
        Check that the optimized search gives the same result as a brute force search.

        We check both the uniform and non-uniform cases, but with a uniform grid both
        times in this test. This checks that the non-uniform search also gives the same
        result, just less optimally.
        """
        n_px = 128
        n = 1000
        i_grid, j_grid = np.meshgrid(np.arange(n_px), np.arange(n_px))
        cell_centres = np.array((i_grid.flatten(), j_grid.flatten())).T
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
        cell_hits, strides, cell_indices, distances = find_grid_intersections(
            cell_centres,
            coords,
            radii,
            non_uniform=non_uniform,
            cell_sizes=np.ones(len(cell_centres)),
        )
        # Check equality:
        assert (np.sort(cell_hits) == np.sort(bf_cell_hits)).all()
        # We expect some particles to be small and miss touching any cell centres:
        assert (radii < 0.5 * np.sqrt(2)).any()
        assert np.unique(cell_hits).size < n  # strictly less

    @pytest.mark.parametrize("non_uniform", (True, False))
    def test_all_coords_matched(self, non_uniform):
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
        coords = np.column_stack((np.random.rand(n), np.random.rand(n))) * n_px
        radii = np.random.rand(n) * 4.75 + 0.25  # some particles small enough to "miss"
        cell_hits, strides, cell_indices, distances = find_grid_intersections(
            cell_centres,
            coords,
            np.clip(radii, 0.5 * np.sqrt(2), np.inf),  # floored
            non_uniform=non_uniform,
            cell_sizes=np.ones(len(cell_centres)),
        )
        assert np.unique(cell_hits).size == n
