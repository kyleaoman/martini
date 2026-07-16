"""Provide tools to search for particles touching grid cell points."""

import numpy as np
from collections import namedtuple


def find_grid_intersections(
    cell_centres: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    non_uniform: bool = False,
    cell_sizes: np.ndarray | None = None,
    many_intersections_threshold: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Search for coordinates that reach grid locations within variable search radii.

    This implementation is valid in any number of dimensions and can handle both
    cell grids where all cells have the same size, or non-uniform grids. Grid cells must,
    however, be cubic. The algorithm chosen is fast with the caveat that it can produce a
    relatively small number of false-positive matches (usually near a cell corner where
    the radius does not quite reach the centre). In the context of kernel-based imaging,
    this implies a few extra kernel evaluations, but those evaluations will be zero so
    the final result is unaffected. Search coordinates with large radii that could overlap
    with very many cells (defined by a threshold) are handled separately with a slower
    method to avoid excessive memory usage. If most search areas cover most of the grid
    this function is not optimal.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Coordinates of cell centres, with one column per dimension. In :mod:`martini` this
        means the integer pixel coordinates (``(0 * U.pix, 0 * U.pix)`` is the centre of
        the first pixel).

    coords : ~numpy.ndarray
        Coordinates to search around for cells, with one column per dimension. In
        :mod:`martini`, this means the ``pixcoords`` of the particles (a particle at
        ``(0 * U.pix, 0 * U.pix)`` is centred in the first pixel).

    radii : ~numpy.ndarray
        Search radii around the ``coords``. Must be an array with the same length as the
        number of particles, a scalar is not accepted.

    non_uniform : bool
        If ``True``, the grid cells can have different sizes (e.g. a quad-tree or oct-tree
        adaptively refined grid). In this case ``cell_sizes`` must be passed. If the grid
        is uniform leave set to ``False`` (the default) for faster execution.

    cell_sizes : ~numpy.ndarray, optional
        The side lengths of the cells when the side lengths are not all equal (see
        ``non_uniform``).

    many_intersections_threshold : int
        When a search area can enclose more than this many grid points, it is handled
        separately (one at a time) to avoid excessive memory usage.

    Returns
    -------
    intersections : ~numpy.ndarray
        Each entry is the index of a search coordinate (index into ``coords``),
        representing a search coordinate matched to a cell. Each search coordinate may
        match to many cells and therefore appear multiple times. Results are grouped by
        the cell indices.

    strides : ~numpy.ndarray
        Each row can be used as a range to select rows from ``intersections`` that share
        a common cell index.

    cell_indices : ~numpy.ndarray
        The unique cell indices (index into ``cell_centres``). Each entry in this array
        corresponds to a range in ``strides``.
    """
    n_cells, n_dim = cell_centres.shape

    if non_uniform and cell_sizes is None:
        raise ValueError("`cell_sizes` must be passed when `non_uniform=True`.")

    min_grid = np.min(cell_centres, axis=0)
    max_grid = np.max(cell_centres, axis=0)
    min_bounds = np.maximum(min_grid, np.ceil(coords - radii[:, np.newaxis])).astype(
        int
    )
    max_bounds = np.minimum(max_grid, np.floor(coords + radii[:, np.newaxis])).astype(
        int
    )
    grid_shape = (max_grid - min_grid + 1).astype(int)
    dim_strides = np.ones(n_dim, dtype=int)
    if n_dim > 1:
        dim_strides[:-1] = np.cumprod(grid_shape[::-1])[::-1][1:]

    # Compute continuous 1D hashes to track valid grid locations
    valid_hashes = np.sum((cell_centres - min_grid) * dim_strides, axis=1)

    if non_uniform:
        cell_sort = np.argsort(valid_hashes)
        sorted_cell_hashes = valid_hashes[cell_sort]
        _, cell_bucket_splits, cell_bucket_counts = np.unique(
            sorted_cell_hashes, return_index=True, return_counts=True
        )

        max_hash = np.max(valid_hashes) + 1
        bucket_to_cell_start = np.full(max_hash, -1, dtype=int)
        bucket_to_cell_count = np.zeros(max_hash, dtype=int)

        unique_hashes = sorted_cell_hashes[cell_bucket_splits]
        bucket_to_cell_start[unique_hashes] = cell_bucket_splits
        bucket_to_cell_count[unique_hashes] = cell_bucket_counts
    else:
        hash_to_cell_idx = np.full(np.max(valid_hashes) + 1, -1, dtype=int)
        hash_to_cell_idx[valid_hashes] = np.arange(n_cells)

    # First we process radii that are "small" (below the many_intersections_threshold).
    spans = np.maximum(0, max_bounds - min_bounds + 1)
    # How many pixels does each coordinate risk hitting within its range:
    pairs_per_data = np.prod(spans, axis=1)
    is_huge = pairs_per_data > many_intersections_threshold
    small_pairs = pairs_per_data[np.logical_not(is_huge)]
    total_small_elements = np.sum(small_pairs)
    small_data_indices = np.flatnonzero(np.logical_not(is_huge))
    rep_data_idx = np.repeat(small_data_indices, small_pairs)
    cum_offsets = np.zeros(len(small_pairs) + 1, dtype=np.int64)
    np.cumsum(small_pairs, out=cum_offsets[1:])
    local_idx = np.arange(total_small_elements) - np.repeat(
        cum_offsets[:-1], small_pairs
    )

    rep_spans = np.repeat(spans[np.logical_not(is_huge)], small_pairs, axis=0)
    span_strides = np.ones((len(local_idx), n_dim), dtype=int)
    if n_dim > 1:
        span_strides[:, :-1] = np.cumprod(rep_spans[:, ::-1], axis=1)[:, ::-1][:, 1:]

    offsets = np.zeros((total_small_elements, n_dim), dtype=int)
    rem = local_idx.copy()
    for d in range(n_dim):
        offsets[:, d] = rem // span_strides[:, d]
        rem %= span_strides[:, d]

    flat_coords = (
        np.repeat(min_bounds[np.logical_not(is_huge)], small_pairs, axis=0) + offsets
    )
    flat_hashes = np.sum((flat_coords - min_grid) * dim_strides, axis=1)

    if non_uniform:
        valid_bucket_mask = np.logical_and(
            flat_hashes < max_hash,
            bucket_to_cell_count[flat_hashes] > 0,
        )
        flat_hashes = flat_hashes[valid_bucket_mask]
        rep_data_idx = rep_data_idx[valid_bucket_mask]

        c_starts = bucket_to_cell_start[flat_hashes]
        c_counts = bucket_to_cell_count[flat_hashes]

        final_data_indices = np.repeat(rep_data_idx, c_counts)
        cell_offsets = np.zeros(len(c_counts) + 1, dtype=int)
        np.cumsum(c_counts, out=cell_offsets[1:])

        cell_local_sequence = np.arange(cell_offsets[-1]) - np.repeat(
            cell_offsets[:-1], c_counts
        )
        final_cell_indices = cell_sort[
            np.repeat(c_starts, c_counts) + cell_local_sequence
        ]

        sq_distances = np.sum(
            (coords[final_data_indices] - cell_centres[final_cell_indices]) ** 2, axis=1
        )
        exact_mask = sq_distances <= (radii[final_data_indices] ** 2)

        final_data_indices = final_data_indices[exact_mask]
        matched_cell_indices = final_cell_indices[exact_mask]
        distances = np.sqrt(sq_distances[exact_mask])
    else:
        valid_mask = np.logical_and(
            flat_hashes < len(hash_to_cell_idx),
            np.isin(flat_hashes, valid_hashes),
        )
        flat_coords = flat_coords[valid_mask]
        matched_cell_indices = hash_to_cell_idx[flat_hashes[valid_mask]]
        final_data_indices = rep_data_idx[valid_mask]
        sq_distances = np.sum(
            (coords[final_data_indices] - cell_centres[matched_cell_indices]) ** 2,
            axis=1,
        )
        exact_mask = sq_distances <= (radii[final_data_indices] ** 2)

        matched_cell_indices = matched_cell_indices[exact_mask]
        final_data_indices = final_data_indices[exact_mask]
        distances = np.sqrt(sq_distances[exact_mask])

    # Next we process radii that are "large" (above the many_intersections_threshold).
    # This is needed to avoid huge memory usage.
    huge_data_indices = np.flatnonzero(is_huge)
    if len(huge_data_indices) > 0:
        huge_data_lists = [final_data_indices]
        huge_cell_lists = [matched_cell_indices]
        huge_dist_lists = [distances]

        for idx in huge_data_indices:
            coord = coords[idx]
            radius = radii[idx]

            sq_diffs = np.sum((cell_centres - coord) ** 2, axis=1)
            spatial_mask = sq_diffs <= (radius**2)

            if np.any(spatial_mask):
                huge_data_lists.append(np.full(np.sum(spatial_mask), idx, dtype=int))
                huge_dist_lists.append(np.sqrt(sq_diffs[spatial_mask]))
                if non_uniform:
                    huge_cell_lists.append(np.flatnonzero(spatial_mask))
                else:
                    hashes = np.sum(
                        (cell_centres[spatial_mask] - min_grid) * dim_strides, axis=1
                    )
                    huge_cell_lists.append(hash_to_cell_idx[hashes])

        # flat_coords = np.concatenate(huge_coord_lists, axis=0)
        final_data_indices = np.concatenate(huge_data_lists, axis=0)
        matched_cell_indices = np.concatenate(huge_cell_lists, axis=0)
        distances = np.concatenate(huge_dist_lists, axis=0)

    # Finally organise data to return.
    sort_idx = np.argsort(matched_cell_indices)
    intersections = final_data_indices[sort_idx]
    sorted_distances = distances[sort_idx]
    cell_indices, split_indices, counts = np.unique(
        matched_cell_indices[sort_idx], return_index=True, return_counts=True
    )
    strides = np.column_stack((split_indices, split_indices + counts))

    ret_tuple = namedtuple(
        "ret_tuple", ["intersections", "strides", "cell_indices", "distances"]
    )
    return ret_tuple(
        intersections=intersections,
        strides=strides,
        cell_indices=cell_indices,
        distances=sorted_distances,
    )
