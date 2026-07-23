"""Provide tools to search for particles touching grid cell points."""

import itertools
import numpy as np
from scipy.spatial import KDTree
from typing import NamedTuple


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
        The Euclidian distance between the search coordinate and cell centre for each
        match.

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


def build_tree(cell_centres: np.ndarray) -> KDTree:
    """
    Build a KDTree from the cell centres.

    Parameters
    ----------
    cell_centres : ~numpy.ndarray
        Coordinates of cell centres, with one column per dimension. In :mod:`martini` this
        means the integer pixel coordinates (``(0 * U.pix, 0 * U.pix)`` is the centre of
        the first pixel).
    """
    return KDTree(cell_centres, compact_nodes=True, balanced_tree=True)


def find_grid_intersections(
    grid_tree: KDTree,
    cell_centres: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    ncpu: int = 1,
) -> FindGridIntersectionsResult:
    r"""
    Search for coordinates that reach grid locations within variable search radii.

    This implementation is valid in any number of dimensions and can handle both
    cell grids where all cells have the same size, or non-uniform grids.

    Note that a search coordinate with a small radius can slip "between" cell centres if
    the sphere does not enclose any cell centres. For uniform grids this can be mitigated
    by flooring the search radius at :math:`\sqrt{n_\mathrm{dim}}/2`. This assumes that
    grid cells are (hyper)cubic.

    In the context of :mod:`martini` these particles will normally evaluate the
    Dirac-delta kernel which will ensure that all of their contribution is deposited in
    the grid cell containing them, even if their artificially increased search radius
    reaches multiple grid cell centres.

    For non-uniform grids the flooring strategy fails and no strategy has been identified,
    yet, but :mod:`mochi` may provide inspiration.

    Parameters
    ----------
    grid_tree : ~scipy.spatial.KDTree
        A :class:`~scipy.spatial.KDTree` built from the pixel indices.

    cell_centres : ~numpy.ndarray
        Coordinates of cell centres, with one column per dimension. In :mod:`martini` this
        means the integer pixel coordinates (``(0 * U.pix, 0 * U.pix)`` is the centre of
        the first pixel).

    coords : ~numpy.ndarray
        Coordinates to search around for cells, with one column per dimension. In
        :mod:`martini`, this means the ``coords`` of the particles (a particle at
        ``(0 * U.pix, 0 * U.pix)`` is centred in the first pixel).

    radii : ~numpy.ndarray
        Search radii around the ``coords``. Must be an array with the same length as the
        number of particles, a scalar is not accepted.

    ncpu : int
        Number of cores to use for KDTree query.

    Returns
    -------
    intersections : ~numpy.ndarray
        Each entry is the index of a search coordinate (index into ``coords``),
        representing a search coordinate matched to a cell. Each search coordinate may
        match to many cells and therefore appear multiple times. Results are grouped by
        the cell indices.

    distances : ~numpy.ndarray
        The Euclidian distance between the search coordinate and cell centre for each
        match.

    cell_indices : ~numpy.ndarray
        The unique cell indices (index into ``cell_centres``). Each entry in this array
        corresponds to a range in ``strides``.

    strides : ~numpy.ndarray
        Each row can be used as a range to select rows from ``intersections`` that share
        a common cell index.
    """
    candidate_lists = grid_tree.query_ball_point(x=coords, r=radii, p=2.0, workers=ncpu)
    data_counts = np.array([len(lst) for lst in candidate_lists], dtype=np.int64)
    total_intersections = np.sum(data_counts)

    if total_intersections == 0:
        return FindGridIntersectionsResult(
            intersections=np.empty(0),
            distances=np.empty(0),
            cell_indices=np.empty(0),
            strides=np.empty((0, 2)),
        )

    flat_data_indices = np.repeat(np.arange(len(coords)), data_counts)
    flat_cell_indices = np.fromiter(
        itertools.chain.from_iterable(candidate_lists), dtype=np.int32
    )
    distances = coords[flat_data_indices] - cell_centres[flat_cell_indices]  # vectors

    sort_idx = np.argsort(flat_cell_indices)
    intersections = flat_data_indices[sort_idx]
    sorted_distances = distances[sort_idx]
    cell_indices, split_indices, counts = np.unique(
        flat_cell_indices[sort_idx], return_index=True, return_counts=True
    )
    strides = np.column_stack((split_indices, split_indices + counts))
    return FindGridIntersectionsResult(
        intersections=intersections,
        distances=sorted_distances,
        cell_indices=cell_indices,
        strides=strides,
    )
