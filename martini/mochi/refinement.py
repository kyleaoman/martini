"""Provide tools for recursively refining a cell grid."""

import numpy as np
from astropy import units as U
from typing import Callable
from functools import partial
from martini.mochi._dtypes import CELL_DTYPE as _CELL_DTYPE

_RF = np.sqrt(3) / 2


def _refine_grid_bisect(
    cell: np.ndarray,
    mask: bool | np.ndarray,
    in_cell: np.ndarray,
    new_cells: list[tuple[float, float, float, float]],
    new_cells_over: list[bool],
    new_cells_masks: list,
) -> None:
    """
    Bisect operation for grid refinement algorithms.

    Parameters
    ----------
    cell : ~numpy.ndarray
        Lower left coordinate and size of cell. Coordinate elements can be accessed as
        ``cell["x"]``, etc., and size as ``cell["size"]``.

    mask : bool or np.ndarray
        ??.

    in_cell : ~numpy.ndarray
        Flag for each particle indicating whether it is in this cell.

    new_cells : list
        ??.

    new_cells_over : list
        ??.

    new_cells_masks : list
        ??.
    """
    new_size = cell["size"] / 2.0
    new_cells.extend(
        [
            (
                float(cell["x"] + dx * new_size),
                float(cell["y"] + dy * new_size),
                float(cell["z"] + dz * new_size),
                float(new_size),
            )
            for dx in range(2)
            for dy in range(2)
            for dz in range(2)
        ]
    )
    new_cells_over.extend([False] * 8)
    if isinstance(mask, bool):
        raise ValueError("Expected a mask array, got a boolean.")
    new_cells_masks.extend([mask[in_cell]] * 8)
    return


def _pass_complete_cell(cells_lists: list, content_list: list) -> None:
    """
    Append new cells to a cell list.

    Parameters
    ----------
    cells_lists : list
        ??.

    content_list : list
        ??.
    """
    for i in range(len(cells_lists)):
        cells_lists[i].append(content_list[i])


def _refine_grid(
    in_cell_function: Callable,  # fill in arg & return types
    bisect_condition: Callable,  # fill in arg & return types
    cells: np.ndarray,
    positions: U.Quantity[U.pix],
    radii: U.Quantity[U.pix],
    threshold: float,
    stop_iter: int = 8,
) -> np.ndarray:
    """
    Start from a coarse grid, refine until no cells satisfy ``bisect_condition``.

    Parameters
    ----------
    in_cell_function : Callable
        Delimits the particles to consider for the cell.

    bisect_condition : Callable
        If returns ``True``, the cell is bisected.

    cells : ~numpy.ndarray
        Contains entries with the 3D position of the lower corner (accessible as
        ``cells["x"]``, etc.) and sizes (accessible as ``cells["size"]``) of the initial
        cells.

    positions : ~astropy.units.Quantity
        Array of particle positions with units of pixels.

    radii : ~astropy.units.Quantity
        Array of particle smoothing lengths with units of pixels.

    threshold : float
        Sensitivity threshold used by ``bisect_condition``.

    stop_iter : int
        Maximum recursive splitting depth.

    Returns
    -------
    ~numpy.ndarray
        Contains entries with the 3D position of the lower corner (accessible as
        ``cells["x"]``, etc.) and sizes (accessible as ``cells["size"]``) of the refined
        cell grid.
    """
    cells_number = len(cells)
    cells_over = [False] * cells_number
    cells_masks: list[bool | np.ndarray] = [np.arange(len(radii))] * cells_number
    new_cells: list[tuple[float, float, float, float]] = []
    new_cells_over: list[bool] = []
    new_cells_masks: list[bool | np.ndarray] = []

    iter = 0
    while iter < stop_iter:
        for n in range(cells_number):
            cell = cells[n]
            if cells_over[n]:
                _pass_complete_cell(
                    [new_cells, new_cells_over, new_cells_masks], [cells[n], True, True]
                )
                continue
            in_cell = in_cell_function(
                cells_masks[n], positions, radii, cell, threshold
            )
            if bisect_condition(
                cell["size"], in_cell, threshold, radii[cells_masks[n]]
            ):
                _refine_grid_bisect(
                    cell,
                    cells_masks[n],
                    in_cell,
                    new_cells,
                    new_cells_over,
                    new_cells_masks,
                )
            else:
                _pass_complete_cell(
                    [new_cells, new_cells_over, new_cells_masks], [cells[n], True, True]
                )
        cells = np.array(new_cells, dtype=_CELL_DTYPE)

        if len(cells) == cells_number or iter == stop_iter:
            break
        cells_number = len(cells)
        cells_over = new_cells_over
        cells_masks = new_cells_masks
        new_cells = []
        new_cells_over = []
        new_cells_masks = []
        iter += 1
    return cells


def _occupancy_in_cell(
    mask: np.ndarray,
    particles_pos: U.Quantity[U.pix],
    particles_radii: U.Quantity[U.pix],
    cell: np.void,
    threshold: float,  # is this needed by this or any other occupancy function?
) -> np.ndarray:
    """
    Describe.

    ??.

    Parameters
    ----------
    mask : ~numpy.ndarray
        ??.

    particles_pos : ~astropy.units.Quantity
        Particle locations in the cell grid with units of pixels.

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.

    cell : ~numpy.void
        The cell to consider. Cells are encoded as the coordinates of the lower corner
        (accessible as ``cell["x"]``, etc.) and the side length (``cell["size"]``).

    threshold : float
        ??.
    """
    return (
        np.sum(
            np.abs(
                particles_pos[mask]
                - [cell["x"], cell["y"], cell["z"]]
                - cell["size"] / 2
            ),
            axis=1,
        )
        < cell["size"] * 2
    )


def _is_not_single_occupancy(
    cell_size: float,
    in_cell: np.ndarray,
    threshold: float,
    particles_radii: U.Quantity[U.pix],
) -> bool:
    """
    Describe.

    ??.

    Parameters
    ----------
    cell_size : float
        ??.

    in_cell : ~numpy.ndarray
        ??.

    threshold : float
        ??.

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.
    """
    count = np.sum(in_cell)
    return count > threshold


def _intersect_in_cell(
    mask: np.ndarray,
    particles_pos: U.Quantity[U.pix],
    particles_radii: U.Quantity[U.pix],
    cell: np.void,
    threshold: float,  # is this needed by this or any other occupancy function?
) -> np.ndarray:
    """
    Describe.

    ??.

    Parameters
    ----------
    mask : ~numpy.ndarray
        ??.

    particles_pos : ~astropy.units.Quantity
        Particle locations in the cell grid with units of pixels.

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.

    cell : ~numpy.void
        The cell to consider. Cells are encoded as the coordinates of the lower corner
        (accessible as ``cell["x"]``, etc.) and the side length (``cell["size"]``).

    threshold : float
        ??.
    """
    small_particle = (
        particles_radii[mask] * threshold < cell["size"]
    )  # No need to consider particles larger than cell
    return (
        np.linalg.norm(
            particles_pos[mask] - [cell["x"], cell["y"], cell["z"]] - cell["size"] / 2,
            axis=1,
        )
        < particles_radii[mask] + cell["size"] * _RF
    ) & small_particle


def _is_any_particle_included(
    cell_size: float,
    in_cell: np.ndarray,
    threshold: float,
    particles_radii: U.Quantity[U.pix],
) -> np.bool:
    """
    Describe.

    ??.

    Parameters
    ----------
    cell_size : float
        ??.

    in_cell : ~numpy.ndarray
        ??.

    threshold : float
        ??.

    particles_radii : ~astropy.units.Quantity
        Particle sizes (i.e. smoothing ranges) in units of pixels.
    """
    return np.any(in_cell)


refine_grid_to_occupancy = partial(
    _refine_grid, _occupancy_in_cell, _is_not_single_occupancy
)
refine_grid_to_particle_scale = partial(
    _refine_grid, _intersect_in_cell, _is_any_particle_included
)
