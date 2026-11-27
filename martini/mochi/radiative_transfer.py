"""Base Mochi only includes optically thin models, but can be extended."""

import numpy as np
from typing import TYPE_CHECKING
from martini._util import NUMBA_AVAILABLE
from martini._grid_search import _threads
from martini.datacube import DataCube
from martini.spectral_models import _BaseSpectrum
from martini.mochi.cell_grid import CellGrid
import astropy.units as U

if NUMBA_AVAILABLE:
    from numba import njit
if TYPE_CHECKING or not NUMBA_AVAILABLE:
    prange = range  # pragma: no cover
else:
    from numba import prange


def _los_overlap(
    lower_cell_edge: float | np.ndarray,
    cell_size: float | np.ndarray,
    region_lower_bound: float,
    region_upper_bound: float,
) -> float | np.ndarray:
    """
    Evaluate the fraction of a cell's extent along the line of sight inside a region.

    Parameters
    ----------
    lower_cell_edge : float or ~numpy.ndarray
        Lower edge of the cell(s) along the line of sight, in pixels.

    cell_size : float or ~numpy.ndarray
        Size of the cell(s), in pixels.

    region_lower_bound : float
        Lower edge of the region along the line of sight, in pixels.

    region_upper_bound : float
        Upper edge of the region along the line of sight, in pixels.

    Returns
    -------
    float or ~numpy.ndarray
        The fraction of the extent of the cell(s) along the line of sight in the interval
        defined by the region bounds.
    """
    return (
        np.maximum(
            0.0,
            np.minimum(lower_cell_edge + cell_size, region_upper_bound)
            - np.maximum(lower_cell_edge, region_lower_bound),
        )
        / cell_size
    )


def _deposit_footprints(
    field_spectra: np.ndarray,
    weights: np.ndarray,
    footprint_begin: np.ndarray,
    footprint_end: np.ndarray,
    cube: np.ndarray,
) -> None:
    """
    Add the weighted spectrum of each cell to the pixels that it covers.

    Compiled with :mod:`numba` if it is available, in parallel over channels (so threads
    never write to the same pixel). The cells are added in order, skipping cells with zero
    weight.

    Parameters
    ----------
    field_spectra : ~numpy.ndarray
        Spectrum of each cell per pixel that it covers, shape ``(n_channels, n_cells)``.

    weights : ~numpy.ndarray
        Weight of each cell, shape ``(n_cells,)``.

    footprint_begin : ~numpy.ndarray
        First pixel covered by each cell along the two axes of the image, shape
        ``(n_cells, 2)``.

    footprint_end : ~numpy.ndarray
        One past the last pixel covered by each cell along the two axes of the image,
        shape ``(n_cells, 2)``.

    cube : ~numpy.ndarray
        The cube that the spectra are added to, shape ``(n_channels, ny, nz)``. Modified
        in place.
    """
    for c in prange(field_spectra.shape[0]):
        for i in range(field_spectra.shape[1]):
            if weights[i] == 0:
                continue
            value = weights[i] * field_spectra[c, i]
            for y in range(footprint_begin[i, 0], footprint_end[i, 0]):
                for z in range(footprint_begin[i, 1], footprint_end[i, 1]):
                    cube[c, y, z] += value


if NUMBA_AVAILABLE:
    _deposit_footprints = njit(parallel=True, cache=True)(_deposit_footprints)


def calculate_field_spectrum(
    cell_grid: CellGrid,
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    cell_volumes: U.Quantity[U.pix**3],
    cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled],
    cell_distances: U.Quantity[U.Mpc],
) -> U.Quantity[U.Msun / U.Mpc**2]:
    """
    Evaluate the spectrum given the fields interpolated onto the grid.

    The spectrum is the mass contained in a series of velocity bins, divided by the
    square of the distance of each cell from the observer.

    Parameters
    ----------
    cell_grid : ~martini.mochi.cell_grid.CellGrid
        The cell grid with interpolated fields available.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    spectral_model : _BaseSpectrum
        Spectral model used to evaluate the spectrum.

    cell_volumes : ~astropy.units.Quantity
        Rectilinear volume of each cell, with units of pixels cubed.

    cell_distortion_corrections : ~astropy.units.Quantity
        Dimensionless spherical distortion correction to the volume of each cell.

    cell_distances : ~astropy.units.Quantity
        Distance from the observer to each cell, with dimension of length.

    Returns
    -------
    ~astropy.units.Quantity
        The evaluated mass spectrum divided by distance squared, with shape
        ``(n_channels, n_cells)``.
    """
    field_temperatures = cell_grid.interpolated_fields["temperatures"]
    field_masses = cell_grid.interpolated_fields["masses_HI"]
    field_velocities = cell_grid.interpolated_fields["velocities"]
    field_temperatures[field_masses == 0] = 1 * field_temperatures.unit
    # field_masses is a density per unit corrected volume, convert to mass in each cell,
    # then divide by distance squared since flux scales with each cell's own distance:
    numerator = (
        field_masses
        * cell_volumes
        * cell_distortion_corrections
        / cell_distances.to(U.Mpc) ** 2
    )
    # Some refactoring needed here for better integration with spectral model, or maybe
    # some changes needed in spectral model code.
    # To consider:
    #  - "extra_data" is just explicitly given, but we can save even touching the
    #    temperatures at all in the DiracDeltaSpectrum case.
    #  - Casting to Quantity is not ideal, could use SpectralCoord for the interpolated
    #    velocity field (or can it at least be a view?).
    #  - This can presumably be parallelized like in the pre-calculated spectra.
    #    The spectral_model._BaseSpectrum assumes that we're dealing with particles stored
    #    in a SPHSource, but the calculation is basically the same when using cells here.
    #    Could add an abstraction layer so that we can work with any source of velocity
    #    information.
    field_spectrum = spectral_model.spectral_function(
        U.Quantity(datacube.velocity_channel_edges[np.newaxis, 1:]).astype(
            spectral_model.spec_dtype
        ),
        U.Quantity(datacube.velocity_channel_edges[np.newaxis, :-1]).astype(
            spectral_model.spec_dtype
        ),
        field_velocities[:, np.newaxis].astype(spectral_model.spec_dtype),
        extra_data={
            "sigma": np.sqrt(field_temperatures)[:, np.newaxis].astype(
                spectral_model.spec_dtype
            )
        },
    ).T
    return numerator * field_spectrum


def optically_thin(
    cell_grid: CellGrid,
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    index_type: type = np.uintc,
    ncpu: int = 1,
) -> U.Quantity[U.Msun / U.Mpc**2]:
    """
    Assemble fields into an HI cube using optically thin approximation.

    ??.

    Parameters
    ----------
    cell_grid : ~martini.mochi.cell_grid.CellGrid
        The cell grid with interpolated fields available.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    spectral_model : _BaseSpectrum
        The spectral model used to evaluate the spectra.

    index_type : type, optional
        The data type used to store cell indices.

    ncpu : int, optional
        Number of threads to use (if :mod:`numba` is available) to add the spectra of
        cells to the pixels that they cover. A value of ``-1`` means as many threads as
        available cores.

    Returns
    -------
    ~astropy.units.Quantity
        The mock spectral cube, of HI mass divided by distance squared in each pixel and
        channel.
    """
    adaptive = hasattr(cell_grid, "adaptive_cells")
    cells = cell_grid.adaptive_cells if adaptive else cell_grid.cells
    field_spectra = calculate_field_spectrum(
        cell_grid,
        datacube,
        spectral_model,
        cell_grid.adaptive_cell_volumes if adaptive else cell_grid.cell_volumes,
        (
            cell_grid.adaptive_cell_distortion_corrections
            if adaptive
            else cell_grid.cell_distortion_corrections
        ),
        cell_grid.adaptive_cell_distances if adaptive else cell_grid.cell_distances,
    )
    # The cell grid may extend beyond the target region to obey constraints on the cell
    # side length and refinement requirements. Mass in the parts of cells beyond the
    # target region along the LoS is here, assuming that the density is uniform within
    # each cell.
    region_lower_bound, region_upper_bound = cell_grid.los_region
    if adaptive or cell_grid.cell_size != 1:
        yz_cells = np.column_stack([cells[i] for i in "yz"]) + 0.5
        sizes = cells["size"][:, np.newaxis]
        cells_begin = np.floor(yz_cells).astype(index_type)
        cells_end = np.ceil(yz_cells + sizes).astype(index_type)
        if np.any(cells_end - cells_begin != np.maximum(sizes, 1)):
            raise RuntimeError(
                "Cell grid is not aligned with pixel grid, cells must have power-of-two "
                "sizes with edges on pixel edges."
            )
        footprints = np.prod(cells_end - cells_begin, axis=1)
        cube_unit = field_spectra.unit
        field_spectra = (field_spectra / footprints).to_value(cube_unit)
        cube = np.zeros(
            (
                field_spectra.shape[0],
                datacube.datacube_shape[0],
                datacube.datacube_shape[1],
            )
        )
        cell_los_lower_edge, cell_los_size = cells["x"], cells["size"]
        weights = np.where(
            (cell_los_lower_edge < region_lower_bound)
            | (cell_los_lower_edge + cell_los_size > region_upper_bound),
            _los_overlap(
                cell_los_lower_edge,
                cell_los_size,
                region_lower_bound,
                region_upper_bound,
            ),
            1.0,
        )
        with _threads(ncpu):
            _deposit_footprints(
                np.ascontiguousarray(field_spectra),
                weights,
                np.ascontiguousarray(cells_begin),
                np.ascontiguousarray(cells_end),
                cube,
            )
        cube *= cube_unit
    else:
        # Cell grid maps one-to-one onto data cube pixels.
        grid_shape = cell_grid.grid_shape
        layer_weights = np.array(
            [
                _los_overlap(lower, 1.0, *cell_grid.los_region)
                for lower in cells["x"].reshape(grid_shape)[:, 0, 0]
            ]
        )
        cube = (
            np.einsum(
                "czxy,z->cxy",
                field_spectra.value.reshape((-1,) + grid_shape),
                layer_weights,
            )
            * field_spectra.unit
        )
    return np.flip(np.moveaxis(cube, 1, 2), axis=2)
