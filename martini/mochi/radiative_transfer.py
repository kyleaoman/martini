"""Base Mochi only includes optically thin models, but can be extended."""

import warnings
import numpy as np
from martini.datacube import DataCube
from martini.spectral_models import _BaseSpectrum
from martini.mochi.cell_grid import AdaptiveCellGrid
import astropy.units as U
from typing import Callable


def calculate_field_spectrum(
    adaptive_cell_grid: AdaptiveCellGrid,
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    cell_volume: U.Quantity[U.pix**3],
) -> U.Quantity[U.Msun]:
    """
    Evaluate the spectrum given the fields interpolated onto the grid.

    The spectrum is the mass contained in a series of velocity bins.

    Parameters
    ----------
    adaptive_cell_grid : ~martini.mochi.cell_grid.AdaptiveCellGrid
        The cell grid with interpolated fields available.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    spectral_model : _BaseSpectrum
        Spectral model used to evaluate the spectrum.

    cell_volume : ~astropy.units.Quantity
        Volume of a grid cell with units of pixels cubed.

    Returns
    -------
    ~astropy.units.Quantity
        The evaluated mass spectrum.
    """
    field_temperatures = adaptive_cell_grid.interpolated_fields["temperatures"]
    field_masses = adaptive_cell_grid.interpolated_fields["masses_HI"]
    field_velocities = adaptive_cell_grid.interpolated_fields["velocities"]
    field_temperatures[field_masses == 0] = 1 * field_temperatures.unit
    numerator = field_masses * cell_volume
    # Some refactoring needed here for better integration with spectral model, or maybe
    # some changes needed in spectral model code.
    # To consider:
    #  - "extra_data" is just explicitly given, but we can save even touching the
    #    temperatures at all in the DiracDeltaSpectrum case.
    #  - The edge slice logic is repeated from the spectral_models.py code, maybe move
    #    that to helpers on the datacube object.
    #  - Casting to Quantity is not ideal, could use SpectralCoord for the interpolated
    #    velocity field (or can it at least be a view?).
    #  - This can presumably be parallelized like in the pre-calculated spectra.
    #    The spectral_model._BaseSpectrum assumes that we're dealing with particles stored
    #    in a SPHSource, but the calculation is basically the same when using cells here.
    #    Could add an abstraction layer so that we can work with any source of velocity
    #    information.
    if all(np.diff(datacube.velocity_channel_edges) > 0):
        lower_edges_slice: slice = np.s_[:-1]
        upper_edges_slice: slice = np.s_[1:]
    elif all(np.diff(datacube.velocity_channel_edges) < 0):
        lower_edges_slice = np.s_[1:]
        upper_edges_slice = np.s_[:-1]
    else:
        raise ValueError("Channel edges are not monotonic sequence.")
    field_spectrum = spectral_model.spectral_function(
        U.Quantity(
            datacube.velocity_channel_edges[lower_edges_slice, np.newaxis]
        ).astype(spectral_model.spec_dtype),
        U.Quantity(
            datacube.velocity_channel_edges[upper_edges_slice, np.newaxis]
        ).astype(spectral_model.spec_dtype),
        field_velocities[np.newaxis].astype(spectral_model.spec_dtype),
        extra_data={
            "sigma": np.sqrt(field_temperatures)[np.newaxis].astype(
                spectral_model.spec_dtype
            )
        },
    )
    return numerator * field_spectrum


def optically_thin(
    cell_grid: AdaptiveCellGrid,  # should actually be CellGrid when implemented
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    **kwargs,
) -> U.Quantity[U.Msun]:
    """
    Assemble fields into an HI cube using optically thin approximation.

    ??.

    Parameters
    ----------
    cell_grid : AdaptiveCellGrid
        The cell grid with interpolated fields available.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    spectral_model : _BaseSpectrum
        The spectral model used to evaluate spectra.

    Returns
    -------
    ~astropy.units.Quantity
        The mock spectral cube.
    """
    field_mHI = cell_grid.interpolated_fields["masses_HI"].reshape(
        cell_grid.final_grid_shape
    )
    field_temperature = cell_grid.interpolated_fields["temperatures"].reshape(
        cell_grid.final_grid_shape
    )
    field_velocity = cell_grid.interpolated_fields["velocities"].reshape(
        cell_grid.final_grid_shape
    )
    field_temperature[field_mHI == 0] = 1 * field_temperature.unit
    numerator = (
        field_mHI
        / np.sqrt(2 * np.pi * field_temperature)  # not so sure about this...
        * np.abs(np.diff(datacube.velocity_channel_edges))[:, np.newaxis]
        * cell_grid.final_cell_volume
    )
    cube = (
        np.zeros(
            (
                datacube.n_channels,
                cell_grid.final_grid_shape[1],
                cell_grid.final_grid_shape[2],
            )
        )
        * numerator.unit
    )
    diff = (
        field_velocity[None, ...] - datacube.velocity_channel_mids[:, None, None, None]
    )
    gaussians = np.exp(-(diff**2) / (2 * field_temperature[None, ...]))
    cube = np.sum(numerator[None, ...] * gaussians, axis=1)  # sum over LOS axis
    cube = np.flip(np.moveaxis(cube, 1, 2), axis=2)
    return cube


def adaptive_optically_thin(
    adaptive_cell_grid: AdaptiveCellGrid,
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    *,
    index_type: type = np.uintc,
    default_renderer: Callable = optically_thin,  # fill in arg & return types
    **kwargs,
) -> U.Quantity[U.Msun]:
    """
    Assemble fields into an HI cube using optically thin approximation.

    ??.

    Parameters
    ----------
    adaptive_cell_grid : ~martini.mochi.cell_grid.AdaptiveCellGrid
        The cell grid with interpolated fields available.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    spectral_model : _BaseSpectrum
        The spectral model used to evaluate the spectra.

    index_type : type, optional
        The data type used to store cell indices.

    default_renderer : Callable, optional
        The renderer to fall back to if this one cannot be used (usually because ``cells``
        was not provided).

    Returns
    -------
    ~astropy.units.Quantity
        The mock spectral cube.
    """
    if not hasattr(adaptive_cell_grid, "adaptive_cells"):
        warnings.warn(
            f"Expected adaptive cells, trying {default_renderer.__name__} instead.",
            UserWarning,
        )
        return default_renderer(
            adaptive_cell_grid,
            datacube,
            spectral_model,
            **kwargs,
        )
    x0 = np.min(adaptive_cell_grid.adaptive_cells["x"])
    y0 = np.min(adaptive_cell_grid.adaptive_cells["y"])
    z0 = np.min(adaptive_cell_grid.adaptive_cells["z"])
    xyz_0 = np.array([x0, y0, z0])
    xyz_cells = np.column_stack([adaptive_cell_grid.adaptive_cells[i] for i in "xyz"])
    dx = np.min(adaptive_cell_grid.adaptive_cells["size"])
    cell_unit = adaptive_cell_grid.final_cell_volume.unit ** (1 / 3)
    element_volume = dx**3 * cell_unit**3
    N = len(adaptive_cell_grid.adaptive_cells)
    cell_range: np.ndarray = np.arange(N, dtype=index_type)
    cells_begin = np.round((xyz_cells - xyz_0) / dx).astype(index_type)
    cells_end = np.round(
        (xyz_cells - xyz_0 + adaptive_cell_grid.adaptive_cells["size"][:, np.newaxis])
        / dx
    ).astype(index_type)
    field_spectra = calculate_field_spectrum(
        adaptive_cell_grid,
        datacube,
        spectral_model,
        element_volume,
    )
    cube_unit = field_spectra.unit
    field_spectra *= cells_end[:, 0] - cells_begin[:, 0]
    field_spectra = field_spectra[:, :, None, None].value
    cube = np.zeros(
        (
            field_spectra.shape[0],
            adaptive_cell_grid.final_grid_shape[1],
            adaptive_cell_grid.final_grid_shape[2],
        )
    )
    for i in cell_range:
        x_start, y_start, z_start = cells_begin[i]
        x_end, y_end, z_end = cells_end[i]
        cube[:, y_start:y_end, z_start:z_end] += field_spectra[:, i]
    return np.flip(np.moveaxis(cube, 1, 2), axis=2) * cube_unit
