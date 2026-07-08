"""Base Mochi only includes optically thin models, but can be extended."""

import warnings
import numpy as np
from martini.datacube import DataCube
import astropy.units as U
from typing import Callable


def calculate_field_spectrum(
    field_mass: U.Quantity[U.Msun],
    field_velocity: U.Quantity[U.km / U.s],
    field_temperature: U.Quantity[U.km**2 / U.s**2],
    cell_volume: U.Quantity[U.pix**3],
    datacube: DataCube,
) -> U.Quantity[U.Msun]:
    """
    Evaluate the spectrum given the fields interpolated onto the grid.

    The spectrum is the mass contained in a series of velocity bins.

    Parameters
    ----------
    field_mass : ~astropy.units.Quantity
        Mass interpolated onto the grid with dimensions of mass.

    field_velocity : ~astropy.units.Quantity
        Line of sight velocity interpolated onto the grid with dimensions of speed.

    field_temperature : ~astropy.units.Quantity
        Temperature (thermal velocity dispersion) interpolated onto the grid with
        dimensions of speed squared.

    cell_volume : ~astropy.units.Quantity
        Volume of a grid cell with units of pixels cubed.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    Returns
    -------
    ~astropy.units.Quantity
        The evaluated mass spectrum.
    """
    field_temperature[field_mass == 0] = 1 * field_temperature.unit
    numerator = (
        field_mass
        / np.sqrt(2 * np.pi * field_temperature)
        * datacube.channel_width
        * cell_volume
    )
    diff = field_velocity[None, ...] - datacube.velocity_channel_mids[:, None]
    field_spectrum = numerator * np.exp(-(diff**2) / (2 * field_temperature[None, ...]))
    return field_spectrum


def optically_thin(
    field_mHI: U.Quantity[U.Msun],
    field_velocity: U.Quantity[U.km / U.s],
    field_temperature: U.Quantity[U.km**2 / U.s**2],
    datacube: DataCube,
    volume_element: U.Quantity[U.pix**-3],
    volume_shape: tuple[int, int, int],
    **kwargs,
) -> U.Quantity[U.Msun]:
    """
    Assemble fields into an HI cube using optically thin approximation.

    ??

    Parameters
    ----------
    field_mHI : ~astropy.units.Quantity
        HI masses interpolated onto the grid, with dimensions of mass.

    field_velocity : ~astropy.units.Quantity
        Line of sight velocities interpolated onto the grid, with dimensions of speed.

    field_temperature : ~astropy.units.Quantity
        Temperatures (thermal velocity dispersions) interpolated onto the grid, with
        dimensions of speed squared.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    volume_element : ~astropy.units.Quantity
        Volume elements.

    volume_shape : tuple
        Spatial shape of ``field_mHI``, ``field_velocity``, ``field_temperature``.

    Returns
    -------
    ~astropy.units.Quantity
        The mock spectral cube.
    """
    field_mHI = field_mHI.reshape(volume_shape)
    field_temperature = field_temperature.reshape(volume_shape)
    field_velocity = field_velocity.reshape(volume_shape)
    field_temperature[field_mHI == 0] = 1 * field_temperature.unit
    numerator = (
        field_mHI
        / np.sqrt(2 * np.pi * field_temperature)
        * datacube.channel_width
        * volume_element
    )
    cube = (
        np.zeros((datacube.n_channels, volume_shape[1], volume_shape[2]))
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
    field_mHI: U.Quantity[U.Msun],
    field_velocity: U.Quantity[U.km / U.s],
    field_temperature: U.Quantity[U.km**2 / U.s**2],
    datacube: DataCube,
    cell_volume: U.Quantity[U.pix**3],
    volume_shape: tuple[int, int, int],
    cells: np.ndarray | None = None,
    cell_unit: U.Unit = U.dimensionless_unscaled,
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
    field_mHI : ~astropy.units.Quantity
        HI masses interpolated onto the grid, with dimensions of mass.

    field_velocity : ~astropy.units.Quantity
        Line of sight velocities interpolated onto the grid, with dimensions of speed.

    field_temperature : ~astropy.units.Quantity
        Temperatures (thermal velocity dispersions) interpolated onto the grid, with
        dimensions of speed squared.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    cell_volume : ~astropy.units.Quantity
        Volume elements.

    volume_shape : tuple
        Spatial shape of ``field_mHI``, ``field_velocity``, ``field_temperature``.

    cells : ~numpy.ndarray, optional
        The cells. Cells are encoded as the coordinates of the lower corner (accessible as
        ``cell["x"]``, etc.) and the side length (``cell["size"]``).

    cell_unit : ~astropy.units.Unit, optional
        The units for the cell dimensions (normally pixels).

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
    if cells is None:
        warnings.warn(
            "Expected argument `cells`, will attempt defaulting to "
            f"{default_renderer.__name__} instead.",
            UserWarning,
        )
        return default_renderer(
            field_mHI,
            field_velocity,
            field_temperature,
            datacube,
            cell_volume,
            volume_shape,
            **kwargs,
        )
    x0 = np.min(cells["x"])
    y0 = np.min(cells["y"])
    z0 = np.min(cells["z"])
    xyz_0 = np.array([x0, y0, z0])
    xyz_cells = np.column_stack([cells[i] for i in "xyz"])
    dx = np.min(cells["size"])
    element_volume = dx**3 * cell_unit**3
    N = len(cells)
    cell_range: np.ndarray = np.arange(N, dtype=index_type)
    cells_begin = np.round((xyz_cells - xyz_0) / dx).astype(index_type)
    cells_end = np.round(
        (xyz_cells - xyz_0 + cells["size"][:, np.newaxis]) / dx
    ).astype(index_type)
    field_spectra = calculate_field_spectrum(
        field_mHI, field_velocity, field_temperature, element_volume, datacube
    )
    cube_unit = field_spectra.unit
    field_spectra *= cells_end[:, 0] - cells_begin[:, 0]
    field_spectra = field_spectra[:, :, None, None].value
    cube = np.zeros((field_spectra.shape[0], volume_shape[1], volume_shape[2]))
    for i in cell_range:
        x_start, y_start, z_start = cells_begin[i]
        x_end, y_end, z_end = cells_end[i]
        cube[:, y_start:y_end, z_start:z_end] += field_spectra[:, i]
    return np.flip(np.moveaxis(cube, 1, 2), axis=2) * cube_unit
