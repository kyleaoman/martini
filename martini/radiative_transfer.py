"""
Base Mochi only includes optically thin models
Feel free to write your own
"""

import warnings
import numpy as np
from astropy.units import dimensionless_unscaled


def calculate_field_spectrum(
    field_mass, field_velocity, field_temperature, cell_volume, channel_size, n_channels
):
    # replace spectrum_range with actual spectral bins from datacube
    spectrum_range = channel_size * (np.arange(n_channels) - (n_channels - 1) / 2)
    field_temperature[field_mass == 0] = 1 * field_temperature.unit
    numerator = (
        field_mass / np.sqrt(2 * np.pi * field_temperature) * channel_size * cell_volume
    )
    diff = field_velocity[None, ...] - spectrum_range[:, None]
    field_spectrum = numerator * np.exp(-(diff**2) / (2 * field_temperature[None, ...]))
    return field_spectrum


def opticallyThin(
    field_mHI,
    field_velocity,
    field_temperature,
    channel_size,
    volume_element,
    volume_shape,
    *,
    n_channels=None,
    **kwargs,
):
    """
    Assemble fields into an HI cube using optically thin approximation

    Parameters
    ----------
    field_mHI:
            HI masses
    field_velocity:
            radial velocities
    field_temperature:
            velocity dispersions
    channel_size:
            size of channel in velocity units
    volume_element:
            volume elements
    volume_shape:
            spatial shape of field_mHI, field_velocity, field_temperature

    Returns
    -------
    mock cube
    """
    spectrum_range = (
        channel_size * (np.arange(n_channels) - (n_channels - 1) / 2)
    ).reshape(n_channels, 1, 1, 1)
    field_mHI = field_mHI.reshape(volume_shape)
    field_temperature = field_temperature.reshape(volume_shape)
    field_velocity = field_velocity.reshape(volume_shape)
    field_temperature[field_mHI == 0] = 1 * field_temperature.unit
    numerator = (
        field_mHI
        / np.sqrt(2 * np.pi * field_temperature)
        * channel_size
        * volume_element
    )
    cube = np.zeros((n_channels, volume_shape[1], volume_shape[2])) * numerator.unit
    spectrum_range = channel_size * (np.arange(n_channels) - (n_channels - 1) / 2)
    diff = field_velocity[None, ...] - spectrum_range[:, None, None, None]
    gaussians = np.exp(-(diff**2) / (2 * field_temperature[None, ...]))
    cube = np.sum(numerator[None, ...] * gaussians, axis=1)  # sum over LOS axis
    cube = np.flip(np.moveaxis(cube, 1, 2), axis=2)
    return cube


def adaptiveOpticallyThin(
    field_mHI,
    field_velocity,
    field_temperature,
    channel_size,
    cell_volume,
    volume_shape,
    cells=None,
    cell_unit=dimensionless_unscaled,
    *,
    index_type=np.uintc,
    default_renderer=opticallyThin,
    n_channels=None,
    **kwargs,
):
    if cells is None:
        warnings.warn(
            "cells is expected, will attempt defaulting to "
            + default_renderer.__name__,
            UserWarning,
        )
        return default_renderer(
            field_mHI,
            field_velocity,
            field_temperature,
            channel_size,
            cell_volume,
            volume_shape,
            **kwargs,
        )
    xyz0 = np.min(cells, axis=0)
    dx = xyz0[-1]
    element_volume = dx**3 * cell_unit**3
    xyz0[-1] = 0
    N = len(cells)
    cell_range = np.arange(N, dtype=index_type)
    cells_begin = np.round((cells[:, :-1] - xyz0[:-1]) / dx).astype(index_type)
    cells_end = np.round(
        (cells[:, :-1] - xyz0[:-1] + cells[:, -1][:, np.newaxis]) / dx
    ).astype(index_type)
    field_spectra = calculate_field_spectrum(
        field_mHI,
        field_velocity,
        field_temperature,
        element_volume,
        channel_size,
        n_channels,
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
