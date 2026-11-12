"""Base Mochi only includes optically thin models, but can be extended."""

import numpy as np
from martini.datacube import DataCube
from martini.spectral_models import _BaseSpectrum
from martini.mochi.cell_grid import CellGrid
import astropy.units as U


def calculate_field_spectrum(
    cell_grid: CellGrid,
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    cell_volumes: U.Quantity[U.pix**3],
    cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled],
) -> U.Quantity[U.Msun]:
    """
    Evaluate the spectrum given the fields interpolated onto the grid.

    The spectrum is the mass contained in a series of velocity bins.

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

    Returns
    -------
    ~astropy.units.Quantity
        The evaluated mass spectrum, with shape ``(n_channels, n_cells)``.
    """
    field_temperatures = cell_grid.interpolated_fields["temperatures"]
    field_masses = cell_grid.interpolated_fields["masses_HI"]
    field_velocities = cell_grid.interpolated_fields["velocities"]
    field_temperatures[field_masses == 0] = 1 * field_temperatures.unit
    # field_masses is a density per unit corrected volume, convert to mass in each cell:
    numerator = field_masses * cell_volumes * cell_distortion_corrections
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
) -> U.Quantity[U.Msun]:
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

    Returns
    -------
    ~astropy.units.Quantity
        The mock spectral cube.
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
    )
    if adaptive:
        # Each cell's mass is shared equally between the sky-plane (y, z) pixels that it
        # covers. Cells smaller than a pixel are assigned to the pixel containing their
        # centre. Pixel edges are aligned with the corner of the initial (coarse) grid.
        yz_0 = np.array([np.min(cell_grid.cells[i]) for i in "yz"])
        yz_cells = np.column_stack([cells[i] for i in "yz"]) - yz_0
        sizes = cells["size"][:, np.newaxis]
        cells_begin = np.floor(yz_cells + 0.5 * np.minimum(sizes, 1)).astype(index_type)
        cells_end = cells_begin + np.maximum(np.round(sizes), 1).astype(index_type)
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
        for i in np.arange(len(cells), dtype=index_type):
            y_start, z_start = cells_begin[i]
            y_end, z_end = cells_end[i]
            cube[:, y_start:y_end, z_start:z_end] += field_spectra[:, i, None, None]
        cube *= cube_unit
    else:
        # field_spectra has shape (n_channels, n_cells), cells are ordered with the LoS
        # axis varying slowest, then sum along the LoS:
        cube = field_spectra.reshape((-1,) + cell_grid.grid_shape).sum(axis=1)
    return np.flip(np.moveaxis(cube, 1, 2), axis=2)
