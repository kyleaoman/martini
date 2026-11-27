"""Base Mochi only includes optically thin models, but can be extended."""

import warnings
import numpy as np
from typing import TYPE_CHECKING
from martini._util import NUMBA_AVAILABLE, _plan_segments, resolve_ncpu
from martini._grid_search import _balanced_blocks, _threads
from martini.datacube import DataCube
from martini.spectral_models import _BaseSpectrum
from martini.mochi.cell_grid import CellGrid
import astropy.units as U
from tqdm.auto import tqdm

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


def _insert_cell_spectra_loop(
    cube: np.ndarray,
    spectra: np.ndarray,
    weights: np.ndarray,
    intersections: np.ndarray,
    cell_indices: np.ndarray,
    strides: np.ndarray,
    block_edges: np.ndarray,
) -> None:
    """
    Add the weighted spectra of cells into the pixels that they cover.

    Similar to :func:`~martini.martini._insert_weighted_spectra_loop`, but the weight
    of each cell is the same in every pixel that it covers (instead of kernel weights).
    Compiled with :mod:`numba` if it is available. Each pixel is visited by one thread
    only, so the writes are thread safe, and the cells covering a pixel are added in the
    order of ``intersections``.

    Parameters
    ----------
    cube : ~numpy.ndarray
        The cube array, shape ``(ny, nz, n_channels)``, modified in place.

    spectra : ~numpy.ndarray
        The spectra of the cells, shape ``(n_cells, n_channels)``.

    weights : ~numpy.ndarray
        The weight of each cell.

    intersections : ~numpy.ndarray
        Cell-pixel overlaps grouped by pixel, see :func:`_cell_pixel_overlaps`.

    cell_indices : ~numpy.ndarray
        The (flattened) index of each pixel with overlaps.

    strides : ~numpy.ndarray
        The range of ``intersections`` for each pixel.

    block_edges : ~numpy.ndarray
        Pixels are processed in parallel in blocks ``block_edges[b]:block_edges[b + 1]``
        (indices into ``cell_indices``).
    """
    nz = cube.shape[1]
    n_channels = cube.shape[2]
    for b in prange(block_edges.size - 1):
        for c in range(block_edges[b], block_edges[b + 1]):
            i = cell_indices[c] // nz
            j = cell_indices[c] - i * nz
            for m in range(strides[c, 0], strides[c, 1]):
                p = intersections[m]
                w = weights[p]
                for k in range(n_channels):
                    cube[i, j, k] += w * spectra[p, k]


if NUMBA_AVAILABLE:
    _insert_cell_spectra_loop = njit(parallel=True, cache=True)(
        _insert_cell_spectra_loop
    )


def _insert_cell_spectra(
    cube: np.ndarray,
    spectra: np.ndarray,
    weights: np.ndarray,
    intersections: np.ndarray,
    cell_indices: np.ndarray,
    strides: np.ndarray,
    progressbar: bool = False,
) -> None:
    """
    Add the weighted spectra of cells into the pixels that they cover, with numpy.

    A fallback for :func:`_insert_cell_spectra_loop` when compiled code is unavailable or
    disabled: a serial loop over pixels, summing the weighted spectra of the cells
    covering each pixel.

    Parameters
    ----------
    cube : ~numpy.ndarray
        The cube array, shape ``(ny, nz, n_channels)``, modified in place.

    spectra : ~numpy.ndarray
        The spectra of the cells, shape ``(n_cells, n_channels)``.

    weights : ~numpy.ndarray
        The weight of each cell.

    intersections : ~numpy.ndarray
        Cell-pixel overlaps grouped by pixel, see :func:`_cell_pixel_overlaps`.

    cell_indices : ~numpy.ndarray
        The (flattened) index of each pixel with overlaps.

    strides : ~numpy.ndarray
        The range of ``intersections`` for each pixel.

    progressbar : bool, optional
        If ``True``, show a progress bar.
    """
    nz = cube.shape[1]
    for c in tqdm(range(cell_indices.size), disable=not progressbar):
        i, j = divmod(cell_indices[c], nz)
        pairs = intersections[strides[c, 0] : strides[c, 1]]
        cube[i, j] += weights[pairs] @ spectra[pairs]


def _cell_pixel_overlaps(
    footprint_begin: np.ndarray,
    footprint_end: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the pixels covered by each cell, grouped by pixel.

    Pixels outside of the image are left out. The cells covering each pixel are listed in
    ascending order.

    Parameters
    ----------
    footprint_begin : ~numpy.ndarray
        First pixel covered by each cell along the two axes of the image, shape
        ``(n_cells, 2)``.

    footprint_end : ~numpy.ndarray
        One past the last pixel covered by each cell along the two axes of the image,
        shape ``(n_cells, 2)``.

    image_shape : tuple
        Number of pixels along the two axes of the image.

    Returns
    -------
    intersections : ~numpy.ndarray
        Index of the cell of each cell-pixel overlap, grouped by pixel.

    cell_indices : ~numpy.ndarray
        The (flattened, C-order) index of each pixel with overlaps.

    strides : ~numpy.ndarray
        The range of ``intersections`` for each pixel, shape ``(n_pixels, 2)``.
    """
    begin = np.asarray(footprint_begin, dtype=np.int64)
    extent = np.asarray(footprint_end, dtype=np.int64) - begin
    counts = extent[:, 0] * extent[:, 1]
    cell = np.repeat(np.arange(counts.size), counts)
    offset = np.arange(cell.size) - np.repeat(np.cumsum(counts) - counts, counts)
    y = begin[cell, 0] + offset // extent[cell, 1]
    z = begin[cell, 1] + offset % extent[cell, 1]
    in_image = (y >= 0) & (y < image_shape[0]) & (z >= 0) & (z < image_shape[1])
    cell, flat = cell[in_image], (y * image_shape[1] + z)[in_image]
    order = np.argsort(flat, kind="stable")
    cell_indices, starts, hits = np.unique(
        flat[order], return_index=True, return_counts=True
    )
    strides = np.column_stack((starts, starts + hits))
    return cell[order], cell_indices, strides


def _radiative_transfer_mem_estimate(
    footprints: np.ndarray, n_channels: int, spec_itemsize: int, n_px: int
) -> tuple[np.ndarray, float, float]:
    """
    Estimate the memory needed to add the spectra of cells into the cube.

    Estimates memory (bytes), with ``s`` the size of an element of the spectra:

    - Per cell, the spectrum and temporary copies while evaluating it
      (``n_channels * (2 * s + 4)``), and per cell-pixel overlap the temporary arrays
      used to group the overlaps by pixel (64B).
    - Per batch of cells, a fixed overhead of up to 40B per pixel (pixels with overlaps
      and the corresponding ranges of overlaps).
    - Regardless of batching, the cube (``8 * n_channels`` per pixel) and the footprint
      and weight of every cell (64B).

    Parameters
    ----------
    footprints : ~numpy.ndarray
        Number of pixels covered by each cell.

    n_channels : int
        Number of spectral channels.

    spec_itemsize : int
        Size of an element of the spectra, in bytes.

    n_px : int
        Number of pixels in the image.

    Returns
    -------
    tuple
        The memory needed for each cell in its batch, the fixed memory needed for each
        batch, and the memory needed throughout regardless of batching, in GB.
    """
    mem_estimate = (
        n_channels * (2 * spec_itemsize + 4) + 64 * footprints
    ) / 1024**3  # GB
    segment_overhead_mem_estimate = 40 * n_px / 1024**3  # GB
    baseline_mem_estimate = (8 * n_channels * n_px + 64 * footprints.size) / 1024**3
    return mem_estimate, segment_overhead_mem_estimate, baseline_mem_estimate


def _plan_radiative_transfer_segments(
    footprints: np.ndarray,
    n_channels: int,
    spec_itemsize: int,
    n_px: int,
    mem_allowance_GB: float,
) -> list[slice]:
    """
    Divide the cells into batches to add their spectra into the cube within an allowance.

    Parameters
    ----------
    footprints : ~numpy.ndarray
        Number of pixels covered by each cell.

    n_channels : int
        Number of spectral channels.

    spec_itemsize : int
        Size of an element of the spectra, in bytes.

    n_px : int
        Number of pixels in the image.

    mem_allowance_GB : float
        Memory available to add the spectra of the cells into the cube, in GB.

    Returns
    -------
    list
        :class:`slice` objects (with explicit ``start`` and ``stop``) selecting the
        batches of cells.

    Raises
    ------
    RuntimeError
        If the allowance is insufficient, even when processing cells in batches.
    """
    mem_estimate, segment_overhead, baseline = _radiative_transfer_mem_estimate(
        footprints, n_channels, spec_itemsize, n_px
    )
    advice = (
        "Increase `mem_lim_GB` (the allowance is what remains of it after the estimated "
        "baseline memory, e.g. to store particle data)."
    )
    single_batch_mem_estimate = np.sum(mem_estimate) + segment_overhead + baseline
    if single_batch_mem_estimate <= mem_allowance_GB:
        return [slice(0, footprints.size)]
    warnings.warn(
        f"Memory allowance for the radiative transfer ({mem_allowance_GB:.2f} GB) is "
        "insufficient to process all cells at once (estimated memory required: "
        f"{single_batch_mem_estimate:.2f} GB). Processing in batches instead, this could "
        f"take longer. {advice}",
        RuntimeWarning,
    )
    insufficient = RuntimeError(
        f"Memory allowance for the radiative transfer ({mem_allowance_GB:.2f} GB) is "
        "insufficient, even when cells are processed in batches (estimated memory "
        f"required for the cube and cells: {segment_overhead + baseline:.2f} GB). "
        f"{advice}"
    )
    batch_allowance = mem_allowance_GB - segment_overhead - baseline
    if batch_allowance <= 0:
        raise insufficient
    try:
        return _plan_segments(mem_estimate, batch_allowance)
    except ValueError as e:
        raise insufficient from e


def calculate_field_spectrum(
    cell_grid: CellGrid,
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    cell_volumes: U.Quantity[U.pix**3],
    cell_distortion_corrections: U.Quantity[U.dimensionless_unscaled],
    cell_distances: U.Quantity[U.Mpc],
    mask: slice | np.ndarray = slice(None),
    ncpu: int = 1,
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

    mask : slice or ~numpy.ndarray, optional
        Evaluate the spectra only for a subset of cells, enables processing cells in
        batches.

    ncpu : int, optional
        Number of threads used by the spectral model.

    Returns
    -------
    ~astropy.units.Quantity
        The evaluated mass spectrum divided by distance squared, with shape
        ``(n_channels, n_cells)`` (the cells selected by ``mask``).
    """
    field_temperatures = cell_grid.interpolated_fields["temperatures"][mask]
    field_masses = cell_grid.interpolated_fields["masses_HI"][mask]
    field_velocities = cell_grid.interpolated_fields["velocities"][mask]
    field_temperatures[field_masses == 0] = 1 * field_temperatures.unit
    # field_masses is a density per unit corrected volume, convert to mass in each cell,
    # then divide by distance squared since flux scales with each cell's own distance:
    numerator = (
        field_masses
        * cell_volumes[mask]
        * cell_distortion_corrections[mask]
        / cell_distances[mask].to(U.Mpc) ** 2
    )
    # Some refactoring needed here for better integration with spectral model, or maybe
    # some changes needed in spectral model code.
    # To consider:
    #  - "extra_data" is just explicitly given, but we can save even touching the
    #    temperatures at all in the DiracDeltaSpectrum case.
    #  - Casting to Quantity is not ideal, could use SpectralCoord for the interpolated
    #    velocity field (or can it at least be a view?).
    #  - The spectral_model._BaseSpectrum assumes that we're dealing with particles stored
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
        ncpu=ncpu,
    ).T
    return numerator * field_spectrum


def optically_thin(
    cell_grid: CellGrid,
    datacube: DataCube,
    spectral_model: _BaseSpectrum,
    ncpu: int = 1,
    mem_allowance_GB: float = np.inf,
    compiled: bool = NUMBA_AVAILABLE,
    progressbar: bool = False,
) -> U.Quantity[U.Msun / U.Mpc**2]:
    """
    Assemble fields into an HI cube using optically thin approximation.

    The spectrum of each cell is shared equally between the pixels that it covers. The
    spectra are evaluated for batches of cells if the estimated memory needed exceeds
    ``mem_allowance_GB``.

    Parameters
    ----------
    cell_grid : ~martini.mochi.cell_grid.CellGrid
        The cell grid with interpolated fields available.

    datacube : ~martini.datacube.DataCube
        Target datacube object, used to retrieve spectral channels.

    spectral_model : _BaseSpectrum
        The spectral model used to evaluate the spectra.

    ncpu : int, optional
        Number of threads to use (if :mod:`numba` is available) to evaluate the spectra
        and add them to the pixels that the cells cover. A value of ``-1`` means as many
        threads as available cores.

    mem_allowance_GB : float, optional
        Memory available to evaluate the spectra and add them into the cube, in GB. This
        is a "soft" limit.

    compiled : bool, optional
        If ``True`` (the default if :mod:`numba` is available), the spectra are added to
        the pixels in compiled code. Otherwise, a serial loop over pixels sums them.

    progressbar : bool, optional
        If ``True``, show a progress bar of the serial loop over pixels (no progress bar
        is shown for compiled code).

    Returns
    -------
    ~astropy.units.Quantity
        The mock spectral cube, of HI mass divided by distance squared in each pixel and
        channel.
    """
    adaptive = hasattr(cell_grid, "adaptive_cells")
    cells = cell_grid.adaptive_cells if adaptive else cell_grid.cells
    yz_cells = np.column_stack([cells[i] for i in "yz"]) + 0.5
    sizes = cells["size"][:, np.newaxis]
    cells_begin = np.floor(yz_cells).astype(np.int64)
    cells_end = np.ceil(yz_cells + sizes).astype(np.int64)
    if np.any(cells_end - cells_begin != np.maximum(sizes, 1)):
        raise RuntimeError(
            "Cell grid is not aligned with pixel grid, cells must have power-of-two "
            "sizes with edges on pixel edges."
        )
    footprints = np.prod(cells_end - cells_begin, axis=1)
    # The cell grid may extend beyond the target region to obey constraints on the cell
    # side length and refinement requirements. Mass in the parts of cells beyond the
    # target region along the LoS is excluded here, assuming that the density is uniform
    # within each cell.
    region_lower_bound, region_upper_bound = cell_grid.los_region
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
    contributing = np.flatnonzero(weights != 0)
    image_shape = (datacube.datacube_shape[0], datacube.datacube_shape[1])
    n_channels = datacube.velocity_channel_edges.size - 1
    ncpu = resolve_ncpu(ncpu)
    field_spectrum_args = (
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
    # the unit of the cube, as evaluated by calculate_field_spectrum:
    cube_unit = (
        cell_grid.interpolated_fields["masses_HI"].unit
        * field_spectrum_args[3].unit
        * field_spectrum_args[4].unit
        / U.Mpc**2
    )
    cube = np.zeros(image_shape + (n_channels,))
    segments = (
        _plan_radiative_transfer_segments(
            footprints[contributing],
            n_channels,
            np.dtype(spectral_model.spec_dtype).itemsize,
            image_shape[0] * image_shape[1],
            mem_allowance_GB,
        )
        if contributing.size > 0
        else []
    )
    for segment in segments:
        batch = contributing[segment]
        field_spectra = calculate_field_spectrum(
            *field_spectrum_args, mask=batch, ncpu=ncpu
        )
        # one row per cell, shared equally between the pixels that the cell covers:
        spectra = np.ascontiguousarray(
            (field_spectra / footprints[batch]).to_value(cube_unit).T
        )
        del field_spectra
        intersections, cell_indices, strides = _cell_pixel_overlaps(
            cells_begin[batch], cells_end[batch], image_shape
        )
        if compiled:
            with _threads(ncpu):
                _insert_cell_spectra_loop(
                    cube,
                    spectra,
                    weights[batch],
                    intersections,
                    cell_indices,
                    strides,
                    _balanced_blocks(strides[:, 1] - strides[:, 0], 8 * ncpu),
                )
        else:
            _insert_cell_spectra(
                cube,
                spectra,
                weights[batch],
                intersections,
                cell_indices,
                strides,
                progressbar=progressbar,
            )
        # free this batch before evaluating the next:
        del spectra, intersections, cell_indices, strides
    return np.flip(np.moveaxis(cube, (0, 1, 2), (2, 1, 0)), axis=2) * cube_unit
