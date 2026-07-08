"""Interpolant functions to render particle properties onto a grid."""

from scipy.spatial import distance, KDTree
from sklearn.neighbors import KDTree as lKDTree
from astropy import units as U
import numpy as np
from functools import partial
from typing import Callable
from collections.abc import Iterable


def eval_kernel(
    x_eval: U.Quantity[U.pix],
    x_particle: U.Quantity[U.pix],
    h: U.Quantity[U.pix],
    kernel: Callable[
        [U.Quantity[U.dimensionless_unscaled]],
        np.ndarray,
    ],
) -> U.Quantity[U.pix**-3]:
    """
    Evaluate the kernel.

    Parameters
    ----------
    x_eval : ~astropy.units.Quantity
        Positions at which to evaluate kernel, with units of pixels.
    x_particle : ~astropy.units.Quantity
        Positions of particles for which to evaluate kernel, with units of pixels.
    h : ~astropy.units.Quantity
        Particle smoothing lengths, with units of pixels.
    kernel : Callable
        Kernel function accepting a dimensionless array argument and returning an array.

    Returns
    -------
    ~astropy.units.Quantity
        Evaluated kernel at ``x_eval`` for particles at positions ``x_particle`` with
        smoothing lengths ``h``. Expected units of ``pixels**-3``.
    """
    q = distance.cdist(x_eval / h, x_particle / h)
    return kernel(q) / (h**3)


def _eval_cache_kernel(q: float, kernel_cache: np.ndarray) -> float:
    """
    Get the value of the kernel function on a pre-computed discrete grid.

    Parameters
    ----------
    q : float
        Evaluation location where ``1`` is the radius of compact support.

    kernel_cache : np.ndarray
        The kernel evaluated on a discrete grid.

    Returns
    -------
    float
        The approximate kernel amplitude at location ``q``.
    """
    kernel_cache_resolution = len(kernel_cache)
    return kernel_cache[(np.clip(q, 0, 1) * kernel_cache_resolution).astype(np.uint8)]


def sph_loop(
    masses: np.ndarray,
    masses_HI: np.ndarray,
    momenta: np.ndarray,
    temperatures: np.ndarray,
    smoothing_lengths: np.ndarray,
    dist: np.ndarray,
    slices: np.ndarray,
    cell_volumes: np.ndarray,
    kernel_cache: np.ndarray,
    n_pos: int,
    velocity_unit: U.Unit,
    mass_unit: U.Unit,
    volume_unit: U.Unit,
    mask_out_of_bound: bool,
) -> tuple[U.Quantity[U.km / U.s], U.Quantity[U.Msun], U.Quantity[U.km**2 / U.s**2]]:
    """
    Use SPH formalism to scatter particles onto the grid.

    ??.

    Parameters
    ----------
    masses : ~numpy.ndarray
        Particle masses as an array with implicit units, the units are specified with the
        ``mass_unit`` argument.

    masses_HI : ~numpy.ndarray
        Particle HI masses as an array with implicit units, the units are specified with
        the ``mass_unit`` argument.

    momenta : ~numpy.ndarray
        Particle line-of-sight momenta as an array with implicit units, the units are
        specified with the ``mass_unit`` and ``velocity_unit`` argument.

    temperatures : ~numpy.ndarray
        Particle temperatures (thermal velocity dispersions) as an array with implicit
        units, the units are specified with the ``velocity_unit`` argument (temperature is
        ``velocity_unit**2``).

    smoothing_lengths : ~numpy.ndarray
        Particle smoothing lengths as an array with implicit units. Should have the same
        implicit units as the ``dist`` argument.

    dist : ~numpy.ndarray
        Distances of particles from kernel evaluation points as an array with implicit
        units. Should have the same implicit units as the ``dist`` argument.

    slices : ~numpy.ndarray
        ??.

    cell_volumes : ~numpy.ndarray
        ??.

    kernel_cache : ~numpy.ndarray
        Kernel amplitude pre-computed on a discrete grid for fast lookup.

    n_pos : int
        ??.

    velocity_unit : ~astropy.units.Unit
        Units for arguments with dimensions of velocity (or temperature as velocity
        squared).

    mass_unit : ~astropy.units.Unit
        Units for arguments with dimensions of mass.

    volume_unit : ~astropy.units.Unit
        Units for arguments with dimensions of volume.

    mask_out_of_bound : ~astropy.units.Unit
        ??.

    Returns
    -------
    U.Quantity[U.km / U.s]
        The velocity field evaluated on the cell grid.

    U.Quantity[U.Msun]
        The HI mass field evaluated on the cell grid.

    U.Quantity[U.km**2 / U.s**2])
        The temperature (thermal velocity dispersion) field evaluated on the cell grid.
    """
    field_masses_HI = np.zeros(n_pos)
    field_masses = np.zeros(n_pos)
    field_momenta = np.zeros(n_pos)
    field_temperatures = np.zeros(n_pos)
    h3 = smoothing_lengths**3
    n_part = len(smoothing_lengths)
    for i in range(n_part):
        if len(slices[i]) == 0:
            continue
        particleKernel = (
            _eval_cache_kernel(dist[i] / smoothing_lengths[i], kernel_cache) / h3[i]
        )
        if isinstance(mask_out_of_bound, bool):
            raise ValueError("Expected an array of booleans, got a boolean instead.")
        if not mask_out_of_bound[i]:
            # Since the particle is not out bound, we know the kernel should sum to 1.
            # The kernel not summing to 1 is due to resolution effects.
            particleKernel /= np.sum(particleKernel * cell_volumes[slices[i]])
        field_masses[slices[i]] += particleKernel * masses[i]
        field_masses_HI[slices[i]] += particleKernel * masses_HI[i]
        field_momenta[slices[i]] += particleKernel * momenta[i]
        field_temperatures[slices[i]] += particleKernel * temperatures[i]
    kernelSlice = field_masses != 0
    final_velocities = np.zeros(n_pos) * velocity_unit
    final_temperatures = np.zeros(n_pos) * velocity_unit**2
    final_masses_HI = field_masses_HI * mass_unit / volume_unit
    final_velocities[kernelSlice] = (
        field_momenta[kernelSlice] * velocity_unit / field_masses[kernelSlice]
    )
    final_temperatures[kernelSlice] = (
        field_temperatures[kernelSlice] * velocity_unit**2 / field_masses[kernelSlice]
    )
    return final_velocities, final_masses_HI, final_temperatures


def mfm_loop(
    masses: np.ndarray,
    masses_HI: np.ndarray,
    momenta: np.ndarray,
    temperatures: np.ndarray,
    smoothing_lengths: np.ndarray,
    dist: np.ndarray,
    slices: np.ndarray,
    cell_volumes: np.ndarray,
    kernel_cache: np.ndarray,
    n_pos: int,
    velocity_unit: U.Unit,
    mass_unit: U.Unit,
    volume_unit: U.Unit,
    mask_out_of_bound: bool,
) -> tuple[U.Quantity[U.km / U.s], U.Quantity[U.Msun], U.Quantity[U.km**2 / U.s**2]]:
    """
    Use MFM formalism to scatter particles onto the grid.

    ??.

    Parameters
    ----------
    masses : ~numpy.ndarray
        Particle masses as an array with implicit units, the units are specified with the
        ``mass_unit`` argument.

    masses_HI : ~numpy.ndarray
        Particle HI masses as an array with implicit units, the units are specified with
        the ``mass_unit`` argument.

    momenta : ~numpy.ndarray
        Particle line-of-sight momenta as an array with implicit units, the units are
        specified with the ``mass_unit`` and ``velocity_unit`` arguments.

    temperatures : ~numpy.ndarray
        Particle temperatures (thermal velocity dispersions) as an array with implicit
        units, the units are specified with the ``velocity_unit`` argument (temperature is
        ``velocity_unit**2``).

    smoothing_lengths : ~numpy.ndarray
        Particle smoothing lengths as an array with implicit units. Should have the same
        implicit units as the ``dist`` argument.

    dist : ~numpy.ndarray
        Distances of particles from kernel evaluation points as an array with implicit
        units. Should have the same implicit units as the ``dist`` argument.

    slices : ~numpy.ndarray
        ??.

    cell_volumes : ~numpy.ndarray
        ??.

    kernel_cache : ~numpy.ndarray
        Kernel amplitude pre-computed on a discrete grid for fast lookup.

    n_pos : int
        ??.

    velocity_unit : ~astropy.units.Unit
        Units for arguments with dimensions of velocity (or temperature as velocity
        squared).

    mass_unit : ~astropy.units.Unit
        Units for arguments with dimensions of mass.

    volume_unit : ~astropy.units.Unit
        Units for arguments with dimensions of volume.

    mask_out_of_bound : ~astropy.units.Unit
        ??.

    Returns
    -------
    U.Quantity[U.km / U.s]
        The velocity field evaluated on the cell grid.

    U.Quantity[U.Msun]
        The HI mass field evaluated on the cell grid.

    U.Quantity[U.km**2 / U.s**2])
        The temperature (thermal velocity dispersion) field evaluated on the cell grid.
    """
    field_masses_HI = np.zeros(n_pos)
    field_masses = np.zeros(n_pos)
    field_momenta = np.zeros(n_pos)
    field_temperatures = np.zeros(n_pos)
    h3 = smoothing_lengths**3
    n_part = len(smoothing_lengths)
    total_kernel = np.zeros(n_pos)
    for i in range(n_part):
        if len(slices[i]) == 0:
            continue
        particle_kernel = (
            _eval_cache_kernel(dist[i] / smoothing_lengths[i], kernel_cache) / h3[i]
        )
        total_kernel[slices[i]] += particle_kernel
        slices[i] = slices[i][particle_kernel != 0]
        dist[i] = dist[i][particle_kernel != 0]
    field_masses_HI = np.zeros(n_pos)
    field_masses = np.zeros(n_pos)
    field_momenta = np.zeros(n_pos)
    field_temperatures = np.zeros(n_pos)
    for i in range(n_part):
        if len(slices[i]) == 0:
            continue
        particle_kernel = (
            _eval_cache_kernel(dist[i] / smoothing_lengths[i], kernel_cache) / h3[i]
        )
        volume = np.sum(
            particle_kernel * (cell_volumes[slices[i]] / total_kernel[slices[i]])
        )
        if isinstance(mask_out_of_bound, bool):
            raise ValueError("Expected an array of booleans, got a boolean instead.")
        if mask_out_of_bound[i]:
            volume *= (
                np.pi
                * 4
                / 3
                * smoothing_lengths[i] ** 3
                / np.sum(cell_volumes[slices[i]])
            )  # for out of bounds particles, the volume is scaled up
        field_masses_HI[slices[i]] += particle_kernel * masses_HI[i] / volume
        field_masses[slices[i]] += particle_kernel * masses[i] / volume
        field_momenta[slices[i]] += particle_kernel * momenta[i] / volume
        field_temperatures[slices[i]] += particle_kernel * temperatures[i] / volume
    kernel_slice = total_kernel != 0
    final_velocities = np.zeros(n_pos) * velocity_unit
    final_temperatures = np.zeros(n_pos) * velocity_unit**2
    final_masses_HI = np.zeros(n_pos) * mass_unit / volume_unit
    final_masses = np.zeros(n_pos)
    final_masses_HI[kernel_slice] = (
        field_masses_HI[kernel_slice]
        * mass_unit
        / volume_unit
        / total_kernel[kernel_slice]
    )
    final_masses[kernel_slice] = field_masses[kernel_slice] / total_kernel[kernel_slice]
    final_velocities[kernel_slice] = (
        field_momenta[kernel_slice]
        * velocity_unit
        / total_kernel[kernel_slice]
        / final_masses[kernel_slice]
    )
    final_temperatures[kernel_slice] = (
        field_temperatures[kernel_slice]
        * velocity_unit**2
        / total_kernel[kernel_slice]
        / final_masses[kernel_slice]
    )
    return final_velocities, final_masses_HI, final_temperatures


def _getOutOfBoundParticles(
    particle_positions: np.ndarray,
    particle_radii: np.ndarray,
    field_positions: np.ndarray,
) -> np.ndarray:
    """
    Find particles that fall outside of the region where fields are being evaluated.

    Parameters
    ----------
    particle_positions : ~numpy.ndarray
        Array of particle positions.

    particle_radii : ~numpy.ndarray
        Array of particle sizes (radii of compact support).

    field_positions : ~numpy.ndarray
        Array of locations where the fields are being evaluated.

    Returns
    -------
    ~numpy.ndarray
        Array containing booleans, ``True`` for particles that are outside the region.
    """
    # Martini has a pre-emptive particle masking function. Move this functionality there?
    lowBound = np.min(field_positions, axis=0)
    topBound = np.max(field_positions, axis=0)
    mask_out_of_bound = (
        (particle_positions + particle_radii[:, np.newaxis]) > topBound
    ) | ((particle_positions - particle_radii[:, np.newaxis]) < lowBound)
    mask_out_of_bound = np.any(mask_out_of_bound, axis=1)
    return mask_out_of_bound


def particle_scatter(
    main_loop: Callable,  # fill in arg & return types
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: Callable,  # fill in arg & return types
    field_positions: U.Quantity[U.pix],
    d_volume: U.Quantity[U.pix],
    *,
    kernel_cache_resolution: int = 256,
) -> tuple[U.Quantity[U.km / U.s], U.Quantity[U.Msun], U.Quantity[U.km**2 / U.s**2]]:
    """
    Scatter particles onto the cell grid. Can use SPH, MFM or other backends.

    ??.

    Parameters
    ----------
    main_loop : Callable
        The function handling the scatter operation, e.g. ``sph_loop`` or ``mfm_loop``.

    positions : ~astropy.units.Quantity
        Particle positions with units of pixels.

    velocities : ~astropy.units.Quantity
        Particle radial velocities with dimensions of speed.

    smoothing_lengths : ~astropy.units.Quantity
        Particle smoothing lengths with units of pixels.

    masses_HI : ~astropy.units.Quantity
        Particle HI masses with dimensions of mass.

    temperatures : ~astropy.units.Quantity
        Particle temperatures (thermal velocity dispersions) with dimensions of speed
        squared.

    masses : ~astropy.units.Quantity
        Particle masses with dimensions of mass.

    kernel : Callable
        Kernel function.

    field_positions : ~numpy.ndarray
        Positions at which to interpolate fields, implicitly with units of pixels.

    d_volume : ~astropy.units.Quantity
        Volume element size for ``field_positions`` with units of pixels.

    kernel_cache_resolution : int
        Number of grid points on which to sample the kernel for fast lookup.

    Returns
    -------
    U.Quantity[U.km / U.s]
        The velocity field evaluated on the cell grid.

    U.Quantity[U.Msun]
        The HI mass field evaluated on the cell grid.

    U.Quantity[U.km**2 / U.s**2])
        The temperature (thermal velocity dispersion) field evaluated on the cell grid.
    """
    kernel_cache = kernel(np.linspace(0, 1, kernel_cache_resolution))
    mask_out_of_bound = _getOutOfBoundParticles(
        positions, smoothing_lengths, field_positions
    )
    masses *= U.dimensionless_unscaled
    if velocities.ndim != 1:
        # more than one dimension of velocity is given, use radial velocity
        velocities = velocities[:, 0]
    n_pos = len(field_positions)
    if not isinstance(d_volume, Iterable):
        d_volume = np.ones(n_pos) * d_volume
    slices, dist = lKDTree(field_positions.value).query_radius(
        positions.value, smoothing_lengths.value, return_distance=True
    )
    momenta = velocities.value * masses.value
    thermal = temperatures.to_value(velocities.unit**2) * masses.value
    return main_loop(
        masses.value,
        masses_HI.value,
        momenta,
        thermal,
        smoothing_lengths.value,
        dist,
        slices,
        d_volume.value,
        kernel_cache,
        n_pos,
        velocities.unit,
        masses_HI.unit,
        smoothing_lengths.unit**3,
        mask_out_of_bound,
    )


sph = partial(particle_scatter, sph_loop)
mfm = partial(particle_scatter, mfm_loop)


def _eval_voronoi_field(
    particle_quantities: U.Quantity,
    nearest_particle_indices: np.ndarray | np.int32 | np.int64,
    missed_particle_cell_indices: np.ndarray | np.int32 | np.int64,
    missed_particle_mask: np.ndarray,
    field_n_particle: np.ndarray,
) -> U.Quantity:
    """
    Evaluate the field at grid points from the particles nearest to them.

    This is a Voronoi tesselation-based interpolation. The arguments use "particle" but
    these refer to Voronoi cells, this is to avoid ambiguity with the grid cells that are
    being interpolated onto.

    Parameters
    ----------
    particle_quantities : ~astropy.units.Quantity
        The values of the field as carried by the Voronoi cells.

    nearest_particle_indices : ~numpy.ndarray or int
        The index of the Voronoi cell enclosing each grid cell.

    missed_particle_cell_indices : ~numpy.ndarray or int
        ??.

    missed_particle_mask : ~numpy.ndarray
        ??.

    field_n_particle : ~numpy.ndarray
        ??.

    Returns
    -------
    ~astropy.units.Quantity
        Field interpolated onto the cell grid.
    """
    field_quantity = particle_quantities[nearest_particle_indices]
    field_quantity[missed_particle_cell_indices] += particle_quantities[
        missed_particle_mask
    ]
    field_quantity /= field_n_particle
    return field_quantity


def voronoi_mesh(
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: Callable,  # fill in arg & return types
    field_positions: U.Quantity[U.pix],
    d_volume: U.Quantity[U.pix],
    **kwargs: int,
) -> tuple[U.Quantity[U.km / U.s], U.Quantity[U.Msun], U.Quantity[U.km**2 / U.s**2]]:
    """
    Compute the interpolated fields using a Voronoi mesh.

    Assumes that ``field_positions`` creates a box.

    ??.

    Parameters
    ----------
    positions : ~astropy.units.Quantity
        Voronoi cell positions with units of pixels.

    velocities : ~astropy.units.Quantity
        Voronoi cell radial velocities with dimensions of speed.

    smoothing_lengths : ~astropy.units.Quantity
        Unused.

    masses_HI : ~astropy.units.Quantity
        Voronoi cell HI masses with dimensions of mass.

    temperatures : ~astropy.units.Quantity
        Voronoi cell temperatures (thermal velocity dispersions) with dimensions of speed
        squared.

    masses : ~astropy.units.Quantity
        Unused.

    kernel : Callable
        Unused.

    field_positions : ~numpy.ndarray
        Positions at which to interpolate fields, implicitly with units of pixels.

    d_volume : ~astropy.units.Quantity
        Volume element size for ``field_positions`` with units of pixels.

    Returns
    -------
    ~astropy.units.Quantity
        Interpolated velocity field with dimensions of speed.

    ~astropy.units.Quantity
        Interpolated HI mass field with dimensions of mass.

    ~astropy.units.Quantity
        Interpolated thermal velocity dispersion field with dimensions of speed squared.
    """
    masses *= U.dimensionless_unscaled
    if velocities.ndim != 1:
        # more than one dimension of velocity is given, use radial velocity
        velocities = velocities[:, 0]
    particle_indices = np.arange(len(positions))
    _, nearest_particle_indices = KDTree(positions).query(
        field_positions
    )  # nearest neighbor assignment of particles to field pos

    # construct a mask for inbound particles but not assigned to a cell
    inbound_particle_mask = np.all(
        positions > field_positions.min(axis=0), axis=1
    ) & np.all(
        positions < field_positions.max(axis=0), axis=1
    )  # assume box shape for field pos
    used_particle_mask = np.isin(particle_indices, nearest_particle_indices)
    missed_particle_mask = inbound_particle_mask & ~used_particle_mask
    missed_particle_indices = particle_indices[missed_particle_mask]
    _, missed_particle_cell_indices = KDTree(field_positions).query(
        positions[missed_particle_mask]
    )

    particle_masks = nearest_particle_indices == particle_indices[:, np.newaxis]
    particle_masks[missed_particle_indices, missed_particle_cell_indices] = True

    field_n_particle = np.ones(len(field_positions), dtype=np.uint64)
    field_n_particle[missed_particle_cell_indices] += 1

    particle_volumes = np.einsum(
        "ij,j->i", particle_masks, d_volume / field_n_particle
    )  # for shared cells, the volume is divided between the particles
    densities = np.zeros(masses_HI.shape) * masses_HI.unit / particle_volumes.unit
    volume_mask = ~(particle_volumes == 0)
    densities[volume_mask] = masses_HI[volume_mask] / particle_volumes[volume_mask]
    field_velocities = _eval_voronoi_field(
        velocities,
        nearest_particle_indices,
        missed_particle_cell_indices,
        missed_particle_mask,
        field_n_particle,
    )
    field_masses_HI = _eval_voronoi_field(
        densities,
        nearest_particle_indices,
        missed_particle_cell_indices,
        missed_particle_mask,
        field_n_particle,
    )
    field_temperatures = _eval_voronoi_field(
        temperatures,
        nearest_particle_indices,
        missed_particle_cell_indices,
        missed_particle_mask,
        field_n_particle,
    )
    return field_velocities, field_masses_HI, field_temperatures


def manual_sph(
    positions: U.Quantity[U.pix],
    velocities: U.Quantity[U.km / U.s],
    smoothing_lengths: U.Quantity[U.pix],
    masses_HI: U.Quantity[U.Msun],
    temperatures: U.Quantity[U.km**2 / U.s**2],
    masses: U.Quantity[U.Msun],
    kernel: Callable,  # fill in arg & return types
    field_positions: U.Quantity[U.pix],
    d_volume: U.Quantity[U.pix],
    **kwargs: int,
) -> tuple[U.Quantity[U.km / U.s], U.Quantity[U.Msun], U.Quantity[U.km**2 / U.s**2]]:
    """
    Compute the interpolated fields using SPH interpolation.

    Different SPH schemes have different definitions for velocity interpolation. This
    interpolant assumes that the conserved quantities are interpolated. This SPH
    interpolant serves for testing purposes and writes the equations out explicitely.
    Consequently, it is slow but safe.

    Parameters
    ----------
    positions : ~astropy.units.Quantity
        Particle positions with units of pixels.

    velocities : ~astropy.units.Quantity
        Particle radial velocities with dimensions of speed.

    smoothing_lengths : ~astropy.units.Quantity
        Particle smoothing lengths with units of pixels.

    masses_HI : ~astropy.units.Quantity
        Particle HI masses with dimensions of mass.

    temperatures : ~astropy.units.Quantity
        Particle temperatures (thermal velocity dispersions) with dimensions of speed
        squared.

    masses : ~astropy.units.Quantity
        Particle masses with dimensions of mass.

    kernel : Callable
        Kernel function.

    field_positions : ~numpy.ndarray
        Positions at which to interpolate fields, implicitly with units of pixels.

    d_volume : ~astropy.units.Quantity
        Volume element size for ``field_positions`` with units of pixels.

    Returns
    -------
    ~astropy.units.Quantity
        Interpolated velocity field with dimensions of speed.

    ~astropy.units.Quantity
        Interpolated HI mass field with dimensions of mass.

    ~astropy.units.Quantity
        Interpolated temperature (thermal velocity dispersion) field with dimensions of
        speed squared.
    """
    masses *= U.dimensionless_unscaled
    n_part, n_dim = positions.shape
    if velocities.ndim != 1:
        # more than one dimension of velocity is given, use radial velocity
        velocities = velocities[:, 0]
    n_pos = len(field_positions)
    if not isinstance(d_volume, Iterable):
        d_volume = np.ones(n_pos) * d_volume
    slices = KDTree(field_positions).query_ball_point(positions, smoothing_lengths)
    field_masses_HI = np.zeros(n_pos) * masses_HI.unit / d_volume.unit
    field_masses = np.zeros(n_pos) * masses.unit / d_volume.unit
    field_velocities = np.zeros(n_pos) * velocities.unit * masses.unit / d_volume.unit
    field_temperatures = (
        np.zeros(n_pos) * velocities.unit**2 * masses.unit / d_volume.unit
    )
    for i in range(n_part):
        particle_kernel = eval_kernel(
            field_positions[slices[i]],
            positions[i].reshape((1, n_dim)),
            smoothing_lengths[i],
            kernel,
        )[:, 0]
        field_masses[slices[i]] += particle_kernel * masses[i]
        field_masses_HI[slices[i]] += particle_kernel * masses_HI[i]
        field_velocities[slices[i]] += (
            particle_kernel * velocities[i] * masses[i]
        )  # quantity of movement is conserved
        field_temperatures[slices[i]] += (
            particle_kernel * temperatures[i] * masses[i]
        )  # thermal energy is conserved
    del slices
    kernel_slice = field_masses != 0
    final_velocities = np.zeros(n_pos) * velocities.unit
    final_temperatures = np.zeros(n_pos) * velocities.unit**2
    final_velocities[kernel_slice] = (
        field_velocities[kernel_slice] / field_masses[kernel_slice]
    )
    final_temperatures[kernel_slice] = (
        field_temperatures[kernel_slice] / field_masses[kernel_slice]
    )
    return final_velocities, field_masses_HI, final_temperatures
