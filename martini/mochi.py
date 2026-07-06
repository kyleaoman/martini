"""
Create mock observations of HI sources from cosmological hydrodynamical simulations.

Provides :class:`~martini.martini.Martini`, the main class of the package, and a
simplified :class:`~martini.martini.GlobalProfile` class for use when only a spectrum (no
spatial information) is desired.
"""

import numpy as np
from astropy import units as U, constants as C
from martini.martini import Martini
from martini.datacube import DataCube
from martini.beams import _BaseBeam
from martini.sources import SPHSource
from martini.sph_kernels import _BaseSPHKernel
from martini.spectral_models import _BaseSpectrum
from martini.noise import _BaseNoise
from typing import Callable
from functools import partial
from . import interpolants, radiative_transfer


def _refine_grid_bisect(
    size, x, y, z, mask, in_cell, new_cells, new_cells_over, new_cells_masks
):
    """
    Bisect operation for refine grid algorithms
    """
    new_size = size / 2.0
    new_cells.extend(
        [
            (x + dx * new_size, y + dy * new_size, z + dz * new_size, new_size)
            for dx in range(2)
            for dy in range(2)
            for dz in range(2)
        ]
    )
    new_cells_over.extend([False] * 8)
    new_cells_masks.extend([mask[in_cell]] * 8)


def _pass_complete_cell(cells_lists, content_list):
    for i in range(len(cells_lists)):
        cells_lists[i].append(content_list[i])


def refine_grid(
    in_cell_function,
    bisect_condition,
    cells,
    positions,
    radii,
    threshold: float,
    stop_iter: int = 8,
):
    """
    Starting from a coarse grid, refine until no cell satisfy bisectCondition.

    Parameters
    ----------
    incellFunction: function
            delimits the particles to consider for the cell
    bisectCondition: function
            if returns true, the cell is bisected
    cells: list
            list of [x,y,z,h] where x,y,z is the 3D position of the low corner and h is the size of the cell
    positions: array N x 3
            array of particle positions
    radii: array N
            array of particle radii
    threshold:
            sensitivity threshold used by bisectCondition.
    Returns
    -------
    newCells:
            array of cells [x,y,z,h] where x,y,z is the 3D position of the low corner and h is the size of the cell
    """
    cells_number = len(cells)
    cells_over = np.zeros(cells_number, dtype=bool)
    cells_masks = [np.arange(len(radii))] * cells_number
    new_cells = []
    new_cells_over = []
    new_cells_masks = []

    iter = 0
    while iter < stop_iter:
        for n in range(cells_number):
            x, y, z, size = cells[n]
            if cells_over[n]:
                _pass_complete_cell(
                    [new_cells, new_cells_over, new_cells_masks], [cells[n], True, True]
                )
                continue
            in_cell = in_cell_function(
                cells_masks[n], positions, radii, [x, y, z], size, threshold
            )
            if bisect_condition(size, in_cell, threshold, radii[cells_masks[n]]):
                _refine_grid_bisect(
                    size,
                    x,
                    y,
                    z,
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
        cells = new_cells

        if len(cells) == cells_number or iter == stop_iter:
            break
        cells_number = len(cells)
        cells_over = new_cells_over
        cells_masks = new_cells_masks
        new_cells = []
        new_cells_over = []
        new_cells_masks = []
        iter += 1
    return np.array(cells)


def occupancy_in_cell(
    mask, particles_pos, particles_radii, cell_pos, cell_size, threshold
):
    return (
        np.sum(np.abs(particles_pos[mask] - cell_pos - cell_size / 2), axis=1)
        < cell_size * 2
    )


def is_not_single_occupancy(cell_size, in_cell, threshold, particles_radii):
    count = np.sum(in_cell)
    return count > threshold


refine_grid_to_occupancy = partial(
    refine_grid, occupancy_in_cell, is_not_single_occupancy
)


RF = np.sqrt(3) / 2


def intersect_in_cell(
    mask, particles_pos, particles_radii, cell_pos, cell_size, threshold
):
    small_particle = (
        particles_radii[mask] * threshold < cell_size
    )  # No need to consider particles larger than cell
    return (
        np.linalg.norm(particles_pos[mask] - cell_pos - cell_size / 2, axis=1)
        < particles_radii[mask] + cell_size * RF
    ) & small_particle


def is_any_particle_included(cell_size, in_cell, threshold, particles_radii):
    return np.any(in_cell)


refine_grid_to_particle_scale = partial(
    refine_grid, intersect_in_cell, is_any_particle_included
)


def get_cell_centres(cells: np.ndarray) -> np.ndarray:
    return cells[:, :-1] + cells[:, -1][:, np.newaxis] / 2


def get_cell_volumes(cells: np.ndarray) -> np.ndarray:
    # needs revision to be spherical
    return cells[:, -1] ** 3


def create_regular_array(cells, xyz_range, dtype=np.uintc):
    """Converts an adaptive set of cells into a regular array"""
    xyz0 = np.min(cells, axis=0)
    dx = xyz0[-1]
    xyz0[-1] = 0
    grid_shape = [int((myRange[1] - myRange[0]) // dx) for myRange in xyz_range]
    N = len(cells)
    cell_range = np.arange(N, dtype=dtype)
    grid = np.empty(
        grid_shape, dtype=dtype
    )  # np.empty(grid_shape, dtype=int) #grid = np.full(grid_shape, np.prod(grid_shape)+10, dtype = int) slower but good for testing
    cellsBegin = np.round((cells[:, :-1] - xyz0[:-1]) / dx).astype(int)
    cellsFinish = np.round(
        (cells[:, :-1] - xyz0[:-1] + cells[:, -1][:, np.newaxis]) / dx
    ).astype(int)
    for i in cell_range:
        x_start, y_start, z_start = cellsBegin[i]
        x_end, y_end, z_end = cellsFinish[i]
        grid[x_start:x_end, y_start:y_end, z_start:z_end] = i
    return grid, dx**3


class Mochi(Martini):
    """
    Creates synthetic HI data cubes from simulation data.

    Usual use of martini involves first creating instances of classes from each
    of the required and optional sub-modules, then creating a
    :class:`~martini.martini.Martini` with these instances as arguments. The object can
    then be used to create synthetic observations, usually by calling
    :meth:`~martini.martini.Martini.insert_source_in_cube`,
    (optionally) :meth:`~martini.martini.Martini.add_noise`, (optionally)
    :meth:`~martini.martini.Martini.convolve_beam` and
    :meth:`~martini.martini.Martini.write_fits` in order.

    Parameters
    ----------
    source : SPHSource
        An instance of a class derived from
        :class:`~martini.sources.sph_source.SPHSource`.
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). See :doc:`sub-module documentation </sources/index>`.

    datacube : ~martini.datacube.DataCube
        A :class:`~martini.datacube.DataCube` instance.
        A description of the datacube to create, including pixels, channels,
        sky position. See :doc:`sub-module documentation </datacube/index>`.

    beam : _BaseBeam, optional
        An instance of a class derived from `~martini.beams._BaseBeam`.
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See
        :doc:`sub-module documentation </beams/index>`.

    noise : _BaseNoise, optional
        An instance of a class derived from :class:`~martini.noise._BaseNoise`.
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        :doc:`sub-module documentation </noise/index>`.

    sph_kernel : _BaseSPHKernel
        An instance of a class derived from :class:`~martini.sph_kernels._BaseSPHKernel`.
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        :doc:`SPH kernel sub-module documentation </sph_kernels/index>` for guidance.

    spectral_model : _BaseSpectrum
        An instance of a class derived from
        :class:`~martini.spectral_models._BaseSpectrum`.
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See
        :doc:`sub-module documentation </spectral_models/index>`.

    quiet : bool, optional
        If ``True``, suppress output to stdout.

    See Also
    --------
    martini.sources.sph_source.SPHSource
    martini.datacube.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models
    martini.martini.GlobalProfile

    Examples
    --------
    More detailed examples can be found in the examples directory in the github
    distribution of the package.

    The following example illustrates basic use of MARTINI, using a (very!)
    crude model of a gas disk. This example can be run by doing
    'from martini import demo; demo()'::

        # ------make a toy galaxy----------
        N = 500
        phi = np.random.rand(N) * 2 * np.pi
        r = []
        for L in np.random.rand(N):

            def f(r):
                return L - 0.5 * (2 - np.exp(-r) * (np.power(r, 2) + 2 * r + 2))

        r.append(fsolve(f, 1.0)[0])
        r = np.array(r)
        # exponential disk
        r *= 3 / np.sort(r)[N // 2]
        z = -np.log(np.random.rand(N))
        # exponential scale height
        z *= 0.5 / np.sort(z)[N // 2] * np.sign(np.random.rand(N) - 0.5)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz_g = np.vstack((x, y, z)) * U.kpc
        # linear rotation curve
        vphi = 100 * r / 6.0
        vx = -vphi * np.sin(phi)
        vy = vphi * np.cos(phi)
        # small pure random z velocities
        vz = (np.random.rand(N) * 2.0 - 1.0) * 5
        vxyz_g = np.vstack((vx, vy, vz)) * U.km * U.s**-1
        T_g = np.ones(N) * 8e3 * U.K
        mHI_g = np.ones(N) / N * 5.0e9 * U.Msun
        # ~mean interparticle spacing smoothing
        hsm_g = np.ones(N) * 4 / np.sqrt(N) * U.kpc
        # ---------------------------------

        source = SPHSource(
            distance=3.0 * U.Mpc,
            rotation={"L_coords": (60.0 * U.deg, 0.0 * U.deg)},
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            h=0.7,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )

        datacube = DataCube(
            n_px_x=128,
            n_px_y=128,
            n_channels=32,
            px_size=10.0 * U.arcsec,
            channel_width=10.0 * U.km * U.s**-1,
            spectral_centre=source.vsys,
        )

        beam = GaussianBeam(
            bmaj=30.0 * U.arcsec, bmin=30.0 * U.arcsec, bpa=0.0 * U.deg, truncate=4.0
        )

        noise = GaussianNoise(rms=3.0e-5 * U.Jy * U.beam**-1)

        spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

        sph_kernel = CubicSplineKernel()

        M = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
        )

        M.insert_source_in_cube()
        M.add_noise()
        M.convolve_beam()
        M.write_beam_fits(beamfile)
        M.write_fits(cubefile)
        print(f"Wrote demo fits output to {cubefile}, and beam image to {beamfile}.")
        try:
            M.write_hdf5(hdf5file)
        except ModuleNotFoundError:
            print("h5py package not present, skipping hdf5 output demo.")
        else:
            print(f"Wrote demo hdf5 output to {hdf5file}.")
    """

    def __init__(
        self,
        *,
        source: SPHSource,
        datacube: DataCube,
        beam: _BaseBeam | None = None,
        noise: _BaseNoise | None = None,
        sph_kernel: _BaseSPHKernel,
        spectral_model: _BaseSpectrum,
        quiet: bool = False,
    ) -> None:
        super().__init__(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
            quiet=quiet,
        )

        return

    def insert_source_in_cube(
        self,
        skip_validation: bool = False,
        progressbar: bool | None = None,
        ncpu: int = 1,
    ) -> None:
        """
        Populate the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the :class:`~martini.datacube.DataCube`
            is approximated for increased speed. For some combinations of pixel size,
            distance and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            ``RuntimeError`` if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter ``True``.

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If :class:`~martini.martini.Martini` was
            initialised with ``quiet`` set to ``True``, progress bars are switched off
            unless explicitly turned on.

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the :mod:`multiprocess` module (n.b. not the same as
            ``multiprocessing``).
        """
        self.source._init_los_pixcoords(self.datacube)
        self.sph_kernel._init_sm_ranges()
        cube = self.make_adaptive_cube()
        from radio_beam import Beam
        from .post_processing import get_Jy_from_mass

        assert self.beam.bmaj == self.beam.bmin
        jycube = np.moveaxis(
            get_Jy_from_mass(
                cube,
                Beam(self.beam.bmaj),
                self.datacube.px_size,
                self.datacube.channel_width,
                self.source.distance,
            ),
            (0, 1, 2),
            (2, 0, 1),
        )
        self.datacube._array += jycube
        return

    def make_adaptive_cube(
        self,
        interpolant=interpolants.SPH,
        radiative_transfer_model=radiative_transfer.adaptiveOpticallyThin,
        initial_grid_size: int = 2,
        min_radius: U.Quantity[U.pix] = 0.005 * U.pix,
        threshold: float = 0.5,
        refine_algorithm: Callable[
            [np.ndarray, U.Quantity[U.pix], U.Quantity[U.pix], float], np.ndarray
        ] = refine_grid_to_particle_scale,
    ):
        assert self.datacube.n_px_x == self.datacube.n_px_y
        # For now we are restricted to a cube (not cuboid) voxel grid. Means we're padding
        # the z-direction, should amend this later.
        # Hard code a starting grid just to get going:
        initial_grid_size = 2
        min_radius = 0.005 * U.pix
        threshold = 0.5
        pix_range = [(0, self.datacube.n_px_x + 2 * self.datacube.padx)] * 3
        initial_cells = np.column_stack(
            [
                np.linspace(*start_end, initial_grid_size, endpoint=False)
                for start_end in pix_range
            ]
            + [np.ones(initial_grid_size) * self.datacube.n_px_x / initial_grid_size],
        )
        positions = np.column_stack(
            (
                self.source.los_pixcoords,
                self.source.pixcoords[0],
                self.source.pixcoords[1],
            )
        )
        radii = np.clip(self.sph_kernel.sm_ranges, min_radius, np.inf)
        final_cells = refine_algorithm(
            initial_cells, positions.to_value(U.pix), radii.to_value(U.pix), threshold
        )
        cell_centres = get_cell_centres(final_cells) * U.pix
        cell_volumes = get_cell_volumes(final_cells) * U.pix
        field_velocity, field_mHI, field_temperature = interpolant(
            positions,
            self.source.skycoords.radial_velocity,
            radii,
            self.source.mHI_g,
            self.source.T_g * C.k_B / self.source.mHI_g,  # approx thermal energy
            self.source.mHI_g,  # mass goes here, approx for now
            self.sph_kernel.kernel,
            cell_centres,
            cell_volumes,
        )
        cube_field_indices, final_cell_volume = create_regular_array(
            final_cells, pix_range
        )
        final_cell_volume *= cell_volumes.unit
        cube_shape = cube_field_indices.shape
        cube_field_indices = cube_field_indices.flatten()
        return radiative_transfer_model(
            field_mHI,
            field_velocity,
            field_temperature,
            self.datacube.channel_width,
            final_cell_volume,
            cube_shape,
            cells=final_cells,
            cell_unit=U.pix,
            n_channels=self.datacube.n_channels,
        )
