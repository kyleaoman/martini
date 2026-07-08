"""
Create mock observations of HI sources from cosmological hydrodynamical simulations.

Provides :class:`~martini.martini.Mochi`, the main class of the package.
"""

import numpy as np
import cv2
from astropy import units as U
from martini.martini import Martini
from martini.datacube import DataCube
from martini.beams import _BaseBeam
from martini.sources import SPHSource
from martini.sph_kernels import _BaseSPHKernel
from martini.spectral_models import _BaseSpectrum
from martini.noise import _BaseNoise
from typing import Callable
from martini.mochi.cell_grid import AdaptiveCellGrid
from martini.mochi.refinement import refine_grid_to_particle_scale
from martini._unit_conversion import MHI_to_Jy_inplace


def _resize_cube(cube: U.Quantity, target_shape: tuple[int, int]) -> U.Quantity:
    """
    Resize a data cube to the target shape.

    Interpolation is handled with :func:`cv2.resize`.

    Parameters
    ----------
    cube : ~astropy.units.Quantity
        The cube to be resized.

    target_shape : tuple
        The desired shape as a 3-tuple.

    Returns
    -------
    ~astropy.units.Quantity
        The cube interpolated into the new desired shape using :func:`cv2.resize`.
    """
    if np.all(np.array(cube.shape[1:]) == np.array(target_shape)):
        return cube

    unit = cube.unit
    unitless_cube = cube.to_value(cube.unit)
    result = np.zeros((cube.shape[0],) + target_shape)
    for i in range(cube.shape[0]):
        result[i] = cv2.resize(
            unitless_cube[i].astype(float),
            target_shape[::-1],
            interpolation=cv2.INTER_AREA,
        )
    return result * np.prod(cube.shape[1:]) / np.prod(target_shape) * unit


class Mochi(Martini):
    """
    Creates synthetic HI data cubes from simulation data.

    ??.

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

    interpolant : Callable
        Function that interpolates the gas particle (or gas cell) properties onto the
        grid. E.g. :func:`~martini.mochi.interpolants.sph` or
        :func:`~martini.mochi.interpolants.mfm` found in the
        :mod:`~martini.mochi.interpolants` module.

    radiative_transfer : Callable
        Function that evaluates a mock spectral cube (of mass in each pixel-channel
        cell). E.g. :func:`~martini.mochi.radiative_transfer.adaptive_optically_thin`,
        can be found in the :mod:`~martini.mochi.radiative_transfer` module.

    refinement_strategy : Callable
        Function that decides and applies the grid refinement criterion. E.g.
        :func:`~martini.mochi.refinement.refine_grid_to_particle_scale`, can be found in
        the :mod:`~martini.mochi.refinement` module.

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

    Examples
    --------
    ??
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
        interpolant: Callable,  # fill in arg & return types
        radiative_transfer: Callable,  # fill in arg & return types
        refinement_strategy: Callable[
            [np.ndarray, U.Quantity[U.pix], U.Quantity[U.pix], float], np.ndarray
        ] = refine_grid_to_particle_scale,
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
        self.interpolant = interpolant
        self.radiative_transfer = radiative_transfer
        self.refinement_strategy = refinement_strategy
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
        # need to revise irrelevant (?) arguments: skip_validation, progressbar, ncpu
        self.source._init_los_pixcoords(self.datacube)
        self.sph_kernel._init_sm_ranges()
        adaptive_cell_grid = AdaptiveCellGrid(self.datacube)
        adaptive_cell_grid.init_particle_locations(self.source, self.sph_kernel)
        adaptive_cell_grid.eval_grid_refinement(self.refinement_strategy)
        adaptive_cell_grid.interpolate_fields(
            self.source, self.sph_kernel, self.interpolant
        )
        adaptive_cell_grid.create_regular_array()
        cube = adaptive_cell_grid.eval_radiative_transfer(
            self.datacube, self.spectral_model, self.radiative_transfer
        )
        cube *= (
            self.source.distance.to(U.Mpc) ** -2
        )  # use distances of individual cells instead
        cube /= np.abs(np.diff(self.datacube.velocity_channel_edges)).to(U.km / U.s)[
            :, np.newaxis, np.newaxis
        ]

        MHI_to_Jy_inplace(cube)
        cube /= U.pix**2
        target_shape = (
            self.datacube.n_px_x + 2 * self.datacube.padx,
            self.datacube.n_px_y + 2 * self.datacube.pady,
        )
        cube = _resize_cube(cube, target_shape)
        cube = np.flip(cube, 2)
        cube = np.moveaxis(
            cube,
            (0, 1, 2),
            (2, 1, 0),
        )
        insertion_slice = np.s_[..., 0] if self.datacube.stokes_axis else np.s_[...]
        self.datacube._array[insertion_slice] += cube
        return
