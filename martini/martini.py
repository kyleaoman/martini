"""
Create mock observations of HI sources from cosmological hydrodynamical simulations.

Provides :class:`~martini.martini.Martini`, the main class of the package, and a
simplified :class:`~martini.martini.GlobalProfile` class for use when only a spectrum (no
spatial information) is desired.
"""

import subprocess
import os
import contextlib
from tqdm.auto import tqdm
from scipy.signal import fftconvolve
import numpy as np
import astropy.units as U
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle
from astropy import __version__ as astropy_version
from .__version__ import __version__ as martini_version
import warnings
from warnings import warn
from martini.beams import _BaseBeam
from martini.datacube import DataCube, _GlobalProfileDataCube
from martini.sources import SPHSource
from martini.sph_kernels import DiracDeltaKernel, _BaseSPHKernel
from martini.spectral_models import _BaseSpectrum
from martini.noise import _BaseNoise
from martini._grid_search import (
    count_grid_intersections,
    find_grid_intersections,
    FindGridIntersectionsResult,
    _balanced_blocks,
)
from martini.sph_kernels import ScalarKernelIntegral
from martini._util import (
    NUMBA_AVAILABLE,
    numba_threads,
    resolve_ncpu,
    _to_unit_in_place,
    HIfreq,
    _FITS_BITPIX,
)
from martini._logging import _ProgressLogger
from typing import TYPE_CHECKING

if NUMBA_AVAILABLE:
    from numba import njit
    from numba.core.errors import NumbaExperimentalFeatureWarning
if TYPE_CHECKING or not NUMBA_AVAILABLE:
    prange = range  # pragma: no cover
else:
    from numba import prange

if TYPE_CHECKING:
    import h5py
    from matplotlib.figure import Figure

try:
    with open(os.devnull, "w") as _devnull:
        gc = subprocess.check_output(
            ["git", "describe", "--always"],
            stderr=_devnull,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        )
except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
    gc = b""
else:
    martini_version = martini_version + "_commit_" + gc.strip().decode()


def _insert_weighted_spectra_loop(
    cube: np.ndarray,
    spectra: np.ndarray,
    intersections: np.ndarray,
    cell_indices: np.ndarray,
    strides: np.ndarray,
    coords: np.ndarray,
    h: np.ndarray,
    kernel_slot: np.ndarray,
    kernel_integrals: tuple[ScalarKernelIntegral, ...],
    kernel_params: np.ndarray,
    block_edges: np.ndarray,
) -> None:
    """
    Add the kernel-weighted spectra of particles into the pixels that they overlap.

    Kernel weights are evaluated on the fly from the particle positions and are not
    stored. Each pixel is visited by one thread only, so the writes are thread safe.

    The compiled kernel integrals (``_scalar_kernel_integral`` of the kernel classes in
    martini.sph_kernels, compiled by ``_compile_scalar_integral``) are passed into the
    compiled loops as numba "first-class functions". This lets each particle use its own
    kernel (for adaptive kernels) without hard-coding the available kernels here.

    Parameters
    ----------
    cube : ~numpy.ndarray
        The data cube array, shape ``(nx, ny, n_channels)``, modified in place.

    spectra : ~numpy.ndarray
        The spectra of the particles, shape ``(N, n_channels)``.

    intersections : ~numpy.ndarray
        Particle-pixel overlaps grouped by pixel, see
        :func:`~martini._grid_search.find_grid_intersections`.

    cell_indices : ~numpy.ndarray
        The (flattened) index of each pixel with overlaps.

    strides : ~numpy.ndarray
        The range of ``intersections`` for each pixel.

    coords : ~numpy.ndarray
        Particle positions in pixel units, shape ``(N, 2)``.

    h : ~numpy.ndarray
        Rescaled smoothing lengths of the particles in pixels.

    kernel_slot : ~numpy.ndarray
        For each particle, the entry of ``kernel_integrals`` and ``kernel_params`` to
        use.

    kernel_integrals : tuple
        Table of compiled kernel integrals.

    kernel_params : ~numpy.ndarray
        Table of kernel parameters, shape ``(n_kernels, 2)``.

    block_edges : ~numpy.ndarray
        Pixels are processed in parallel in blocks ``block_edges[b]:block_edges[b + 1]``
        (indices into ``cell_indices``).
    """
    ny = cube.shape[1]
    n_channels = cube.shape[2]
    for b in prange(block_edges.size - 1):
        for c in range(block_edges[b], block_edges[b + 1]):
            i = cell_indices[c] // ny
            j = cell_indices[c] - i * ny
            for m in range(strides[c, 0], strides[c, 1]):
                p = intersections[m]
                slot = kernel_slot[p]
                w = kernel_integrals[slot](
                    coords[p, 0] - i,
                    coords[p, 1] - j,
                    h[p],
                    kernel_params[slot, 0],
                    kernel_params[slot, 1],
                )
                if w != 0:
                    for k in range(n_channels):
                        cube[i, j, k] += w * spectra[p, k]


if NUMBA_AVAILABLE:
    _insert_weighted_spectra_loop = njit(parallel=True, cache=True)(
        _insert_weighted_spectra_loop
    )


def _plan_segments(mem_estimate: np.ndarray, allowance: float) -> list[slice]:
    """
    Divide particles into contiguous batches that each fit within a memory allowance.

    Parameters
    ----------
    mem_estimate : ~numpy.ndarray
        Estimated memory required to process each particle.

    allowance : float
        Memory available to process each batch, in the same units as
        ``mem_estimate``.

    Returns
    -------
    list
        List of :class:`slice` objects selecting the particles in each batch.

    Raises
    ------
    ValueError
        If any single particle needs more memory than the allowance.
    """
    if np.any(mem_estimate > allowance):
        raise ValueError("Memory allowance insufficient for a single particle.")
    # Should always have a particle if we get here, but make very sure so we don't
    # enter an infinite loop:
    assert len(mem_estimate) > 0, (
        "Particles processing in batches but no particles present, this is an error."
    )
    segments = []
    segment_start = 0
    while segment_start < len(mem_estimate):
        cumsum = np.cumsum(mem_estimate[segment_start:])
        segment_length = np.searchsorted(cumsum, allowance, side="right")
        segment_end = segment_start + int(segment_length)
        segments.append(slice(segment_start, segment_end))
        segment_start = segment_end
    return segments


class _BaseMartini:
    """
    Common methods for the core Martini class and related classes.

    Parameters
    ----------
    source : SPHSource
        An instance of a class derived from :class:`martini.sources.SPHSource`.
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

    _prune_kwargs : dict
        Arguments to pass through to the :meth:`martini.martini.Martini._prune_particles`
        function, intended for internal use only.

    See Also
    --------
    martini.sources.sph_source.SPHSource
    martini.datacube.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models
    martini.martini.GlobalProfile
    """

    source: SPHSource
    _datacube: DataCube
    beam: _BaseBeam | None
    noise: _BaseNoise | None
    sph_kernel: _BaseSPHKernel
    spectral_model: _BaseSpectrum
    quiet: bool
    _numba_enabled: bool = NUMBA_AVAILABLE  # enables switching off in tests
    # name of the object the source is inserted into, for log messages:
    _target_name: str = "data cube"
    # The DataCube has no line-of-sight distance (z) axis. Subclasses that need particle
    # coordinates along it set the number of pixels along it (_n_px_z) before calling
    # _BaseMartini.__init__. The z axis is not padded: there is no beam along it and
    # particles beyond it should not contribute.
    _n_px_z: int | None = None

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
        _prune_kwargs: dict = {},
    ) -> None:
        self.quiet = quiet
        self._logger = _ProgressLogger()
        self.source = source
        self._datacube = datacube
        self.beam = beam
        self.noise = noise
        self.sph_kernel = sph_kernel
        self.sph_kernel._logger = self._logger
        self.spectral_model = spectral_model

        # Estimate memory for main allocations.
        # - The datacube itself (product of shape times size of an element).
        # - Particle arrays:
        #   - T_g               (N, 1)
        #   - mHI_g             (N, 1)
        #   - hsm_g             (N, 1)
        #   - coordinates_g     (N, 6)
        #   - skycoords         (N, 6)
        #   - spectralcoords    (N, 1)
        #   - pixcoords         (N, 3)
        #   Total 19 * N * 8B (64-bit floats)
        self.baseline_mem_estimate = (
            np.prod(self._datacube.datacube_shape)
            * np.dtype(self._datacube.cube_dtype).itemsize
            + 19 * self.source.npart * 8
        ) / 1024**3  # GB

        if self.beam is not None:
            self.beam.init_kernel(self._datacube)
            self._datacube.add_pad(self.beam.needs_pad())

        self.source._init_skycoords()
        self.source._init_pixcoords(
            self._datacube, n_px_z=self._n_px_z
        )  # after datacube is padded

        self.sph_kernel._init_sm_lengths(source=self.source, datacube=self._datacube)
        self.sph_kernel._init_sm_ranges()
        self._prune_particles(
            **_prune_kwargs
        )  # prunes both source, and kernel if applicable

        return

    def _prune_particles(
        self,
        spatial: bool = True,
        spectral: bool = True,
        mass: bool = True,
        los_distance: bool = False,
    ) -> None:
        """
        Identify and remove particles that cannot contribute to the DataCube.

        This helps speed up the later calculations. Assumes the kernel is 0 at
        distances greater than the kernel size (which may differ from the
        SPH smoothing length).

        Parameters
        ----------
        spatial : bool
            If ``True``, prune particles that fall outside the spatial aperture.
        spectral : bool
            If ``True``, prune particles that fall outside the spectral bandwidth.
        mass : bool
            If ``True``, prune particles that have zero HI mass.
        los_distance : bool
            If ``True``, prune particles that fall outside the line-of-sight distance
            extent of the region. Requires ``_n_px_z`` to be set.
        """
        if los_distance and self.source.los_distance_pixcoords is None:
            raise ValueError(
                "Pruning along the line-of-sight distance axis requires '_n_px_z' (the "
                "number of pixels along the line-of-sight distance axis) to be set."
            )
        with self._logger.step(
            f"Pruning particles that will not contribute to {self._target_name}...",
            "Pruned particles",
            quiet=self.quiet,
        ):
            self._logger.message(
                f"Source module contained {self.source.npart} particles with total HI"
                f" mass of {self.source.mHI_g.sum() << U.Msun:.2e}.",
                quiet=self.quiet,
            )
            spectrum_half_width = self.spectral_model.half_width(self.source) / np.max(
                np.abs(np.diff(self._datacube.velocity_channel_edges))
            )
            # Pixel (or cell) centres are at integer coordinates, so the region spans
            # -0.5 to n - 0.5 along each axis. Keep any particle with a kernel that could
            # touch the region, including particles inside the edge pixels with
            # kernels smaller than a pixel.
            spatial_reject_conditions = (
                (
                    (
                        self.source.pixcoords[:2]
                        + self.sph_kernel.sm_ranges[np.newaxis]
                        < -0.5 * U.pix
                    ).any(axis=0),
                    self.source.pixcoords[0] - self.sph_kernel.sm_ranges
                    > (self._datacube.n_px_x + self._datacube.padx * 2 - 0.5) * U.pix,
                    self.source.pixcoords[1] - self.sph_kernel.sm_ranges
                    > (self._datacube.n_px_y + self._datacube.pady * 2 - 0.5) * U.pix,
                    np.isnan(self.source.pixcoords[:2]).any(axis=0),
                )
                if spatial
                else tuple()
            )
            spectral_reject_conditions = (
                (
                    self.source.pixcoords[2] + 4 * spectrum_half_width * U.pix
                    < 0 * U.pix,
                    self.source.pixcoords[2] - 4 * spectrum_half_width * U.pix
                    > self._datacube.n_channels * U.pix,
                )
                if spectral
                else tuple()
            )
            los_distance_reject_conditions: tuple[np.ndarray, ...] = tuple()
            if los_distance:
                assert self._n_px_z is not None  # placate mypy
                los_distance_reject_conditions = (
                    self.source.los_distance_pixcoords + self.sph_kernel.sm_ranges
                    < -0.5 * U.pix,
                    self.source.los_distance_pixcoords - self.sph_kernel.sm_ranges
                    > (self._n_px_z - 0.5) * U.pix,
                )
            mass_reject_conditions = (self.source.mHI_g == 0,) if mass else tuple()
            accept_mask = np.logical_not(
                np.logical_or.reduce(
                    (np.zeros(self.source.npart, dtype=bool),)
                    + spatial_reject_conditions
                    + los_distance_reject_conditions
                    + spectral_reject_conditions
                    + mass_reject_conditions
                )
            )
            self.source.apply_mask(accept_mask)
            self.sph_kernel._apply_mask(accept_mask)
            self._logger.message(
                f"{self.source.npart} particles remaining with total HI mass of "
                f"{self.source.mHI_g.sum() << U.Msun:.2e}.",
                quiet=self.quiet,
            )
        return

    def _count_px_hits(self, sm_ranges: np.ndarray, ncpu: int) -> np.ndarray:
        """
        Count the pixels overlapped by each particle's smoothing kernel.

        Parameters
        ----------
        sm_ranges : ~numpy.ndarray
            Smoothing kernel ranges of the particles, in pixels.

        ncpu : int
            Number of threads to use.

        Returns
        -------
        ~numpy.ndarray
            Number of pixels overlapped by each particle.
        """
        return count_grid_intersections(
            self._datacube.datacube_shape[:2],
            self.source.pixcoords[:2].to_value(U.pix).T,
            sm_ranges,
            ncpu=ncpu,
        )

    def _find_px_hits(
        self,
        segment: slice,
        sm_ranges: np.ndarray,
        ncpu: int,
        return_distances: bool = True,
    ) -> FindGridIntersectionsResult:
        """
        Find the pixels overlapped by the smoothing kernels of a batch of particles.

        Parameters
        ----------
        segment : slice
            Selects the batch of particles.

        sm_ranges : ~numpy.ndarray
            Smoothing kernel ranges of all particles, in pixels.

        ncpu : int
            Number of threads to use.

        return_distances : bool
            If ``False``, offsets between particles and pixel centres are not returned.

        Returns
        -------
        ~martini._grid_search.FindGridIntersectionsResult
            The particle-pixel overlaps.
        """
        return find_grid_intersections(
            self._datacube.datacube_shape[:2],
            self.source.pixcoords[:2, segment].to_value(U.pix).T,
            sm_ranges[segment],
            ncpu=ncpu,
            dist_dtype=self.spectral_model.spec_dtype,
            return_distances=return_distances,
        )

    def _insert_weighted_spectra(
        self,
        spectra: np.ndarray,
        gs: FindGridIntersectionsResult,
        segment: slice,
        kernel_arrays: tuple[
            np.ndarray, tuple[ScalarKernelIntegral, ...], np.ndarray, np.ndarray
        ],
        ncpu: int,
    ) -> None:
        """
        Evaluate kernel weights on the fly and add weighted spectra into the cube.

        Uses compiled kernel integrals, so that the weights are never stored.

        Parameters
        ----------
        spectra : ~numpy.ndarray
            The spectra of the particles in the batch, as a raw array.

        gs : ~martini._grid_search.FindGridIntersectionsResult
            The particle-pixel overlaps of the particles in the batch.

        segment : slice
            Selects the batch of particles.

        kernel_arrays : tuple
            Kernel description, see
            :meth:`~martini.sph_kernels._BaseSPHKernel._numba_kernel_arrays`.

        ncpu : int
            Number of threads to use.
        """
        if gs.intersections.size == 0:
            return
        cube_view = (
            self._datacube._array[:, :, :, 0].value
            if self._datacube.stokes_axis
            else self._datacube._array.value
        )
        kernel_slot, kernel_integrals, kernel_params, h = kernel_arrays
        hits_per_px = gs.strides[:, 1] - gs.strides[:, 0]
        with numba_threads(ncpu), warnings.catch_warnings():
            # compiled kernel integrals are passed to compiled code as numba
            # "first-class functions", flagged as an experimental feature:
            if NUMBA_AVAILABLE:
                warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)
            _insert_weighted_spectra_loop(
                cube_view,
                spectra,
                gs.intersections,
                gs.cell_indices,
                gs.strides,
                np.ascontiguousarray(
                    self.source.pixcoords[:2, segment].to_value(U.pix).T,
                    dtype=np.float64,
                ),
                h[segment],
                kernel_slot[segment],
                kernel_integrals,
                kernel_params,
                _balanced_blocks(hits_per_px, 8 * ncpu),
            )

    def _weighted_sum_and_insert_in_cube(
        self,
        spectra: np.ndarray,
        weights: np.ndarray,
        flat_cube_indices: np.ndarray,
        strides: np.ndarray,
        intersections: np.ndarray,
        ncpu: int = 1,
        progressbar: bool = False,
    ) -> int:
        """
        Evaluate weighted combination of spectra and write to datacube array.

        Given the pre-computed particle spectra, the kernel weights for their
        contributions to each pixel the information needed to map each weighted spectrum
        to the pixels where it contributes, this function sums up the contributions and
        fills the target datacube array. This serial implementation is a fallback used
        when :mod:`numba` is unavailable or disabled (otherwise kernel weights are
        evaluated on the fly in compiled code). This function expects bare arrays (not
        :class:`~astropy.units.Quantity` arrays); the calling function is responsible
        for checking units.

        Parameters
        ----------
        spectra : ~numpy.ndarray
            The array of spectra, one row per particle, as a raw array (can be a view of
            :class:`~astropy.units.Quantity`).

        weights : ~numpy.ndarray
            The 1D array of weights, one per particle-pixel match, grouped by pixel, as
            a raw array (can be a view of :class:`~astropy.units.Quantity`).

        flat_cube_indices : ~numpy.ndarray
            The 1D array of pixel indices, one per pixel, in the same order as the pixel-
            groups in the ``weights``. These are unpacked into array locations similar
            to `~numpy.unravel_index`.

        strides : ~numpy.ndarray
            The 2D array of strides (start and end locations) that select the ``weights``
            and ``intersections`` for each pixel. Each row defines a stride and has length
            2.

        intersections : ~numpy.ndarray
            The 1D array of particle-pixel matches, grouped by pixel. Each entry therefore
            serves as an index into the ``spectra`` array, the corresponding pixel is
            identified by the ``flat_cube_indices``. Groups within this array
            corresponding to individual pixels are selected using a row from ``strides``.

        ncpu : int
            Requested number of threads. This implementation is serial, a warning is
            issued if more than one thread is requested without :mod:`numba`.

        progressbar : bool, optional
            If ``True``, show a progress bar.

        Returns
        -------
        int
            The number of cores actually used (always ``1``).
        """
        cube_view = (
            self._datacube._array[:, :, :, 0].value
            if self._datacube.stokes_axis
            else self._datacube._array.value
        )
        dec_dim = cube_view.shape[1]
        total_elements = len(flat_cube_indices)

        if ncpu != 1:
            warn("Parallelization not available without 'numba'.", RuntimeWarning)
            ncpu = 1

        for i in tqdm(range(total_elements), disable=not progressbar):
            start_j, end_j = strides[i, 0], strides[i, 1]
            assert start_j < end_j, (
                "Invalid pixel overlap group length encountered, this is an error."
            )

            flat_index = flat_cube_indices[i]
            ra_index = flat_index // dec_dim
            dec_index = flat_index % dec_dim

            j_slice = slice(start_j, end_j)
            w_sub = weights[j_slice, np.newaxis]
            spec_sub = spectra[intersections[j_slice], :]

            cube_view[ra_index, dec_index, :] += np.sum(w_sub * spec_sub, axis=0)
        return ncpu

    def _insertion_mem_estimate(
        self, px_hits: np.ndarray, sm_ranges: np.ndarray, weight_and_insert: bool
    ) -> tuple[np.ndarray, float]:
        """
        Estimate the memory needed to insert the source in the cube.

        Estimates memory (bytes), with s the size of an element of the spectra and d=2
        the number of spatial dimensions:

        - Per particle-pixel overlap, the index of the particle (4B), grouped by pixel.
          If the kernel weights are evaluated on the fly, that's all. Otherwise:

          - The offset vector between the particle and pixel centre (``d*s``).
          - While evaluating weights, memory use comes mostly in chunks of ``s`` per
            overlap. How many such chunks are allocated depends on the kernel
            function, but is not wildly different. The ``WendlandC2`` kernel uses ~17
            for instance, but is among the most efficient. Allow for 24 chunks.

        - Per particle, the spectrum ``(n_channels * s)`` and an entry in the list of
          candidate particles for each pixel row that it may overlap (4B).
        - Per batch of particles, a fixed overhead of up to 24B per pixel (pixels with
          overlaps and the corresponding ranges of overlaps).

        Parameters
        ----------
        px_hits : ~numpy.ndarray
            Number of pixels overlapped by each particle.

        sm_ranges : ~numpy.ndarray
            Smoothing kernel ranges of the particles, in pixels.

        weight_and_insert : bool
            ``True`` if kernel weights are evaluated on the fly while inserting spectra
            (so not stored).

        Returns
        -------
        tuple
            The memory needed for each particle, and the fixed memory needed for each
            batch of particles, in GB.
        """
        spec_itemsize = np.dtype(self.spectral_model.spec_dtype).itemsize
        mem_per_hit = (
            4 if weight_and_insert else 4 + 2 * spec_itemsize + 24 * spec_itemsize
        )
        nx = self._datacube.datacube_shape[0]
        mem_estimate = (
            px_hits * mem_per_hit
            + self._datacube.n_channels * spec_itemsize
            + 4 * np.minimum(2 * sm_ranges + 3, nx)
        ) / 1024**3  # GB
        segment_overhead_mem_estimate = 24 * self._datacube.current_n_px / 1024**3  # GB
        return mem_estimate, segment_overhead_mem_estimate

    def _insert_source_in_cube(
        self,
        skip_validation: bool = False,
        progressbar: bool | None = None,
        ncpu: int = 1,
        mem_lim_GB: float = 4.0,
        quiet: bool | None = None,
        _no_summary: bool = False,
    ) -> None:
        """
        Populate the :class:`~martini.datacube.DataCube` with flux from source particles.

        With numba, kernel weights are evaluated on the fly in compiled code while
        inserting the weighted spectra into the cube (the weights are never stored).
        Without numba, weights are evaluated with numpy and stored, then used to insert
        spectra.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the :class:`martini.datacube.DataCube` is
            approximated for increased speed. For some combinations of pixel size,
            distance and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            ``RuntimeError`` if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter ``True``.

        progressbar : bool, optional
            A progress bar is shown by default if supported (usually not supported when
            :mod:`numba` is installed). If :class:`~martini.martini.Martini` was
            initialised with ``quiet`` set to ``True``, progress bars are switched off
            unless explicitly turned on.

        ncpu : int
            Number of threads to use in main source insertion loop. Using more than one
            thread requires the :mod:`numba` module. Can be set to ``-1`` to use as many
            threads as available cores. Currently speedup from multiple cores is limited,
            see full documentation for details.

        mem_lim_GB : float
            The peak memory usage can get very large if many particle kernels touch many
            pixels in the cube. The particles can be processed in batches to mitigate
            this, but this slows down the code. The memory limit set here (in GB) will
            trigger batching if estimated memory usage is expected to exceed it. This is
            a "soft" limit since some allocations (e.g. temporary arrays when evaluating
            the SPH kernel) cannot be exactly predicted.

        quiet : bool, optional
            If ``True``, suppress output to stdout. If specified, takes precedence over
            ``quiet`` parameter of class.

        _no_summary : bool
            If ``True`` the final summary message is suppressed (e.g. so that a subclass
            can replace it).
        """
        if self.baseline_mem_estimate > mem_lim_GB:
            raise RuntimeError(
                f"Requested memory limit ({mem_lim_GB}GB) is less than estimated "
                f"baseline memory requirement of {self.baseline_mem_estimate}GB. Increase"
                f" `mem_lim_GB` when calling `insert_source_in_cube`."
            )
        ncpu = resolve_ncpu(ncpu)
        quiet = self.quiet if quiet is None else quiet

        weight_and_insert = self._numba_enabled
        kernel_arrays = None
        if weight_and_insert:
            kernel_arrays = self.sph_kernel._numba_kernel_arrays()
            if kernel_arrays is None:
                raise ValueError(
                    f"The SPH kernel {type(self.sph_kernel).__name__} has no compiled "
                    "kernel integral (`_scalar_kernel_integral`), which is required when "
                    "'numba' is available. See the kernels in `martini.sph_kernels` for "
                    "examples."
                )
        if progressbar and weight_and_insert:
            warn(
                "'numba'-accelerated 'martini' does not support progress bar (but it "
                "will be pretty fast anyway!).",
                RuntimeWarning,
            )
        if weight_and_insert:
            progressbar = False
        elif progressbar is None:
            progressbar = not quiet

        log = self._logger
        with log.step(
            f"Inserting source in {self._target_name}...",
            f"Source inserted in {self._target_name}",
            quiet=quiet,
        ):
            self.sph_kernel._confirm_validation(noraise=skip_validation, quiet=quiet)

            clipped_sm_ranges = np.clip(
                self.sph_kernel.sm_ranges.to_value(U.pix), np.sqrt(2) / 2, np.inf
            )
            with log.step(
                "Counting particle-pixel overlaps...", "Counted overlaps", quiet=quiet
            ) as step:
                px_hits = self._count_px_hits(clipped_sm_ranges, ncpu)
                step.end = f"Counted {np.sum(px_hits)} particle-pixel overlaps"
            mem_estimate, segment_overhead_mem_estimate = self._insertion_mem_estimate(
                px_hits, clipped_sm_ranges, weight_and_insert
            )
            single_batch_peak_mem_estimate = (
                np.sum(mem_estimate)
                + segment_overhead_mem_estimate
                + self.baseline_mem_estimate
            )
            if single_batch_peak_mem_estimate > mem_lim_GB:
                warn(
                    f"Requested memory limit ({mem_lim_GB}GB) is insufficient to process "
                    "all particles at once (estimated memory required: "
                    f"{single_batch_peak_mem_estimate:.1f}GB). Processing in batches "
                    "instead, this could take longer. Increase `mem_lim_GB` when calling "
                    "`insert_source_in_cube` if possible.",
                    RuntimeWarning,
                )
                segment_mem_allowance = (
                    mem_lim_GB
                    - self.baseline_mem_estimate
                    - segment_overhead_mem_estimate
                )
                try:
                    segments = _plan_segments(mem_estimate, segment_mem_allowance)
                except ValueError as e:
                    raise RuntimeError(
                        f"Requested memory limit ({mem_lim_GB}GB) minus the estimated "
                        f"baseline memory ({self.baseline_mem_estimate}GB, e.g. to store"
                        " particle data) leaves insufficient memory for "
                        "`insert_source_in_cube`, even when particles are processed in "
                        "batches. Increase `mem_lim_GB` when calling "
                        "`insert_source_in_cube`."
                    ) from e
            else:
                segments = [slice(None)]
            for i, segment in enumerate(segments, 1):
                group: contextlib.AbstractContextManager = (
                    log.step(
                        f"Processing particle group {i} of {len(segments)}...",
                        f"Processed particle group {i} of {len(segments)}",
                        quiet=quiet,
                    )
                    if len(segments) > 1
                    else contextlib.nullcontext()
                )
                with group:
                    with log.step(
                        "Indexing particle-pixel overlaps...",
                        "Indexed particle-pixel overlaps",
                        quiet=quiet,
                    ) as step:
                        gs = self._find_px_hits(
                            segment,
                            clipped_sm_ranges,
                            ncpu,
                            return_distances=not weight_and_insert,
                        )
                        step.annotation = f"on {ncpu} cores"
                    if not weight_and_insert:
                        with log.step(
                            "Evaluating kernel weights...",
                            "Evaluated kernel weights",
                            quiet=quiet,
                        ):
                            segment_start = (
                                0 if segment.start is None else segment.start
                            )
                            assert gs.distances is not None
                            weights = self.sph_kernel._px_weight(
                                U.Quantity(gs.distances.T, U.pix, copy=False),
                                mask=segment_start + gs.intersections,
                            )
                    with log.step(
                        "Evaluating pixel spectra...",
                        "Evaluated pixel spectra",
                        quiet=quiet,
                    ) as step:
                        spectra = self.spectral_model._eval_spectra(
                            self.source, self._datacube, ncpu=ncpu, mask=segment
                        )
                        step.annotation = f"on {ncpu} cores"
                    with log.step(
                        "Evaluating kernel weights and reducing pixel spectra into "
                        "cube..."
                        if weight_and_insert
                        else "Reducing pixel spectra into cube...",
                        "Reduced pixel spectra into cube",
                        quiet=quiet,
                    ) as step:
                        if weight_and_insert:
                            assert kernel_arrays is not None
                            assert (
                                self._datacube.current_units == spectra.unit * U.pix**-2
                            )
                            self._insert_weighted_spectra(
                                spectra.value, gs, segment, kernel_arrays, ncpu
                            )
                            insert_used_ncpu = ncpu
                        else:
                            assert (
                                self._datacube.current_units
                                == spectra.unit * weights.unit
                            )
                            insert_used_ncpu = self._weighted_sum_and_insert_in_cube(
                                spectra.value,
                                weights.value,
                                gs.cell_indices,
                                gs.strides,
                                gs.intersections,
                                ncpu=ncpu,
                                progressbar=progressbar,
                            )
                        step.annotation = f"on {insert_used_ncpu} cores"
            with log.step(
                "Converting cube to Jy/arcsec^2...", "Converted cube", quiet=quiet
            ):
                self._datacube._array = _to_unit_in_place(
                    self._datacube._array,
                    U.Jy / U.arcsec**2,
                    equivalencies=[self._datacube.arcsec2_to_pix],
                )
        if not quiet and not _no_summary:
            # sum over (views of) the cube first, then apply scalar factors, to avoid
            # allocating cube-sized temporary arrays:
            cube_no_pad = self._datacube._array[self._datacube.pad_mask]
            inserted_flux_density = (
                np.sum(cube_no_pad) * self._datacube.px_size**2
            ).to(U.Jy)
            inserted_mass = (
                2.36e5
                * U.Msun
                * self.source.distance.to_value(U.Mpc) ** 2
                * np.sum(
                    (cube_no_pad.sum((0, 1)) * self._datacube.px_size**2)
                    .squeeze()
                    .to_value(U.Jy)
                    * np.abs(
                        np.diff(
                            self._datacube.velocity_channel_edges.to_value(U.km / U.s)
                        )
                    )
                )
            )
            log.message(
                f"Flux density in cube: {inserted_flux_density:.2e}",
                f"Mass in cube (assuming distance {self.source.distance:.2f} and a"
                " spatially resolved source):"
                f" {inserted_mass:.2e}",
                f"  [{inserted_mass / self.source.input_mass * 100 << 1:.0f}%"
                " of initial source mass]",
                f"Maximum pixel: {self._datacube._array.max():.2e}",
                "Median non-zero pixel:"
                f" {np.median(self._datacube._array[self._datacube._array > 0]):.2e}",
                extra_indent=1,
            )
        return

    def reset(self) -> None:
        """Re-initialize the :class:`~martini.datacube.DataCube` with zero-values."""
        init_kwargs = {
            "n_px_x": self._datacube.n_px_x,
            "n_px_y": self._datacube.n_px_y,
            "n_channels": self._datacube.n_channels,
            "px_size": self._datacube.px_size,
            "channel_width": self._datacube.channel_width,
            "spectral_centre": self._datacube.spectral_centre,
            "ra": self._datacube.ra,
            "dec": self._datacube.dec,
            "stokes_axis": self._datacube.stokes_axis,
        }
        self._datacube = DataCube(**init_kwargs)
        if self.beam is not None:
            self._datacube.add_pad(self.beam.needs_pad())
        return

    def preview(
        self,
        max_points: int = 5000,
        fig: int = 1,
        lim: str | U.Quantity[U.deg] | None = None,
        vlim: str | U.Quantity[U.km / U.s] | None = None,
        point_scaling: str = "auto",
        title: str = "",
        save: str | None = None,
    ) -> "Figure":
        """
        Produce a figure showing the source particle coordinates and velocities.

        The data cube region is also outlined.

        Makes a 3-panel figure showing the projection of the source as it will appear in
        the mock observation. The first panel shows the particles in the y-z plane,
        coloured by the x-component of velocity (MARTINI projects the source along the
        x-axis). The second and third panels are position-velocity diagrams showing the
        x-component of velocity against the y and z coordinates, respectively. The red
        boxes drawn on the panels show the data cube extent in position and velocity at
        the distance and systemic velocity of the source.

        Parameters
        ----------
        max_points : int, optional
            Maximum number of points to draw per panel, the particles will be randomly
            subsampled if the source has more.

        fig : int, optional
            Number of the figure in matplotlib, it will be created as ``plt.figure(fig)``.

        lim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity` with dimensions of angle.
            The coordinate axes extend from ``-lim`` to ``lim``. If unspecified, the
            maximum angular separation between the source centre defined by its ``ra`` and
            ``dec`` and the particle sky positions is used. Whether specified or not, the
            axes are expanded if necessary to contain the extent of the data cube.
            Alternatively, the string ``"datacube"`` can be passed. In this case the axis
            limits are set by the extent of the data cube.

        vlim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity` with dimensions of speed.
            The velocity axes and colour bar extend from ``-vlim`` to ``vlim``. If
            unspecified, the maximum absolute velocity of particles in the source is
            used. Whether specified or not, the axes are expanded if necessary to contain
            the extent of the data cube. Alternatively, ``"datacube"`` can be
            passed. In this case the axis limits are set by the extent of the data cube.

        point_scaling : str, optional
            By default points are scaled in size and transparency according to their HI
            mass and the smoothing length (loosely proportional to their surface
            densities, but with different scaling to achieve a visually useful plot). For
            some sources the automatic scaling may not give useful results, using
            ``point_scaling="fixed"`` will plot points of constant size without opacity.

        title : str, optional
            A title for the figure can be provided.

        save : str, optional
            If provided, the figure is saved using ``plt.savefig(save)``. A `.png` or
            `.pdf` suffix is recommended.

        Returns
        -------
        matplotlib.figure.Figure
            The preview :class:`~matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt

        # pass through arguments, except save (which we will do later if desired)
        figure = self.source.preview(
            max_points=max_points,
            fig=fig,
            lim=lim if lim != "datacube" else None,
            vlim=vlim if vlim != "datacube" else None,
            point_scaling=point_scaling,
            title=title,
            save=None,
        )
        y_cube = (
            Angle(self._datacube.ra)
            .wrap_at(self.source.ra + 180 * U.deg)
            .to_value(U.deg)
        )
        z_cube = self._datacube.dec.to_value(U.deg)
        v_cube = self._datacube.spectral_centre.to_value(U.km / U.s)
        dy_cube = 0.5 * (self._datacube.px_size * self._datacube.n_px_x).to_value(
            U.deg
        )  # half-length
        dz_cube = 0.5 * (self._datacube.px_size * self._datacube.n_px_y).to_value(
            U.deg
        )  # half-length
        dv_cube = 0.5 * (
            self._datacube.channel_width * self._datacube.n_channels
        ).to_value(U.km / U.s)
        if dy_cube > 5:
            warn("DataCube RA range > 5deg, preview is very approximate!")
        if dz_cube > 5:
            warn("DataCube Dec range > 5deg, preview is very approximate!")
        if np.abs(self._datacube.dec.to_value(U.deg)) + 0.5 * dz_cube > 85:
            warn("DataCube extent within 5deg of pole, preview is very approximate!")
        sp1, sp2, sp3, cb = figure.get_axes()
        sp1.add_patch(
            plt.Rectangle(
                (
                    (y_cube - dy_cube),
                    (z_cube - dz_cube),
                ),
                2 * dy_cube,
                2 * dz_cube,
                linewidth=5,
                edgecolor="red",
                facecolor="None",
            )
        )
        sp2.add_patch(
            plt.Rectangle(
                (
                    (y_cube - dy_cube),
                    (v_cube - dv_cube),
                ),
                2 * dy_cube,
                2 * dv_cube,
                linewidth=5,
                edgecolor="red",
                facecolor="None",
            )
        )
        sp3.add_patch(
            plt.Rectangle(
                (
                    (z_cube - dz_cube),
                    (v_cube - dv_cube),
                ),
                2 * dz_cube,
                2 * dv_cube,
                linewidth=5,
                edgecolor="red",
                facecolor="None",
            )
        )
        if lim == "datacube":
            sp1.set_xlim((y_cube + dy_cube), (y_cube - dy_cube))
            sp1.set_ylim((z_cube - dz_cube), (z_cube + dz_cube))
            sp2.set_xlim((y_cube + dy_cube), (y_cube - dy_cube))
            sp3.set_xlim((z_cube - dz_cube), (z_cube + dz_cube))
        elif lim is None:
            sp1.set_xlim(
                (
                    max((y_cube + 1.05 * dy_cube), sp1.get_xlim()[0]),
                    min((y_cube - 1.05 * dy_cube), sp1.get_xlim()[1]),
                )
            )
            sp2.set_xlim(
                (
                    max((y_cube + 1.05 * dy_cube), sp2.get_xlim()[0]),
                    min((y_cube - 1.05 * dy_cube), sp2.get_xlim()[1]),
                )
            )
            sp1.set_ylim(
                (
                    min((z_cube - 1.05 * dz_cube), sp1.get_ylim()[0]),
                    max((z_cube + 1.05 * dz_cube), sp1.get_ylim()[1]),
                )
            )
            sp3.set_xlim(
                (
                    min((z_cube - 1.05 * dz_cube), sp3.get_xlim()[0]),
                    max((z_cube + 1.05 * dz_cube), sp3.get_xlim()[1]),
                )
            )
        if vlim == "datacube":
            sp2.set_ylim(
                (v_cube - dv_cube),
                (v_cube + dv_cube),
            )
            sp3.set_ylim(
                (v_cube - dv_cube),
                (v_cube + dv_cube),
            )
        elif vlim is None:
            sp2.set_ylim(
                (
                    min((v_cube - 1.05 * dv_cube), sp2.get_ylim()[0]),
                    max((v_cube + 1.05 * dv_cube), sp2.get_ylim()[1]),
                )
            )
            sp3.set_ylim(
                (
                    min((v_cube - 1.05 * dv_cube), sp3.get_ylim()[0]),
                    max((v_cube + 1.05 * dv_cube), sp3.get_ylim()[1]),
                )
            )

        if save is not None:
            plt.savefig(save)
        return figure


class Martini(_BaseMartini):
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

    @property
    def datacube(self) -> DataCube:
        """
        The :class:`~martini.datacube.DataCube` object for this mock observation.

        Returns
        -------
        ~martini.datacube.DataCube
            The :class:`~martini.datacube.DataCube` contained by this
            :class:`~martini.martini.Martini` instance.
        """
        return self._datacube

    def insert_source_in_cube(
        self,
        skip_validation: bool = False,
        progressbar: bool | None = None,
        ncpu: int = 1,
        mem_lim_GB: float = 4.0,
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
            A progress bar is shown by default if supported (usually not supported when
            :mod:`numba` is installed). If :class:`~martini.martini.Martini` was
            initialised with ``quiet`` set to ``True``, progress bars are switched off
            unless explicitly turned on.

        ncpu : int
            Number of threads to use in main source insertion loop. Using more than one
            thread requires the :mod:`numba` module. Can be set to ``-1`` to use as many
            threads as available cores. Currently speedup from multiple cores is limited,
            see full documentation for details.

        mem_lim_GB : float
            The peak memory usage can get very large if many particle kernels touch many
            pixels in the cube. The particles can be processed in batches to mitigate
            this, but this slows down the code. The memory limit set here (in GB) will
            trigger batching if estimated memory usage is expected to exceed it. This is
            a "soft" limit since some allocations (e.g. temporary arrays when evaluating
            the SPH kernel) cannot be exactly predicted.
        """
        super()._insert_source_in_cube(
            skip_validation=skip_validation,
            progressbar=progressbar,
            ncpu=ncpu,
            mem_lim_GB=mem_lim_GB,
        )

        return

    def convolve_beam(self) -> None:
        """Convolve the beam and data cube."""
        if self.beam is None:
            warn("Skipping beam convolution, no beam object provided to Martini.")
            return

        minimum_padding = self.beam.needs_pad()
        if (self._datacube.padx < minimum_padding[0]) or (
            self._datacube.pady < minimum_padding[1]
        ):
            raise ValueError(
                "datacube padding insufficient for beam convolution (perhaps you loaded a"
                " datacube state with datacube.load_state that was previously initialized"
                " by martini with a smaller beam?)"
            )

        with self._logger.step(
            "Convolving beam...", "Beam convolved", quiet=self.quiet
        ):
            unit = self._datacube.current_units
            assert self.beam.kernel is not None  # placate mypy
            for spatial_slice in self._datacube.spatial_slices:
                # use a view to force in-place modification
                spatial_slice[...] = (
                    fftconvolve(spatial_slice, self.beam.kernel, mode="same") * unit
                )
            self._datacube.drop_pad()
            self._datacube._array = _to_unit_in_place(
                self._datacube._array,
                U.Jy * U.beam**-1,
                equivalencies=U.beam_angular_area(self.beam.area),
            )
        if not self.quiet:
            self._logger.message(
                "Data cube RMS after beam convolution:"
                f" {np.std(self._datacube._array):.2e}",
                f"Maximum pixel: {self._datacube._array.max():.2e}",
                "Median non-zero pixel:"
                f" {np.median(self._datacube._array[self._datacube._array > 0]):.2e}",
                extra_indent=1,
            )
        return

    def add_noise(self) -> None:
        """
        Insert noise into the data cube.

        The units of the noise realization determine when it should be added. Noise
        with units of flux density per unit area (e.g. Jy/arcsec^2, or the current units
        of the data cube before beam convolution) is added before
        :meth:`~martini.martini.Martini.convolve_beam` and is then convolved with the
        beam. Noise with units of flux density per beam (e.g. Jy/beam) is treated as
        already convolved with the beam and is added after
        :meth:`~martini.martini.Martini.convolve_beam`.

        Raises
        ------
        ValueError
            If noise per unit area is added after beam convolution, or noise per beam is
            added before beam convolution.
        """
        if self.noise is None:
            warn("Skipping noise, no noise object provided to Martini.")
            return

        if self.beam is None:
            warn(
                "Skipping noise, no beam object (required to estimate post-convolution"
                " noise level) provided to Martini."
            )
            return

        cube_convolved = self._datacube.current_units.is_equivalent(U.Jy / U.beam)
        with self._logger.step("Adding noise...", "Noise added", quiet=self.quiet):
            noise_cube = self.noise.generate(self._datacube, self.beam)
            noise_convolved = noise_cube.unit.is_equivalent(U.Jy / U.beam)
            if noise_convolved and not cube_convolved:
                raise ValueError(
                    f"Noise in units of {noise_cube.unit} is treated as already "
                    "convolved with the beam, so it must be added after the beam "
                    "convolution: call `add_noise` after `convolve_beam`."
                )
            if cube_convolved and not noise_convolved:
                raise ValueError(
                    f"Noise in units of {noise_cube.unit} is treated as not yet "
                    "convolved with the beam, so it must be added before the beam "
                    "convolution: call `add_noise` before `convolve_beam`."
                )
            if noise_cube.unit != self._datacube.current_units:
                # a single scaling (no equivalency needed between per-beam units, and
                # only the pixel area between per-area units), done in place to avoid
                # copies of the noise cube:
                noise_cube = _to_unit_in_place(
                    noise_cube,
                    self._datacube.current_units,
                    equivalencies=[self._datacube.arcsec2_to_pix],
                )
            self._datacube._array += noise_cube
        if not self.quiet:
            when = "after" if cube_convolved else "before"
            self._logger.message(
                f"Noise cube RMS: {np.std(noise_cube):.2e} ({when} beam convolution).",
                f"Data cube RMS after noise addition ({when} beam convolution): "
                f"{np.std(self._datacube._array):.2e}",
                extra_indent=1,
            )
        return

    def write_fits(
        self,
        filename: str,
        overwrite: bool = True,
        obj_name: str = "MOCK",
        dtype: str | np.dtype | None = None,
    ) -> None:
        """
        Output the data cube to a FITS-format file.

        Parameters
        ----------
        filename : str
            Name of the file to write. ``'.fits'`` will be appended if not already
            present in the filename. ``'.fits.gz'`` etc. will be recognized.

        overwrite : bool, optional
            Whether to allow overwriting existing files.

        obj_name : str
            Name to write in the ``OBJECT`` FITS header field (max 16 characters).

        dtype : str or dtype
            Typecode or data-type to which the array is cast. Should be supported
            by fits. Types not stored natively in FITS (e.g. unsigned integers other
            than ``uint8``) are converted by :mod:`astropy`, which needs more memory.
            Default to not do data type conversion.
        """
        with self._logger.step(
            "Writing data cube to FITS file...",
            "Wrote FITS file",
            quiet=self.quiet,
        ) as step:
            self._datacube.drop_pad()

            filename = str(filename)
            filename = (
                filename + ".fits" if "fits" not in filename.split(".") else filename
            )
            step.end = f"Wrote FITS file {filename}"

            wcs_header = self._datacube.wcs.to_header()
            wcs_header.rename_keyword("WCSAXES", "NAXIS")

            header = fits.Header()
            header.append(("SIMPLE", "T"))
            header.append(("BITPIX", 16))
            header.append(("NAXIS", wcs_header["NAXIS"]))
            header.append(("NAXIS1", self._datacube.n_px_x))
            header.append(("NAXIS2", self._datacube.n_px_y))
            header.append(("NAXIS3", self._datacube.n_channels))
            if self._datacube.stokes_axis:
                header.append(("NAXIS4", 1))
            header.append(("EXTEND", "T"))
            header.append(("CDELT1", wcs_header["CDELT1"]))
            header.append(("CRPIX1", wcs_header["CRPIX1"]))
            header.append(("CRVAL1", wcs_header["CRVAL1"]))
            header.append(("CTYPE1", wcs_header["CTYPE1"]))
            header.append(("CUNIT1", wcs_header["CUNIT1"]))
            header.append(("CDELT2", wcs_header["CDELT2"]))
            header.append(("CRPIX2", wcs_header["CRPIX2"]))
            header.append(("CRVAL2", wcs_header["CRVAL2"]))
            header.append(("CTYPE2", wcs_header["CTYPE2"]))
            header.append(("CUNIT2", wcs_header["CUNIT2"]))
            header.append(("CDELT3", wcs_header["CDELT3"]))
            header.append(("CRPIX3", wcs_header["CRPIX3"]))
            header.append(("CRVAL3", wcs_header["CRVAL3"]))
            header.append(("CTYPE3", wcs_header["CTYPE3"]))
            header.append(("CUNIT3", wcs_header["CUNIT3"]))
            if self._datacube.stokes_axis:
                header.append(("CDELT4", wcs_header["CDELT4"]))
                header.append(("CRPIX4", wcs_header["CRPIX4"]))
                header.append(("CRVAL4", wcs_header["CRVAL4"]))
                header.append(("CTYPE4", wcs_header["CTYPE4"]))
                header.append(("CUNIT4", "PAR"))
            header.append(("EPOCH", 2000))
            header.append(("INSTRUME", "MARTINI", martini_version))
            header.append(("BSCALE", 1.0))
            header.append(("BZERO", 0.0))
            datacube_array_units = self._datacube.current_units
            header.append(
                (
                    "DATAMAX",
                    np.max(self._datacube._array.to_value(datacube_array_units)),
                )
            )
            header.append(
                (
                    "DATAMIN",
                    np.min(self._datacube._array.to_value(datacube_array_units)),
                )
            )
            header.append(("ORIGIN", "astropy v" + astropy_version))
            # long names break fits format, don't let the user set this
            if len(obj_name) > 16:
                warn("obj_name longer than 16 characters, truncating", RuntimeWarning)
                obj_name = obj_name[:16]
            header.append(("OBJECT", obj_name))
            if self.beam is not None:
                header.append(("BPA", self.beam.bpa.to_value(U.deg)))
            header.append(("OBSERVER", "K. Oman"))
            header.append(("BUNIT", datacube_array_units.to_string("fits")))
            header.append(("DATE-OBS", Time.now().to_value("fits")))
            header.append(("MJD-OBS", Time.now().to_value("mjd")))
            if self.beam is not None:
                header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
                header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
            header.append(("BTYPE", "Intensity"))
            header.append(("SPECSYS", wcs_header["SPECSYS"]))
            header.append(("RESTFRQ", wcs_header["RESTFRQ"]))

            # flip axes to write
            data = self._datacube._array.to_value(datacube_array_units).T
            out_dtype = np.dtype(dtype) if dtype is not None else data.dtype
            if out_dtype.name not in _FITS_BITPIX:
                # let astropy handle conversion
                hdu = fits.PrimaryHDU(
                    header=header, data=data.astype(out_dtype, copy=False)
                )
                hdu.writeto(filename, overwrite=overwrite)
                return
            # Avoid byteswapping entire array in-place twice by streaming:
            header = fits.PrimaryHDU(header=header, data=data).header.copy()
            header["BITPIX"] = _FITS_BITPIX[out_dtype.name]
            if os.path.exists(filename):
                if not overwrite:
                    raise OSError(
                        f"File {filename} exists. To overwrite set 'overwrite=True'."
                    )
                os.remove(filename)
            slab = np.empty(data.shape[-2:], dtype=out_dtype.newbyteorder(">"))
            with fits.StreamingHDU(filename, header) as shdu:
                for index in np.ndindex(data.shape[:-2]):
                    slab[...] = data[index]
                    shdu.write(slab)

            return

    def write_beam_fits(
        self,
        filename: str,
        overwrite: bool = True,
    ) -> None:
        """
        Output the beam to a FITS-format file.

        The beam is written to file, with pixel sizes, coordinate system, etc.
        similar to those used for the :class:`~martini.datacube.DataCube`.

        Parameters
        ----------
        filename : str
            Name of the file to write. ``".fits"`` will be appended if not already
            present.

        overwrite : bool, optional
            Whether to allow overwriting existing files.

        Raises
        ------
        ValueError
            If :class:`~martini.martini.Martini` was initialized without a ``beam``.
        """
        with self._logger.step(
            "Writing beam image to FITS file...",
            "Wrote FITS file",
            quiet=self.quiet,
        ) as step:
            if self.beam is None:
                raise ValueError(
                    "Martini.write_beam_fits: Called with beam set to 'None'."
                )
            assert self.beam.kernel is not None

            filename = filename if filename[-5:] == ".fits" else filename + ".fits"
            step.end = f"Wrote FITS file {filename}"

            wcs_header = self._datacube.wcs.to_header()

            beam_kernel_units = self.beam.kernel.unit
            header = fits.Header()
            header.append(("SIMPLE", "T"))
            header.append(("BITPIX", 16))
            header.append(("NAXIS", 3))
            header.append(("NAXIS1", self.beam.kernel.shape[0]))
            header.append(("NAXIS2", self.beam.kernel.shape[1]))
            header.append(("NAXIS3", 1))
            header.append(("EXTEND", "T"))
            header.append(("BSCALE", 1.0))
            header.append(("BZERO", 0.0))
            # this is 1/arcsec^2, is this right?
            header.append(("BUNIT", beam_kernel_units.to_string("fits")))
            header.append(("CRPIX1", self.beam.kernel.shape[0] // 2 + 1))
            header.append(("CDELT1", wcs_header["CDELT1"]))
            header.append(("CRVAL1", wcs_header["CRVAL1"]))
            header.append(("CTYPE1", wcs_header["CTYPE1"]))
            header.append(("CUNIT1", wcs_header["CUNIT1"]))
            header.append(("CRPIX2", self.beam.kernel.shape[1] // 2 + 1))
            header.append(("CDELT2", wcs_header["CDELT2"]))
            header.append(("CRVAL2", wcs_header["CRVAL2"]))
            header.append(("CTYPE2", wcs_header["CTYPE2"]))
            header.append(("CUNIT2", wcs_header["CUNIT2"]))
            header.append(("CRPIX3", 1))
            header.append(("CDELT3", wcs_header["CDELT3"]))
            header.append(("CRVAL3", wcs_header["CRVAL3"]))
            header.append(("CTYPE3", wcs_header["CTYPE3"]))
            header.append(("CUNIT3", wcs_header["CUNIT3"]))
            header.append(("SPECSYS", wcs_header["SPECSYS"]))
            header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
            header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
            header.append(("BPA", self.beam.bpa.to_value(U.deg)))
            header.append(("BTYPE", "beam    "))
            header.append(("EPOCH", 2000))
            header.append(("OBSERVER", "K. Oman"))
            # long names break fits format
            header.append(("OBJECT", "MOCKBEAM"))
            header.append(("INSTRUME", "MARTINI", martini_version))
            header.append(
                ("DATAMAX", np.max(self.beam.kernel.to_value(beam_kernel_units)))
            )
            header.append(
                ("DATAMIN", np.min(self.beam.kernel.to_value(beam_kernel_units)))
            )
            header.append(("ORIGIN", "astropy v" + astropy_version))

            # flip axes to write
            hdu = fits.PrimaryHDU(
                header=header,
                data=self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis].T,
            )
            hdu.writeto(filename, overwrite=True)

            return

    def write_hdf5(
        self,
        filename: str,
        overwrite: bool = True,
        memmap: bool = False,
        compact: bool = False,
    ) -> "h5py.File | None":
        """
        Output the data cube and beam to a HDF5-format file.

        Requires the :mod:`h5py` package.

        Parameters
        ----------
        filename : str
            Name of the file to write. ``'.hdf5'`` will be appended if not already
            present.

        overwrite : bool, optional
            Whether to allow overwriting existing files.

        memmap : bool, optional
            If ``True``, create a file-like object in memory and return it instead
            of writing file to disk.

        compact : bool, optional
            If ``True``, omit pixel coordinate arrays to save disk space. In this
            case pixel coordinates can still be reconstructed from FITS-style
            keywords stored in the FluxCube attributes.
        """
        with self._logger.step(
            "Writing data cube to HDF5 file...",
            "Wrote HDF5 file",
            quiet=self.quiet,
        ) as step:
            import h5py

            self._datacube.drop_pad()

            filename = filename if filename[-5:] == ".hdf5" else filename + ".hdf5"
            step.end = f"Wrote HDF5 file {filename}"

            wcs_header = self._datacube.wcs.to_header()

            mode = "w" if overwrite else "x"
            driver = "core" if memmap else None
            h5_kwargs = {"backing_store": False} if memmap else {}
            f = h5py.File(filename, mode, driver=driver, **h5_kwargs)
            datacube_array_units = self._datacube.current_units
            f["FluxCube"] = self._datacube._array.to_value(
                datacube_array_units
            ).squeeze()
            c = f["FluxCube"]
            origin = 0  # index from 0 like numpy, not from 1
            if not compact:
                # voxel centre coordinates:
                xgrid, ygrid, vgrid = np.meshgrid(
                    np.arange(self._datacube.datacube_shape[0]),
                    np.arange(self._datacube.datacube_shape[1]),
                    np.arange(self._datacube.datacube_shape[2]),
                    indexing="ij",
                )
                cgrid = (
                    np.vstack(
                        (
                            xgrid.flatten(),
                            ygrid.flatten(),
                            vgrid.flatten(),
                            np.zeros(vgrid.shape).flatten(),
                        )
                    ).T
                    if self._datacube.stokes_axis
                    else np.vstack(
                        (
                            xgrid.flatten(),
                            ygrid.flatten(),
                            vgrid.flatten(),
                        )
                    ).T
                )
                wgrid = self._datacube.wcs.all_pix2world(cgrid, origin)
                grid_shape = self._datacube.datacube_shape[:3]
                ragrid = wgrid[:, 0].reshape(grid_shape)
                decgrid = wgrid[:, 1].reshape(grid_shape)
                chgrid = wgrid[:, 2].reshape(grid_shape)
                f["RA"] = ragrid
                f["RA"].attrs["Unit"] = wcs_header["CUNIT1"]
                f["Dec"] = decgrid
                f["Dec"].attrs["Unit"] = wcs_header["CUNIT2"]
                f["channel_mids"] = chgrid
                f["channel_mids"].attrs["Unit"] = wcs_header["CUNIT3"]
                # voxel vertex coordinates (for e.g. pyplot.pcolormesh):
                xgrid_vertices, ygrid_vertices, vgrid_vertices = np.meshgrid(
                    np.arange(self._datacube.datacube_shape[0] + 1) - 0.5,
                    np.arange(self._datacube.datacube_shape[1] + 1) - 0.5,
                    np.arange(self._datacube.datacube_shape[2] + 1) - 0.5,
                    indexing="ij",
                )
                cgrid_vertices = (
                    np.vstack(
                        (
                            xgrid_vertices.flatten(),
                            ygrid_vertices.flatten(),
                            vgrid_vertices.flatten(),
                            np.zeros(vgrid_vertices.shape).flatten(),
                        )
                    ).T
                    if self._datacube.stokes_axis
                    else np.vstack(
                        (
                            xgrid_vertices.flatten(),
                            ygrid_vertices.flatten(),
                            vgrid_vertices.flatten(),
                        )
                    ).T
                )
                wgrid_vertices = self._datacube.wcs.all_pix2world(
                    cgrid_vertices, origin
                )
                vertices_grid_shape = [
                    shape_i + 1 for shape_i in self._datacube.datacube_shape[:3]
                ]
                ragrid_vertices = wgrid_vertices[:, 0].reshape(vertices_grid_shape)
                decgrid_vertices = wgrid_vertices[:, 1].reshape(vertices_grid_shape)
                chgrid_vertices = wgrid_vertices[:, 2].reshape(vertices_grid_shape)
                f["RA_vertices"] = ragrid_vertices
                f["RA_vertices"].attrs["Unit"] = wcs_header["CUNIT1"]
                f["Dec_vertices"] = decgrid_vertices
                f["Dec_vertices"].attrs["Unit"] = wcs_header["CUNIT2"]
                f["channel_vertices"] = chgrid_vertices
                f["channel_vertices"].attrs["Unit"] = wcs_header["CUNIT3"]
                # channels:
                for dataset_name in (
                    "velocity_channel_mids",
                    "velocity_channel_edges",
                    "frequency_channel_mids",
                    "frequency_channel_edges",
                ):
                    f[dataset_name] = getattr(self._datacube, dataset_name)
                    f[dataset_name].attrs["Unit"] = str(
                        getattr(self._datacube, dataset_name).unit
                    )
            c.attrs["AxisOrder"] = "(RA,Dec,Channels)"
            c.attrs["FluxCubeUnit"] = str(self._datacube.current_units)
            c.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
            c.attrs["RA0_in_px"] = wcs_header["CRPIX1"] - 1
            c.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
            c.attrs["RAUnit"] = wcs_header["CUNIT1"]
            c.attrs["RAProjType"] = wcs_header["CTYPE1"]
            c.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
            c.attrs["Dec0_in_px"] = wcs_header["CRPIX2"] - 1
            c.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
            c.attrs["DecUnit"] = wcs_header["CUNIT2"]
            c.attrs["DecProjType"] = wcs_header["CTYPE2"]
            c.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
            c.attrs["V0_in_px"] = wcs_header["CRPIX3"] - 1
            c.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
            c.attrs["VUnit"] = wcs_header["CUNIT3"]
            c.attrs["VProjType"] = wcs_header["CTYPE3"]
            c.attrs["SpecSys"] = wcs_header["SPECSYS"]
            if self.beam is not None:
                c.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
                c.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
                c.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
            c.attrs["DateCreated"] = str(Time.now())
            c.attrs["MartiniVersion"] = martini_version
            c.attrs["AstropyVersion"] = astropy_version
            if self.beam is not None:
                if self.beam.kernel is None:
                    raise ValueError(
                        "Martini.write_hdf5: Called with beam present but beam kernel"
                        " uninitialized."
                    )
                beam_kernel_units = self.beam.kernel.unit
                f["Beam"] = self.beam.kernel.to_value(beam_kernel_units)[
                    ..., np.newaxis
                ]
                b = f["Beam"]
                b.attrs["BeamUnit"] = self.beam.kernel.unit.to_string("fits")
                b.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
                b.attrs["RA0_in_px"] = self.beam.kernel.shape[0] // 2
                b.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
                b.attrs["RAUnit"] = wcs_header["CUNIT1"]
                b.attrs["RAProjType"] = wcs_header["CTYPE1"]
                b.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
                b.attrs["Dec0_in_px"] = self.beam.kernel.shape[1] // 2
                b.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
                b.attrs["DecUnit"] = wcs_header["CUNIT2"]
                b.attrs["DecProjType"] = wcs_header["CTYPE2"]
                b.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
                b.attrs["V0_in_px"] = 0
                b.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
                b.attrs["VUnit"] = wcs_header["CUNIT3"]
                b.attrs["VProjType"] = wcs_header["CTYPE3"]
                b.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
                b.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
                b.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
                b.attrs["DateCreated"] = str(Time.now())
                b.attrs["MartiniVersion"] = martini_version
                b.attrs["AstropyVersion"] = astropy_version

            if memmap:
                return f
            else:
                f.close()
                return None


class GlobalProfile(_BaseMartini):
    """
    Just produce a global spectrum of an HI source.

    A simplified version of the main :class:`~martini.martini.Martini` class for those
    times when you just want a spectrum.

    Sometimes only the spatially integrated spectrum of a source is wanted, which makes
    many parts of the standard MARTINI process unnecessary. This class streamlines the
    process of producing a spatially-integrated spectrum, or "global profile", with some
    specific assumptions:

     - The :class:`~martini.martini.GlobalProfile` class does not assume any spatial
       aperture, instead every particle in the source contributes to the spectrum (unless
       it falls outside of the spectral bandwidth).
     - The positions of particles are still used to calculate the line-of-sight vector
       and the velocity along this direction.

    There is therefore no need or way to specify a beam or SPH kernel as with the main
    :class:`~martini.martini.Martini` class. It is also not possible to use MARTINI's
    noise modules with this class. If these restrictions are found to be too limiting
    (for example, if the spectrum within a spatial mask defined by a signal-to-noise or
    other cut is desired), the best course of action is to produce a spatially-resolved
    mock observation and derive the spectrum from those data as would be done with "real"
    observations. This class is mainly intended to efficiently provide a "quick look" at
    the spectrum, or a reference "ideal" spectrum.

    Parameters
    ----------
    source : ~martini.sources.sph_source.SPHSource
        An instance of a class derived from
        :class:`~martini.sources.sph_source.SPHSource`.
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). See :doc:`sub-module documentation </sources/index>`.

    spectral_model : ~martini.spectral_models._BaseSpectrum
        An instance of a class derived from
        :class:`~martini.spectral_models._BaseSpectrum`.
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See
        :doc:`sub-module documentation </spectral_models/index>`.

    n_channels : int
        Number of channels along the spectral axis.

    channel_width : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of velocity or frequency.
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency.

    spectral_centre : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity` with dimensions of velocity or frequency.
        Velocity (or frequency) of the centre along the spectral axis. Defaults to 0 km/s,
        but should normally be set to the systemic velocity of the source.

    quiet : bool, optional
        If ``True``, suppress output to stdout.

    See Also
    --------
    martini.sources.sph_source.SPHSource
    martini.spectral_models
    martini.martini.Martini

    Examples
    --------
    ::

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

        spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

        M = GlobalProfile(
            source=source,
            spectral_model=spectral_model,
            n_channels=32,
            channel_width=10 * U.km * U.s**-1,
            spectral_centre=source.vsys,
        )

        # spectrum and channel edges and centres can be accessed as:
        M.spectrum  # spectrum is evaluated at first access if not already done
        M.channel_edges
        M.channel_mids

        # the spectrum can be explicitly evaluated with:
        M.insert_source_in_spectrum()
    """

    _target_name = "spectrum"

    @U.quantity_input
    def __init__(
        self,
        *,
        source: SPHSource,
        spectral_model: _BaseSpectrum,
        n_channels: int,
        channel_width: U.Quantity[U.km / U.s] | U.Quantity[U.Hz],
        spectral_centre: U.Quantity[U.km / U.s] | U.Quantity[U.Hz] = 0 * U.km * U.s**-1,
        quiet: bool = False,
    ) -> None:
        super().__init__(
            source=source,
            datacube=_GlobalProfileDataCube(
                n_channels=n_channels,
                channel_width=channel_width,
                spectral_centre=spectral_centre,
            ),
            beam=None,
            noise=None,
            sph_kernel=DiracDeltaKernel(size_in_fwhm=np.inf),
            spectral_model=spectral_model,
            _prune_kwargs={
                "spatial": False,
                "spectral": True,
                "mass": True,
            },
            quiet=quiet,
        )
        self.source.pixcoords[:2] = 0

        return

    def _count_px_hits(self, sm_ranges: np.ndarray, ncpu: int) -> np.ndarray:
        """
        Count the pixels overlapped by each particle's smoothing kernel.

        For :class:`~martini.martini.GlobalProfile` every particle is assumed to
        overlap the (single) pixel, so no search is needed.

        Parameters
        ----------
        sm_ranges : ~numpy.ndarray
            Smoothing kernel ranges of the particles, in pixels (unused).

        ncpu : int
            Number of threads to use (unused).

        Returns
        -------
        ~numpy.ndarray
            Number of pixels overlapped by each particle.
        """
        return np.ones(self.source.npart, dtype=np.int64)

    def _find_px_hits(
        self,
        segment: slice,
        sm_ranges: np.ndarray,
        ncpu: int,
        return_distances: bool = True,
    ) -> FindGridIntersectionsResult:
        """
        Find the pixels overlapped by the smoothing kernels of a batch of particles.

        For :class:`~martini.martini.GlobalProfile` every particle is treated as
        overlapping the single pixel at zero offset. Their line of sight velocities
        are still correctly projected.

        Parameters
        ----------
        segment : slice
            Selects the batch of particles.

        sm_ranges : ~numpy.ndarray
            Smoothing kernel ranges of all particles, in pixels (unused).

        ncpu : int
            Number of threads to use (unused).

        return_distances : bool
            If ``False``, offsets between particles and pixel centres are not returned.

        Returns
        -------
        ~martini._grid_search.FindGridIntersectionsResult
            The particle-pixel overlaps.
        """
        n = len(range(self.source.npart)[segment])
        if n == 0:
            return FindGridIntersectionsResult(
                intersections=np.empty(0),
                distances=np.empty(0) if return_distances else None,
                cell_indices=np.empty(0),
                strides=np.empty((0, 2)),
            )
        return FindGridIntersectionsResult(
            intersections=np.arange(n, dtype=np.int32),
            distances=(
                np.zeros((n, 2), dtype=self.spectral_model.spec_dtype)
                if return_distances
                else None
            ),
            cell_indices=np.zeros(1, dtype=np.int64),
            strides=np.array([[0, n]], dtype=np.int64),
        )

    def insert_source_in_spectrum(
        self, mem_lim_GB: float = 4.0, quiet: bool | None = None
    ) -> None:
        """
        Populate the :class:`~martini.datacube.DataCube` with flux from source particles.

        For the :class:`~martini.martini.GlobalProfile` class we assume that every
        particle in the source contributes (if it falls in the spectral bandwidth)
        regardless of  position on the sky. The line-of-sight vector still depends on
        the particle positions, so the direction to the individual particles is still
        taken into account.

        Parameters
        ----------
        mem_lim_GB : float
            The peak memory usage can get very large if many particle kernels touch many
            pixels in the cube. The particles can be processed in batches to mitigate
            this, but this slows down the code. The memory limit set here (in GB) will
            trigger batching if estimated memory usage is expected to exceed it. This is
            a "soft" limit since some allocations (e.g. temporary arrays when evaluating
            the SPH kernel) cannot be exactly predicted.

        quiet : bool, optional
            If ``True``, suppress output to stdout. If specified, takes precedence over
            ``quiet`` parameter of class.
        """
        quiet = self.quiet if quiet is None else quiet
        # skip_validation=True: all particles can contribute their kernel to the pixel;
        # ncpu=1 since we have 1 pixel and source insertion is parallel over pixels;
        # no progressbar since there's only 1 pixel of progress;
        super()._insert_source_in_cube(
            skip_validation=True,
            progressbar=False,
            ncpu=1,
            mem_lim_GB=mem_lim_GB,
            quiet=quiet,
            _no_summary=True,
        )
        # The datacube in Jy/arcsec^2 is a bit misleading because the source is
        # (presumably) completely unresolved so extrapolating its surface brightness
        # across the entire pixel is incorrect. Correctly integrate out spatial
        # information and convert to Jy:
        self._spectrum = (
            (self._datacube._array.squeeze()).to(
                U.Jy / U.pix**2, equivalencies=[self._datacube.arcsec2_to_pix]
            )
            * U.pix**2
        ).to(U.Jy)
        if not quiet:
            # Need a slightly different calculation for a completely unresolved source.
            inserted_flux_density = self.spectrum.sum()
            inserted_mass = (
                2.36e5
                * U.Msun
                * self.source.distance.to_value(U.Mpc) ** 2
                * inserted_flux_density.to_value(U.Jy)
                * self._datacube.channel_width.to_value(U.km / U.s)
            )
            self._logger.message(
                f"Flux density in spectrum: {inserted_flux_density:.2e}",
                f"Mass in spectrum (assuming distance {self.source.distance:.2f}):"
                f" {inserted_mass:.2e}",
                f"  [{inserted_mass / self.source.input_mass * 100:.0f}%"
                " of initial source mass]",
                extra_indent=1,
            )

    @property
    @U.quantity_input
    def spectrum(self) -> U.Quantity[U.Jy]:
        """
        The spectrum of the source with spatial information integrated out.

        The spectrum is calculated when this property is accessed if this hasn't already
        been done.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of flux density.
            Spatially-integrated spectrum of the source.
        """
        if not hasattr(self, "_spectrum"):
            self.insert_source_in_spectrum()
        return self._spectrum

    @property
    @U.quantity_input
    def channel_edges(self) -> U.Quantity:
        """
        The edges of the channels for the spectrum.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency or velocity.
            Edges of the channels with units depending on
            :class:`~martini.martini.GlobalProfile`'s native channel spacing.

        See Also
        --------
        martini.martini.GlobalProfile.channel_mids
        martini.martini.GlobalProfile.velocity_channel_mids
        martini.martini.GlobalProfile.velocity_channel_edges
        martini.martini.GlobalProfile.frequency_channel_mids
        martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.channel_edges

    @property
    @U.quantity_input
    def channel_mids(self) -> U.Quantity:
        """
        The centres of the channels for the spectrum.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency or velocity.
            Edges of the channels with units depending on
            :class:`~martini.martini.GlobalProfile`'s native channel spacing.

        See Also
        --------
        martini.martini.GlobalProfile.channel_edges
        martini.martini.GlobalProfile.velocity_channel_mids
        martini.martini.GlobalProfile.velocity_channel_edges
        martini.martini.GlobalProfile.frequency_channel_mids
        martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.channel_mids

    @property
    @U.quantity_input
    def frequency_channel_edges(self) -> U.Quantity[U.Hz]:
        """
        The edges of the frequency channels for the spectrum.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency.
            Edges of the channels with frequency units.

        See Also
        --------
        martini.martini.GlobalProfile.velocity_channel_mids
        martini.martini.GlobalProfile.velocity_channel_edges
        martini.martini.GlobalProfile.frequency_channel_mids
        """
        return self._datacube.frequency_channel_edges

    @property
    @U.quantity_input
    def frequency_channel_mids(self) -> U.Quantity[U.Hz]:
        """
        The centres of the frequency channels for the spectrum.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of frequency.
            Edges of the channels with frequency units.

        See Also
        --------
        martini.martini.GlobalProfile.velocity_channel_mids
        martini.martini.GlobalProfile.velocity_channel_edges
        martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.frequency_channel_mids

    @property
    @U.quantity_input
    def velocity_channel_edges(self) -> U.Quantity[U.km / U.s]:
        """
        The edges of the channels for the spectrum in velocity units.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of velocity.
            Edges of the channels with velocity units.

        See Also
        --------
        martini.martini.GlobalProfile.velocity_channel_mids
        martini.martini.GlobalProfile.frequency_channel_mids
        martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.velocity_channel_edges

    @property
    @U.quantity_input
    def velocity_channel_mids(self) -> U.Quantity[U.km / U.s]:
        """
        The centres of the channels for the spectrum.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimensions of velocity.
            Edges of the channels with velocity units.

        See Also
        --------
        martini.martini.GlobalProfile.velocity_channel_edges
        martini.martini.GlobalProfile.frequency_channel_mids
        martini.martini.GlobalProfile.frequency_channel_edges
        """
        return self._datacube.velocity_channel_mids

    @property
    @U.quantity_input
    def channel_width(self) -> U.Quantity:
        """
        The width of the channels for the spectrum.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with dimentions of frequency or velocity.
            Width of the channels with units depending on
            :class:`~martini.martini.GlobalProfile`'s ``channels`` argument.
        """
        return self._datacube.channel_width

    def reset(self) -> None:
        """Re-initialize the spectrum with zero-values."""
        super().reset()
        if hasattr(self, "_spectrum"):
            del self._spectrum
        return

    def plot_spectrum(
        self,
        fig: int = 1,
        title: str = "",
        channels: str = "velocity",
        show_vsys: bool = True,
        save: str | None = None,
    ) -> "Figure":
        """
        Produce a figure showing the spectrum.

        Parameters
        ----------
        fig : int, optional
            Number of the figure in matplotlib, it will be created as ``plt.figure(fig)``.

        title : str, optional
            A title for the figure can be provided.

        channels : str, optional
            The type of spectral axis for the plot, either ``"velocity"`` or
            ``"frequency"``.

        show_vsys : bool, optional
            If ``True``, draw a vertical line at the source systemic velocity.

        save : str, optional
            If provided, the figure is saved using ``plt.savefig(save)``. A `.png` or
            `.pdf` suffix is recommended.

        Returns
        -------
        matplotlib.figure.Figure
            The spectrum :class:`~matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt

        if channels == "velocity":
            channel_mids = self.velocity_channel_mids
        elif channels == "frequency":
            channel_mids = self.frequency_channel_mids
        else:
            raise ValueError(
                "`plot_spectrum` argument `channels` must be 'velocity' or 'frequency'."
            )

        figure = plt.figure(fig, figsize=(4, 3))
        figure.clf()
        figure.suptitle(title)

        xunit = {"velocity": U.km * U.s**-1, "frequency": U.GHz}[channels]

        ax = figure.add_subplot(1, 1, 1)
        ax.plot(
            channel_mids.to_value(xunit),
            self.spectrum.to_value(U.Jy),
            ls="solid",
            color="black",
            lw=2,
        )
        if show_vsys:
            ax.axvline(
                self.source.vsys.to_value(xunit, equivalencies=U.doppler_radio(HIfreq)),
                linestyle="dotted",
                lw=1.5,
                color="black",
            )
        ax.set_ylabel(r"Flux density $[\mathrm{Jy}]$")
        if channels == "velocity":
            ax.set_xlabel(r"Velocity $[\mathrm{km}\,\mathrm{s}^{-1}]$")
        elif channels == "frequency":
            ax.set_xlabel(r"Frequency $[\mathrm{GHz}]$")

        if save is not None:
            plt.savefig(save)
        return figure
