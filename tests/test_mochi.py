"""Tests of the mochi sub-module."""

from martini.mochi.mochi import Mochi, _coarse_shape
from martini.datacube import DataCube
from martini.sources import SPHSource
from martini.beams import GaussianBeam
from martini.spectral_models import GaussianSpectrum, DiracDeltaSpectrum
from martini.sph_kernels import (
    CubicSplineKernel,
    DiracDeltaKernel,
    _CubicSplineKernel,
    _GaussianKernel,
)
import astropy.units as U
from martini.mochi.mochi import AdaptiveCellGrid, CellGrid
from martini.mochi import interpolants as module_interpolants
from martini.mochi import refinement
from martini.mochi._dtypes import CELL_DTYPE
from martini._grid_search import Octree
from martini.mochi.radiative_transfer import (
    optically_thin,
    _los_overlap,
    calculate_field_spectrum,
)
import pytest
import numpy as np
import itertools
from types import SimpleNamespace
from test_martini import check_mass_accuracy
from test_sph_kernels import simple_kernels

interpolants = (
    module_interpolants.sph,
    module_interpolants.mfm,
    module_interpolants.voronoi_mesh,
    module_interpolants.manual_sph,
    module_interpolants.particle_in_cell,
)
refinement_strategies = (
    refinement.refine_grid_to_half_particle_scale,
    refinement.refine_grid_to_single_occupancy,
)
interpolant_ids = ("sph", "mfm", "voronoi_mesh", "manual_sph", "particle_in_cell")
spectral_models = (GaussianSpectrum, DiracDeltaSpectrum)


class _ArbitraryCellGrid(CellGrid):
    """A uniform cell grid that does not identify itself as one, for testing."""

    @property
    def uniform_grid(self) -> None:
        """
        Do not describe the grid as uniform.

        Returns
        -------
        None
            Always ``None``.
        """
        return None

    @property
    def octree(self) -> Octree:
        """
        Build the tree of the cells, as for an adaptive grid.

        Returns
        -------
        ~martini._grid_search.Octree
            The tree of the cells.
        """
        return Octree(
            self.field_positions.to_value(U.pix), self.field_sizes.to_value(U.pix)
        )


def _initial_cells(shape, size):
    """
    Make a uniform grid of initial cells, starting at ``-0.5`` along each axis.

    Parameters
    ----------
    shape : tuple
        Number of cells along each axis.

    size : float
        Side length of the cells.

    Returns
    -------
    ~numpy.ndarray
        The lower corners and sizes of the cells.
    """
    return np.array(
        [
            tuple(np.array(ijk) * size - 0.5) + (size,)
            for ijk in itertools.product(*map(range, shape))
        ],
        dtype=CELL_DTYPE,
    )


def _reference_refinement(cells, positions, radii, keep, max_count, stop_iter):
    """
    Refine cells by recursive bisection, one cell at a time, with numpy.

    Parameters
    ----------
    cells : ~numpy.ndarray
        The initial cells.

    positions : ~numpy.ndarray
        Particle positions.

    radii : ~numpy.ndarray
        Particle radii.

    keep : Callable
        Given the positions and radii of candidate particles, a cell corner and size,
        decide which candidates count towards bisecting the cell.

    max_count : int
        Cells where more particles count are bisected.

    stop_iter : int
        Maximum number of bisections.

    Returns
    -------
    set
        The refined cells, as tuples of lower corner coordinates and size.
    """
    todo = [
        (np.array([c["x"], c["y"], c["z"]]), c["size"], 0, np.arange(len(radii)))
        for c in cells
    ]
    refined = set()
    while todo:
        corner, size, level, candidates = todo.pop()
        kept = candidates[keep(positions[candidates], radii[candidates], corner, size)]
        if level < stop_iter and kept.size > max_count:
            for offset in itertools.product(range(2), repeat=3):
                todo.append(
                    (corner + np.array(offset) * size / 2, size / 2, level + 1, kept)
                )
        else:
            refined.add(tuple(corner) + (size,))
    return refined


def _keep_half_scale(positions, radii, corner, size):
    """
    Decide which particles count for the half-particle-scale criterion.

    Parameters
    ----------
    positions : ~numpy.ndarray
        Particle positions.

    radii : ~numpy.ndarray
        Particle radii.

    corner : ~numpy.ndarray
        Lower corner of the cell.

    size : float
        Side length of the cell.

    Returns
    -------
    ~numpy.ndarray
        Whether each particle counts.
    """
    offsets = positions - corner - size / 2
    return (radii * 0.5 < size) & (
        np.linalg.norm(offsets, axis=1) < radii + size * np.sqrt(3) / 2
    )


def _keep_occupancy(positions, radii, corner, size):
    """
    Decide which particles count for the occupancy criterion.

    Parameters
    ----------
    positions : ~numpy.ndarray
        Particle positions.

    radii : ~numpy.ndarray
        Unused.

    corner : ~numpy.ndarray
        Lower corner of the cell.

    size : float
        Side length of the cell.

    Returns
    -------
    ~numpy.ndarray
        Whether each particle counts.
    """
    return np.sum(np.abs(positions - corner - size / 2), axis=1) < size * 2


class TestAdaptiveCellGridUtils:
    """Test adaptive cell grid utils."""

    def test_keep_particle_occupancy(self):
        """Check that particles near the cell are selected, regardless of radius."""
        cell = (10.0, 100.0, 1000.0, 1.0)
        particles = (
            ((10.5, 100.5, 1000.5), 1, True),
            ((10.5, 100.5, -1000.5), 1e7, False),  # radii should not impact occupancy
        )
        for position, radius, expected in particles:
            assert (
                refinement._keep_particle(
                    refinement._OCCUPANCY, *position, radius, *cell
                )
                == expected
            )

    def test_keep_particle_half_scale(self):
        """Check that small particles intersecting the cell are selected."""
        cell = (10.0, 100.0, 1000.0, 1.0)
        particles = (
            ((10.5, 100.5, 1000.5), 1, True),
            ((10.5, 100.5, -1000.5), 1e7, False),  # too large
            ((10.5, 100.5, -1000.5), 0.1, False),  # too far
            ((10.5, 100.5, 1001.1), 0.25, True),
            ((11, 101, 1001.11), 0.01, False),  # approximated, using the worst case
        )
        for position, radius, expected in particles:
            assert (
                refinement._keep_particle(
                    refinement._HALF_PARTICLE_SCALE, *position, radius, *cell
                )
                == expected
            )

    def test_refine_grid_to_occupancy(self):
        """Check that refinement stops when only one particle occupies cell."""
        cells = np.array([(0, 0, 0, 1)], dtype=CELL_DTYPE)
        positions = np.array([[0.5, 0.5, 0.5]])
        radii = [None]  # Should not be needed for this
        result = refinement.refine_grid_to_single_occupancy(cells, positions, radii)
        assert np.all(cells == refinement._octree_cells(result))

    def test_refine_grid_to_particle_scale(self):
        """Check that the refinement stops appropriately."""
        cells = np.array([(0, 0, 0, 1)], dtype=CELL_DTYPE)
        positions = np.array(
            [
                [0.5, 0.5, 0.5],  # Nearby particle is too big.
                [100, 100, 100],  # Small enough but too far.
            ]
        )
        radii = np.array([10, 0.1])
        result = refinement.refine_grid_to_half_particle_scale(cells, positions, radii)
        assert np.all(cells == refinement._octree_cells(result))

    @pytest.mark.parametrize("stop_iter", (0, 1, 4))
    @pytest.mark.parametrize(
        "strategy, keep, max_count",
        (
            (refinement.refine_grid_to_half_particle_scale, _keep_half_scale, 0),
            (refinement.refine_grid_to_single_occupancy, _keep_occupancy, 2),
        ),
    )
    @pytest.mark.parametrize("clustered", (False, True))
    @pytest.mark.parametrize("ncpu", (1, 2))
    def test_matches_reference(
        self, strategy, keep, max_count, stop_iter, clustered, ncpu
    ):
        """Check the refined cells against a refinement one cell at a time."""
        shape, size = (3, 4, 2), 2.0
        cells = _initial_cells(shape, size)
        extent = np.array(shape) * size
        rng = np.random.default_rng(0)
        positions = rng.uniform(-1.5, extent + 0.5, (200, 3))
        if clustered:
            positions[:100] = rng.normal(extent / 2, extent / 10, (100, 3))
            positions[:5] = np.round(positions[:5]) - 0.5  # on cell corners
        radii = rng.uniform(0.05, 2, 200) ** 2
        tree = strategy(cells, positions, radii, stop_iter, ncpu=ncpu)
        refined = refinement._octree_cells(tree)
        assert set(map(tuple, refined.tolist())) == _reference_refinement(
            cells, positions, radii, keep, max_count, stop_iter
        )
        assert len(set(map(tuple, refined.tolist()))) == len(refined)
        assert tree._octree.depth == np.rint(np.log2(size / refined["size"].min()))
        assert tree._octree.depth <= stop_iter

    @pytest.mark.parametrize("strategy", refinement_strategies)
    def test_tree_matches_rebuilt(self, strategy):
        """
        Check that the tree built during refinement matches one built from its cells.

        If every initial cell is bisected, a tree rebuilt from the cells has smaller root
        nodes, so the searches are compared.
        """
        cells = _initial_cells((3, 4, 2), 2.0)
        rng = np.random.default_rng(1)
        positions = rng.normal((3, 4, 2), 1.0, (100, 3))
        radii = rng.uniform(0.1, 1.5, 100)
        tree = strategy(cells, positions, radii, 4)
        assert tree._octree.depth > 0  # cells were refined
        rebuilt = Octree(tree.cell_centres, tree.cell_sizes)
        coords = rng.uniform(-1.5, 8.5, (300, 3))
        search_radii = rng.uniform(0, 2, 300)
        for emitted, expected in zip(
            tree.find(coords, search_radii), rebuilt.find(coords, search_radii)
        ):
            assert np.array_equal(emitted, expected)
        assert np.array_equal(
            tree.count(coords, search_radii), rebuilt.count(coords, search_radii)
        )
        assert np.array_equal(
            tree.containing_cells(coords), rebuilt.containing_cells(coords)
        )
        if np.any(tree.cell_sizes == 2.0):  # same roots, so the same tree
            for emitted, expected in zip(tree._octree, rebuilt._octree):
                assert np.array_equal(emitted, expected)

    def test_incomplete_initial_grid(self):
        """Check that refinement works when initial cells leave holes in the lattice."""
        cells = _initial_cells((3, 3, 3), 1.0)
        cells = np.delete(cells, [4, 13])  # one with particles nearby
        rng = np.random.default_rng(2)
        positions = rng.uniform(-0.5, 2.5, (50, 3))
        radii = rng.uniform(0.05, 1.0, 50)
        tree = refinement.refine_grid_to_half_particle_scale(cells, positions, radii, 3)
        assert set(map(tuple, refinement._octree_cells(tree).tolist())) == (
            _reference_refinement(cells, positions, radii, _keep_half_scale, 0, 3)
        )

    def test_no_particles(self):
        """Check that cells are not refined without particles."""
        cells = _initial_cells((2, 2, 2), 1.0)
        tree = refinement.refine_grid_to_half_particle_scale(
            cells, np.empty((0, 3)), np.empty(0)
        )
        assert np.array_equal(refinement._octree_cells(tree), cells)

    @pytest.mark.parametrize(
        "change, message",
        (
            ("size", "same size"),
            ("offset", "lattice"),
            ("duplicate", "duplicated"),
            ("empty", "without cells"),
        ),
    )
    def test_bad_initial_cells(self, change, message):
        """Check that initial cells that are not a uniform grid are rejected."""
        cells = _initial_cells((2, 2, 2), 1.0)
        if change == "size":
            cells["size"][0] = 0.5
        elif change == "offset":
            cells["x"][0] += 0.25
        elif change == "duplicate":
            cells[1] = cells[0]
        else:
            cells = cells[:0]
        with pytest.raises(ValueError, match=message):
            refinement.refine_grid_to_half_particle_scale(
                cells, np.zeros((1, 3)), np.ones(1)
            )


class TestAdaptiveCellGrid:
    """Test functionality of adaptive cell grid class."""

    n_px_z = 12
    los_region = (1.0, 9.25)
    grid_shape = (n_px_z, 32, 32)  # 1px cells covering the grid extent and the datacube

    def test_init(self, many_particle_source, dc_zeros):
        """Check class initialization."""
        datacube = dc_zeros
        initial_grid_size = 3
        source = many_particle_source()
        acg = AdaptiveCellGrid(
            source, datacube, self.los_region, initial_grid_size=initial_grid_size
        )
        # insert checks on acg.pix_range
        assert len(acg.initial_cells) == initial_grid_size**3
        # insert checks on cell locations & sizes

    def test_init_particle_locations(self, many_particle_source, dc_zeros):
        """Test actual pixel coordinates in source module tests."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        acg = AdaptiveCellGrid(source, datacube, self.los_region, self.grid_shape)
        acg.init_particle_locations(source, sph_kernel)
        assert acg.positions.shape == (source.hsm_g.size, 3)
        assert acg.radii.shape == (source.hsm_g.size,)
        assert acg.positions.unit == U.pix
        assert acg.radii.unit == U.pix

    def test_eval_grid_refinement(self, many_particle_source, dc_zeros):
        """Check normal completion of eval_grid_refinement."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = AdaptiveCellGrid(source, datacube, self.los_region, self.grid_shape)
        grid.init_particle_locations(source, sph_kernel)
        grid.eval_grid_refinement(
            source, datacube, refinement.refine_grid_to_half_particle_scale
        )
        assert grid.adaptive_cells.dtype == CELL_DTYPE

    @pytest.mark.parametrize("emits_tree", (True, False))
    def test_octree(self, monkeypatch, many_particle_source, dc_zeros, emits_tree):
        """
        Check that the tree of cells is built once, and again after refinement.

        The refinement strategies provide the tree, which is not rebuilt. Strategies
        that return cells instead have their tree built when needed.
        """
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = AdaptiveCellGrid(source, datacube, self.los_region, self.grid_shape)
        grid.init_particle_locations(source, sph_kernel)
        built = []
        init = Octree.__init__

        def count_builds(self, *args):
            """
            Count the trees built.

            Parameters
            ----------
            self : ~martini._grid_search.Octree
                The tree.

            *args : tuple
                Arguments of :class:`~martini._grid_search.Octree`.
            """
            built.append(self)
            init(self, *args)

        def cells_strategy(cells, positions, radii, ncpu=1):
            """
            Refine cells, returning them as an array.

            Parameters
            ----------
            cells : ~numpy.ndarray
                The initial cells.

            positions : ~numpy.ndarray
                Particle positions.

            radii : ~numpy.ndarray
                Particle radii.

            ncpu : int, optional
                Number of threads.

            Returns
            -------
            ~numpy.ndarray
                The refined cells.
            """
            return refinement._octree_cells(
                refinement.refine_grid_to_half_particle_scale(
                    cells, positions, radii, ncpu=ncpu
                )
            )

        monkeypatch.setattr(Octree, "__init__", count_builds)
        trees = []
        for _ in range(2):
            grid.eval_grid_refinement(
                source,
                datacube,
                (
                    refinement.refine_grid_to_half_particle_scale
                    if emits_tree
                    else cells_strategy
                ),
            )
            for interpolant in (module_interpolants.sph, module_interpolants.mfm):
                grid.interpolate_fields(source, sph_kernel, interpolant)
            assert grid.octree is grid.octree
            trees.append(grid.octree)
        # a new tree for each refinement:
        assert trees[0] is not trees[1]
        assert built == ([] if emits_tree else trees)

    def test_init_cell_centres(self, many_particle_source, dc_zeros):
        """Check test_init_cell_centres increment half size to lower corner pos."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = CellGrid(source, datacube, self.los_region, self.grid_shape)
        grid.test_cells = np.array([(0, 0, 0, 1)], dtype=CELL_DTYPE)
        test_centres = np.array([[0.5, 0.5, 0.5]]) * U.pix
        grid.init_cell_centres(prefix="test")
        assert np.all(grid.test_cell_centres == test_centres)

    def test_init_cell_volumes(self, many_particle_source, dc_zeros):
        """Check that rectilinear volume and distortion correction are initialized."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = CellGrid(source, datacube, self.los_region, self.grid_shape)
        size = datacube.datacube_shape[0]
        # LoS centre of the cell at the centre of the region (the source distance):
        grid.test_cells = np.array(
            [(self.n_px_z / 2 - 0.5 - size / 2, -0.5, -0.5, size)], dtype=CELL_DTYPE
        )
        grid.init_cell_centres(prefix="test")
        test_volumes = np.array([datacube.datacube_shape[0] ** 3]) * U.pix**3
        grid.init_cell_volumes(prefix="test")
        assert np.allclose(grid.test_cell_volumes, test_volumes, rtol=1e-6)
        # cell centred on region centre at dec=0 has no distortion:
        grid.init_cell_distances(source, datacube, prefix="test")
        grid.init_cell_distortion_corrections(source, datacube, prefix="test")
        assert np.allclose(grid.test_cell_distortion_corrections, 1, rtol=1e-6)

    def test_init_cell_distances(self, many_particle_source, dc_zeros):
        """Check that cell distances are log-spaced about the source distance."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        grid = CellGrid(source, datacube, self.los_region, self.grid_shape)
        centre = datacube.datacube_shape[0] / 2 - 0.5
        los_centre = self.n_px_z / 2 - 0.5
        offset = 3
        # unit cells centred on the region centre and offset along the LoS:
        grid.test_cells = np.array(
            [
                (los_centre - 0.5, centre - 0.5, centre - 0.5, 1),
                (los_centre + offset - 0.5, centre - 0.5, centre - 0.5, 1),
            ],
            dtype=CELL_DTYPE,
        )
        grid.init_cell_centres(prefix="test")
        grid.init_cell_distances(source, datacube, prefix="test")
        expected_distances = U.Quantity(
            [
                source.distance,
                source.distance * np.exp((offset * datacube.px_size).to_value(U.rad)),
            ]
        )
        assert U.allclose(grid.test_cell_distances, expected_distances, rtol=1e-6)

    def test_cell_distances_match_particle_distances(
        self, many_particle_source, dc_zeros
    ):
        """Check that cells centred on particles are at the particle distances."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = CellGrid(source, datacube, self.los_region, self.grid_shape)
        grid.init_particle_locations(source, sph_kernel)
        grid.test_cells = np.array(
            [(*(position - 0.5), 1) for position in grid.positions.to_value(U.pix)],
            dtype=CELL_DTYPE,
        )
        grid.init_cell_centres(prefix="test")
        grid.init_cell_distances(source, datacube, prefix="test")
        assert U.allclose(
            grid.test_cell_distances, source.skycoords.distance, rtol=1e-6
        )

    def test_init_cell_distortion_corrections_tan(self, many_particle_source):
        """
        Check the solid angle correction for a TAN projection at high declination.

        The solid angle per pixel in a TAN projection is ``sin(theta)**3`` (``theta`` the
        native latitude) times the nominal value, with no dependence on declination. The
        datacube straddles RA=0 to check wrapping.
        """
        datacube = DataCube(
            n_px_x=16,
            n_px_y=16,
            n_channels=4,
            px_size=0.5 * U.deg,
            channel_width=4 * U.km / U.s,
            ra=0 * U.deg,
            dec=60 * U.deg,
        )
        source = many_particle_source()
        grid = CellGrid(source, datacube, self.los_region, (self.n_px_z, 16, 16))
        centre = datacube.datacube_shape[0] / 2 - 0.5
        los_centre = self.n_px_z / 2 - 0.5
        offsets = np.arange(-7, 8)
        sky_pix = np.array([(centre + i, centre + j) for i in offsets for j in offsets])
        # unit cells at the region centre along the LoS:
        grid.test_cells = np.array(
            [(los_centre - 0.5, x - 0.5, y - 0.5, 1) for x, y in sky_pix],
            dtype=CELL_DTYPE,
        )
        grid.init_cell_centres(prefix="test")
        grid.init_cell_distances(source, datacube, prefix="test")
        grid.init_cell_distortion_corrections(source, datacube, prefix="test")
        corrections = grid.test_cell_distortion_corrections.to_value(
            U.dimensionless_unscaled
        )
        theta = datacube.wcs.celestial.wcs.p2s(sky_pix, 0)["theta"]
        assert np.allclose(corrections, np.sin(np.radians(theta)) ** 3, rtol=1e-6)
        # unity at the reference point:
        assert np.isclose(corrections[np.all(sky_pix == centre, axis=1)], 1, rtol=1e-6)
        # no north-south gradient:
        corrections = corrections.reshape((offsets.size, offsets.size))
        assert np.allclose(corrections, corrections[:, ::-1], rtol=1e-6)

    @pytest.mark.parametrize("interpolant", interpolants)
    def test_interpolate_fields(self, many_particle_source, dc_zeros, interpolant):
        """Check that interpolate_fields correctly assigns fields."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = CellGrid(source, datacube, self.los_region, self.grid_shape)
        grid.init_particle_locations(source, sph_kernel)
        grid.interpolate_fields(source, sph_kernel, interpolant)
        fields = grid.interpolated_fields
        for key in fields.keys():
            assert ~np.any(np.isnan(fields[key]))
        assert fields["masses_HI"].unit.is_equivalent(U.Msun / U.pix**3)
        assert fields["velocities"].unit.is_equivalent(U.km / U.s)
        assert fields["temperatures"].unit.is_equivalent(U.km**2 / U.s**2)

    @pytest.mark.parametrize(
        "interpolant", (module_interpolants.sph, module_interpolants.mfm)
    )
    # (3, 8, 8) cells of 4px cover the 12px LoS and 32x32px datacube exactly
    @pytest.mark.parametrize("grid_shape", (grid_shape, (3, 8, 8)))
    def test_uniform_grid_search(
        self, many_particle_source, dc_zeros, interpolant, grid_shape
    ):
        """Check the uniform grid search matches the search for arbitrary cells."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = CellGrid(source, datacube, self.los_region, grid_shape)
        if grid_shape != self.grid_shape:
            assert grid.cell_size != 1  # check the mapping to grid units
        grid.init_particle_locations(source, sph_kernel)
        assert grid.uniform_grid is not None
        grid.interpolate_fields(source, sph_kernel, interpolant)  # uniform search
        # the same cells, without the uniform grid description (arbitrary cells):
        arbitrary = _ArbitraryCellGrid(source, datacube, self.los_region, grid_shape)
        arbitrary.init_particle_locations(source, sph_kernel)
        arbitrary.interpolate_fields(source, sph_kernel, interpolant)
        fields = grid.interpolated_fields
        assert fields.keys() == arbitrary.interpolated_fields.keys()
        for key in fields.keys():
            assert np.any(fields[key] != 0)
            assert U.allclose(fields[key], arbitrary.interpolated_fields[key])

    @pytest.mark.parametrize(
        ("grid_shape", "cell_size", "expected_size"),
        ((grid_shape, None, 1.0), ((6, 16, 16), None, 2.0), ((2, 4, 4), 8.0, 8.0)),
    )
    def test_grid_extent(
        self, many_particle_source, dc_zeros, grid_shape, cell_size, expected_size
    ):
        """
        Check the extent of a grid with a given shape.

        The grid starts at the lower edges of the first pixels along each axis. The cell
        size covers the datacube on the sky if not given. The source distance is at the
        centre of the grid along the LoS.
        """
        source = many_particle_source()
        grid = CellGrid(
            source, dc_zeros, self.los_region, grid_shape, cell_size=cell_size
        )
        assert grid.grid_shape == grid_shape
        assert grid.cell_size == expected_size
        for axis, n_cells in zip("xyz", grid_shape):
            assert grid.cells[axis].min() == -0.5
            high = (grid.cells[axis] + grid.cells["size"]).max()
            assert high == -0.5 + n_cells * expected_size
        assert grid.los_extent == grid_shape[0] * expected_size
        assert grid.uniform_grid == (grid_shape, expected_size)
        # a cell centred on the grid along the LoS is at the source distance:
        los_centre = grid.los_extent / 2 - 0.5
        grid.test_cells = np.array([(los_centre - 0.5, 0, 0, 1)], dtype=CELL_DTYPE)
        grid.init_cell_centres(prefix="test")
        grid.init_cell_distances(source, dc_zeros, prefix="test")
        assert U.isclose(grid.test_cell_distances[0], source.distance)

    @pytest.mark.parametrize(
        ("lower", "size", "expected"),
        (
            (2.0, 1.0, 1.0),  # inside
            (-3.0, 1.0, 0.0),  # outside, below
            (12.0, 1.0, 0.0),  # outside, above
            (0.5, 1.0, 0.5),  # straddling the lower edge
            (8.75, 1.0, 0.5),  # straddling the upper edge
            (-1.0, 12.0, 8.25 / 12),  # covering the region and beyond
        ),
    )
    def test_los_overlap(self, lower, size, expected):
        """Check the fraction of a cell's extent along the LoS within the region."""
        assert np.isclose(_los_overlap(lower, size, *self.los_region), expected)

    @pytest.mark.parametrize(
        ("grid_kind", "layer_weights"),
        (
            # self.los_region is (1.0, 9.25); layers of 1px start at -0.5, layers of 2px
            # at -0.5, 1.5, 3.5, ..., layers of 0.5px at -0.5, 0.0, 0.5, ... (so the
            # lower edge of the region coincides with a layer edge)
            ("uniform 1px", {0: 0, 1: 0.5, 4: 1, 9: 0.75, 10: 0, 11: 0}),
            ("adaptive 1px", {0: 0, 1: 0.5, 4: 1, 9: 0.75, 10: 0, 11: 0}),
            ("uniform 2px", {0: 0.25, 2: 1, 4: 0.875, 5: 0}),
            ("uniform 0.5px", {2: 0, 3: 1, 19: 0.5, 20: 0}),
        ),
    )
    @pytest.mark.parametrize("ncpu", (1, 2))
    def test_radiative_transfer_excludes_beyond_region(
        self, many_particle_source, dc_zeros, grid_kind, layer_weights, ncpu
    ):
        """
        Check that only the mass within the region along the LoS reaches the cube.

        Fill one layer of cells along the LoS at a time and compare the cube with the
        region set to the full grid and to ``self.los_region``: layers beyond the region
        do not contribute, layers straddling an edge contribute the fraction of their mass
        inside the region. Covers both the case where cell columns map one-to-one onto
        data cube pixels and the case where cells cover multiple pixels and/or are covered
        by multiple pixels.
        """
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        if grid_kind == "adaptive 1px":
            sph_kernel = CubicSplineKernel()
            sph_kernel._init_sm_lengths(source, datacube)
            sph_kernel._init_sm_ranges(ndim=3)
            grid = AdaptiveCellGrid(
                source,
                datacube,
                self.los_region,
                (self.n_px_z,) + tuple(datacube.datacube_shape[:2]),
                initial_cell_size=1,
            )
            grid.init_particle_locations(source, sph_kernel)
            # leave the cells unrefined:
            grid.eval_grid_refinement(
                source, datacube, lambda cells, positions, radii, ncpu: cells
            )
            cells = grid.adaptive_cells
        else:
            grid_shape = {
                "uniform 1px": self.grid_shape,
                "uniform 2px": (self.n_px_z // 2, 16, 16),
                "uniform 0.5px": (2 * self.n_px_z, 64, 64),
            }[grid_kind]
            grid = CellGrid(source, datacube, self.los_region, grid_shape)
            cells = grid.cells
        layers = np.rint((cells["x"] + 0.5) / cells["size"]).astype(int)
        mid_channel = datacube.velocity_channel_edges[7:9].mean()
        full_region = (-0.5, -0.5 + layers.max() * cells["size"].max() + 1e3)
        for layer, weight in layer_weights.items():
            n_cells = len(cells)
            grid.interpolated_fields = {
                "masses_HI": np.where(layers == layer, 1.0, 0.0) * U.Msun / U.pix**3,
                "velocities": np.ones(n_cells) * mid_channel.to(U.km / U.s),
                "temperatures": np.ones(n_cells) * U.km**2 / U.s**2,
            }
            grid.los_region = full_region
            cube_full = optically_thin(grid, datacube, DiracDeltaSpectrum(), ncpu=ncpu)
            grid.los_region = self.los_region
            cube = optically_thin(grid, datacube, DiracDeltaSpectrum(), ncpu=ncpu)
            assert cube_full.sum() > 0
            assert U.allclose(cube, weight * cube_full, atol=0 * cube_full.unit)

    def test_radiative_transfer_cell_size(self, many_particle_source, dc_zeros):
        """
        Check that a uniform grid of 2px cells gives the same mass as one of 1px cells.

        Cells larger than a pixel share their mass equally between the pixels that they
        cover. The cell distances and distortion corrections differ slightly between the
        grids.
        """
        datacube = dc_zeros
        source = many_particle_source()
        mid_channel = datacube.velocity_channel_edges[7:9].mean()
        cubes = []
        for grid_shape in (self.grid_shape, (self.n_px_z // 2, 16, 16)):
            grid = CellGrid(source, datacube, self.los_region, grid_shape)
            n_cells = len(grid.cells)
            grid.interpolated_fields = {
                "masses_HI": np.ones(n_cells) * U.Msun / U.pix**3,
                "velocities": np.ones(n_cells) * mid_channel.to(U.km / U.s),
                "temperatures": np.ones(n_cells) * U.km**2 / U.s**2,
            }
            cubes.append(optically_thin(grid, datacube, DiracDeltaSpectrum()))
        assert cubes[0].shape == cubes[1].shape
        assert U.isclose(cubes[1].sum(), cubes[0].sum(), rtol=1e-3)

    @pytest.mark.parametrize("ncpu", (1, 2))
    def test_radiative_transfer_matches_loop(
        self, many_particle_source, dc_zeros, ncpu
    ):
        """
        Check the cube of a refined grid against adding one cell at a time.

        The LoS region cuts through cells, so some contribute part of their mass. The
        spectra of the cells are added to each pixel in the same order, so the cubes are
        identical.
        """
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube, n_px_z=self.n_px_z)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges(ndim=3)
        grid = AdaptiveCellGrid(
            source,
            datacube,
            self.los_region,
            (self.n_px_z // 2, 16, 16),
            initial_cell_size=2,
        )
        grid.init_particle_locations(source, sph_kernel)
        # refine around clustered points (the particles of the source are too smooth):
        initial = grid.initial_cells
        corners = np.column_stack([initial[axis] for axis in "xyz"])
        extent = corners.max(axis=0) + 2 - corners.min(axis=0)
        rng = np.random.default_rng(0)
        points = rng.normal(corners.min(axis=0) + extent / 2, extent / 8, (300, 3))
        grid.eval_grid_refinement(
            source,
            datacube,
            lambda cells, positions, radii, ncpu: (
                refinement.refine_grid_to_single_occupancy(
                    cells, points, stop_iter=3, ncpu=ncpu
                )
            ),
            ncpu=ncpu,
        )
        cells = grid.adaptive_cells
        assert np.unique(cells["size"]).size > 2  # cells larger and smaller than pixels
        channels = datacube.velocity_channel_edges.to_value(U.km / U.s)
        grid.interpolated_fields = {
            "masses_HI": rng.uniform(0, 1, len(cells)) * U.Msun / U.pix**3,
            "velocities": rng.uniform(channels.min(), channels.max(), len(cells))
            * U.km
            / U.s,
            "temperatures": rng.uniform(1, 100, len(cells)) * U.km**2 / U.s**2,
        }
        spectral_model = GaussianSpectrum()
        spectral_model.spec_dtype = np.float64
        cube = optically_thin(grid, datacube, spectral_model, ncpu=ncpu)
        # reference: add the spectrum of one cell at a time to the pixels it covers
        field_spectra = calculate_field_spectrum(
            grid,
            datacube,
            spectral_model,
            grid.adaptive_cell_volumes,
            grid.adaptive_cell_distortion_corrections,
            grid.adaptive_cell_distances,
        )
        begin = np.floor(np.column_stack([cells["y"], cells["z"]]) + 0.5).astype(int)
        end = np.ceil(
            np.column_stack([cells["y"], cells["z"]]) + 0.5 + cells["size"][:, None]
        ).astype(int)
        field_spectra = (field_spectra / np.prod(end - begin, axis=1)).value
        expected = np.zeros((field_spectra.shape[0], *datacube.datacube_shape[:2]))
        for i in range(len(cells)):
            weight = 1.0
            if (
                cells["x"][i] < self.los_region[0]
                or cells["x"][i] + cells["size"][i] > self.los_region[1]
            ):
                weight = _los_overlap(cells["x"][i], cells["size"][i], *self.los_region)
                if weight == 0:
                    continue
            expected[:, begin[i, 0] : end[i, 0], begin[i, 1] : end[i, 1]] += (
                weight * field_spectra[:, i, None, None]
            )
        assert expected.sum() > 0
        assert np.array_equal(cube.value, np.flip(np.moveaxis(expected, 1, 2), axis=2))

    def test_radiative_transfer_misaligned(self, many_particle_source, dc_zeros):
        """Check that cells not aligned with the pixel grid are rejected."""
        datacube = dc_zeros
        source = many_particle_source()
        grid = CellGrid(source, datacube, self.los_region, (self.n_px_z // 2, 16, 16))
        grid.cells["y"] += 0.5
        n_cells = len(grid.cells)
        grid.interpolated_fields = {
            "masses_HI": np.ones(n_cells) * U.Msun / U.pix**3,
            "velocities": np.zeros(n_cells) * U.km / U.s,
            "temperatures": np.ones(n_cells) * U.km**2 / U.s**2,
        }
        with pytest.raises(RuntimeError, match="not aligned with pixel grid"):
            optically_thin(grid, datacube, DiracDeltaSpectrum())


def _uniform_test_grid(shape=(6, 5, 4), size=1.0):
    """
    Make a uniform grid of cells with the attributes used by the interpolants.

    Cell centres are ordered as in :func:`~martini._grid_search.find_grid_intersections`
    and the grid starts at ``-0.5`` along each axis, as for
    :class:`~martini.mochi.cell_grid.CellGrid`.

    Parameters
    ----------
    shape : tuple, optional
        Number of cells along each axis.

    size : float, optional
        Side length of the cells, in pixels.

    Returns
    -------
    ~types.SimpleNamespace
        The cell grid.
    """
    centres = (np.array(list(itertools.product(*map(range, shape)))) + 0.5) * size - 0.5
    n_pos = len(centres)
    return SimpleNamespace(
        field_positions=centres * U.pix,
        field_volumes=np.full(n_pos, size**3) * U.pix**3,
        field_sizes=np.full(n_pos, size) * U.pix,
        field_distortion_corrections=np.linspace(0.9, 1.1, n_pos)
        * U.dimensionless_unscaled,
        uniform_grid=(shape, size),
    )


def _adaptive_test_grid():
    """
    Make an adaptive grid of cells with the attributes used by the interpolants.

    2x2x2 coarse cells of size 2, one bisected, one of whose children is bisected (as in
    ``_small_tree_grid`` in ``test_grid_search.py``), starting at ``-0.5``.

    Returns
    -------
    ~types.SimpleNamespace
        The cell grid.
    """
    corners, sizes = [], []
    for corner in itertools.product((0, 2), repeat=3):
        if corner != (0, 0, 0):
            corners.append(corner)
            sizes.append(2.0)
    for corner in itertools.product((0, 1), repeat=3):
        if corner != (1, 1, 1):
            corners.append(corner)
            sizes.append(1.0)
    for corner in itertools.product((1, 1.5), repeat=3):
        corners.append(corner)
        sizes.append(0.5)
    sizes = np.array(sizes)
    centres = np.array(corners, dtype=float) + sizes[:, np.newaxis] / 2 - 0.5
    n_pos = len(centres)
    return SimpleNamespace(
        field_positions=centres * U.pix,
        field_volumes=sizes**3 * U.pix**3,
        field_sizes=sizes * U.pix,
        field_distortion_corrections=np.linspace(0.9, 1.1, n_pos)
        * U.dimensionless_unscaled,
        uniform_grid=None,
        octree=Octree(centres, sizes),
    )


def _scatter_test_particles(n=40, extent=3.5, h=(0.3, 2.0), seed=0):
    """
    Make random particle properties, including particles on cell boundaries.

    Parameters
    ----------
    n : int, optional
        Number of random particles (four more are placed on cell boundaries).

    extent : float, optional
        Particles are placed between ``-0.5`` and ``extent - 0.5`` pixels along each axis.

    h : tuple, optional
        Range of smoothing lengths, in pixels.

    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Particle properties, as keyword arguments for the interpolants.
    """
    rng = np.random.default_rng(seed)
    positions = np.vstack(
        (
            rng.uniform(-0.5, extent - 0.5, (n, 3)),
            # on cell faces, edges and corners (cell edges are at half-integers):
            [[0.5, 1.2, 1.3], [0.5, 1.5, 0.7], [1.5, 0.5, 2.5], [-0.5, -0.5, -0.5]],
        )
    )
    n = len(positions)
    return dict(
        positions=positions * U.pix,
        velocities=rng.uniform(-50, 50, n) * U.km / U.s,
        smoothing_lengths=rng.uniform(*h, n) * U.pix,
        masses_HI=rng.uniform(0.5, 2, n) * U.Msun,
        temperatures=rng.uniform(100, 200, n) * U.km**2 / U.s**2,
        masses=rng.uniform(1, 3, n) * U.Msun,
    )


class TestKernelInterpolation:
    """Test interpolation onto cells with SPH kernels (compiled and numpy)."""

    @pytest.mark.parametrize("interpolant", ("sph", "mfm"))
    @pytest.mark.parametrize("grid", ("uniform", "adaptive"))
    @pytest.mark.parametrize(
        "kernel", (_CubicSplineKernel, _GaussianKernel, DiracDeltaKernel)
    )
    def test_compiled_matches_numpy(self, interpolant, grid, kernel):
        """Check that compiled and numpy evaluation of kernel weights agree."""
        cell_grid = _uniform_test_grid() if grid == "uniform" else _adaptive_test_grid()
        particles = _scatter_test_particles()
        k = kernel()
        k.sm_lengths = particles["smoothing_lengths"]
        results = [
            getattr(module_interpolants, interpolant)(
                **particles, kernel=k, cell_grid=cell_grid, compiled=compiled
            )
            for compiled in (True, False)
        ]
        assert np.count_nonzero(results[0]["masses_HI"]) > 0
        for field in ("masses_HI", "velocities", "temperatures"):
            assert U.allclose(results[0][field], results[1][field], rtol=1e-10)

    @pytest.mark.parametrize("compiled", (True, False))
    @pytest.mark.parametrize("interpolant", ("sph", "mfm"))
    @pytest.mark.parametrize("grid", ("uniform", "adaptive"))
    def test_dirac_delta_matches_particle_in_cell(self, compiled, interpolant, grid):
        """
        Check that a Dirac-delta kernel deposits each particle in the cell containing it.

        Includes particles on faces, edges and corners shared by several cells, and
        particles much smaller than the cells (the search must still find their cells).
        """
        cell_grid = _uniform_test_grid() if grid == "uniform" else _adaptive_test_grid()
        particles = _scatter_test_particles(h=(0.001, 0.01))
        k = DiracDeltaKernel()
        k.sm_lengths = particles["smoothing_lengths"]
        result = getattr(module_interpolants, interpolant)(
            **particles, kernel=k, cell_grid=cell_grid, compiled=compiled
        )
        expected = module_interpolants.particle_in_cell(
            **particles, kernel=k, cell_grid=cell_grid
        )
        for field in ("masses_HI", "velocities", "temperatures"):
            assert U.allclose(result[field], expected[field], rtol=1e-10)
        # all particles are inside the grid, and all of their mass is deposited:
        mass = (
            result["masses_HI"]
            * cell_grid.field_volumes
            * cell_grid.field_distortion_corrections
        ).sum()
        assert U.isclose(mass, particles["masses_HI"].sum(), rtol=1e-12)

    @pytest.mark.parametrize("compiled", (True, False))
    @pytest.mark.parametrize("interpolant", ("sph", "mfm"))
    def test_kernel_smaller_than_cells(self, compiled, interpolant):
        """Check that a kernel reaching no cell centre deposits nothing (and no NaNs)."""
        cell_grid = _uniform_test_grid()
        particles = _scatter_test_particles(h=(0.001, 0.01))
        # away from cell centres (at integer coordinates):
        particles["positions"] = np.full((len(particles["masses"]), 3), 1.3) * U.pix
        k = _CubicSplineKernel()
        k.sm_lengths = particles["smoothing_lengths"]
        result = getattr(module_interpolants, interpolant)(
            **particles, kernel=k, cell_grid=cell_grid, compiled=compiled
        )
        for field in ("masses_HI", "velocities", "temperatures"):
            assert np.all(result[field] == 0)

    @pytest.mark.parametrize("interpolant", ("sph", "mfm"))
    def test_threads(self, interpolant):
        """Check that using several threads gives the same result as one."""
        cell_grid = _uniform_test_grid(shape=(12, 12, 12), size=0.5)
        particles = _scatter_test_particles(n=400, extent=5.5)
        k = _CubicSplineKernel()
        k.sm_lengths = particles["smoothing_lengths"]
        results = [
            getattr(module_interpolants, interpolant)(
                **particles, kernel=k, cell_grid=cell_grid, ncpu=ncpu
            )
            for ncpu in (1, 2)
        ]
        for field in ("masses_HI", "velocities", "temperatures"):
            assert U.allclose(results[0][field], results[1][field], rtol=1e-12)

    @pytest.mark.parametrize("compiled", (True, False))
    def test_adaptive_kernel_support(self, compiled):
        """
        Check that adaptive kernels evaluate their preferred kernel over its support.

        Mochi ignores the martini kernel validation. A decision on what to do when for
        example a kernel is smaller than its containing cell and misses the centre is
        needed later (maybe cell refinement prevents this? maybe fall back to
        DiracDeltaKernel through the AdaptiveKernel mechanism? other?). Here we check
        the current behaviour of always taking the preferred kernel of an adaptive
        kernel for kernel evaluation in Mochi.
        """
        cell_grid = _uniform_test_grid(shape=(16, 16, 16), size=1.0)
        particles = _scatter_test_particles(n=1, extent=1, h=(3, 3))
        particles = {k: v[:1] for k, v in particles.items()}
        particles["positions"] = np.full((1, 3), 7.3) * U.pix
        results = []
        for kernel_index in (0, 1):  # preferred, DiracDeltaKernel
            k = CubicSplineKernel()
            k.sm_lengths = particles["smoothing_lengths"]
            k.kernel_indices = np.array([kernel_index])
            k._rescale = np.array([k.kernels[kernel_index]._rescale])
            k.size_in_fwhm = np.array([k.kernels[kernel_index].size_in_fwhm])
            results.append(
                module_interpolants.sph(
                    **particles, kernel=k, cell_grid=cell_grid, compiled=compiled
                )
            )
        assert U.allclose(results[0]["masses_HI"], results[1]["masses_HI"], rtol=1e-12)

    def test_missing_compiled_kernel(self):
        """Check that a kernel without a compiled implementation is rejected."""

        class CustomKernel(_CubicSplineKernel):
            _scalar_kernel = None

        particles = _scatter_test_particles()
        k = CustomKernel()
        k.sm_lengths = particles["smoothing_lengths"]
        with pytest.raises(ValueError, match="has no compiled kernel"):
            module_interpolants.sph(
                **particles, kernel=k, cell_grid=_uniform_test_grid(), compiled=True
            )
        # the numpy implementation needs no compiled kernel:
        module_interpolants.sph(
            **particles, kernel=k, cell_grid=_uniform_test_grid(), compiled=False
        )

    @pytest.mark.parametrize("interpolant", ("sph", "mfm"))
    @pytest.mark.parametrize("grid", ("uniform", "adaptive"))
    @pytest.mark.parametrize("compiled", (True, False))
    def test_batches_match_single_batch(self, monkeypatch, interpolant, grid, compiled):
        """Check that processing particles in batches gives the same fields."""
        cell_grid = _uniform_test_grid() if grid == "uniform" else _adaptive_test_grid()
        particles = _scatter_test_particles()
        k = _CubicSplineKernel()
        k.sm_lengths = particles["smoothing_lengths"]
        kwargs = dict(**particles, kernel=k, cell_grid=cell_grid, compiled=compiled)
        interp = getattr(module_interpolants, interpolant)
        single = interp(**kwargs)
        per_particle, per_batch, fixed, planned = _batching_estimate(
            monkeypatch, interp, **kwargs
        )
        # room for about a quarter of the particles in each batch:
        allowance = fixed + per_batch + max(per_particle.max(), per_particle.sum() / 4)
        with pytest.warns(RuntimeWarning, match="Processing in batches"):
            batched = interp(**kwargs, mem_allowance_GB=allowance)
        segments = planned[0]
        assert len(segments) > 2
        assert segments[0].start == 0 and segments[-1].stop == len(per_particle)
        assert all(a.stop == b.start for a, b in zip(segments[:-1], segments[1:]))
        assert single.keys() == batched.keys()
        for key in single:
            expected = single[key].value
            assert U.allclose(
                batched[key].value, expected, rtol=1e-12, atol=1e-12 * expected.max()
            )
        # particles are cut off by the edge of the grid (MFM corrects their volumes):
        assert np.any(
            module_interpolants._get_out_of_bound_particles(
                particles["positions"],
                particles["smoothing_lengths"] * k.size_in_fwhm,
                cell_grid.field_positions,
            )
        )

    @pytest.mark.parametrize("interpolant", ("sph", "mfm"))
    @pytest.mark.parametrize("grid", ("uniform", "adaptive"))
    @pytest.mark.parametrize("compiled", (True, False))
    def test_no_pairs(self, interpolant, grid, compiled):
        """Check that particles that reach no cells give empty fields."""
        cell_grid = _uniform_test_grid() if grid == "uniform" else _adaptive_test_grid()
        particles = _scatter_test_particles(n=3)
        particles["positions"] = particles["positions"] + 100 * U.pix
        k = _CubicSplineKernel()
        k.sm_lengths = particles["smoothing_lengths"]
        fields = getattr(module_interpolants, interpolant)(
            **particles, kernel=k, cell_grid=cell_grid, compiled=compiled
        )
        for field in fields.values():
            assert np.all(field.value == 0)

    @pytest.mark.parametrize("grid", ("uniform", "adaptive"))
    def test_insufficient_allowance(self, monkeypatch, grid):
        """Check that an allowance too small even for batches is rejected."""
        cell_grid = _uniform_test_grid() if grid == "uniform" else _adaptive_test_grid()
        particles = _scatter_test_particles()
        k = _CubicSplineKernel()
        k.sm_lengths = particles["smoothing_lengths"]
        kwargs = dict(**particles, kernel=k, cell_grid=cell_grid)
        per_particle, per_batch, fixed, _ = _batching_estimate(
            monkeypatch, module_interpolants.mfm, **kwargs
        )
        # too small for the cells, then for the particle with the most pairs:
        for allowance in (fixed, fixed + per_batch + per_particle.max() / 2):
            with (
                pytest.warns(RuntimeWarning, match="Processing in batches"),
                pytest.raises(RuntimeError, match="insufficient, even when"),
            ):
                module_interpolants.mfm(**kwargs, mem_allowance_GB=allowance)


def _batching_estimate(monkeypatch, interpolant, **kwargs):
    """
    Get the memory estimate of an interpolation, and record how particles are batched.

    Parameters
    ----------
    monkeypatch : ~pytest.MonkeyPatch
        Used to record the batches planned by later calls.

    interpolant : Callable
        The interpolant (:func:`~martini.mochi.interpolants.sph` or
        :func:`~martini.mochi.interpolants.mfm`).

    **kwargs : dict
        Keyword arguments for the interpolant.

    Returns
    -------
    tuple
        The memory estimate for each particle, the fixed memory per batch and the
        memory needed regardless of batching (see
        :func:`~martini.mochi.interpolants._scatter_mem_estimate`), and the list that
        records the batches planned by later calls.
    """
    estimates, planned = [], []
    estimate = module_interpolants._scatter_mem_estimate
    plan = module_interpolants._plan_segments

    def record_estimate(*args):
        """
        Record the memory estimate.

        Parameters
        ----------
        *args : tuple
            Arguments of :func:`~martini.mochi.interpolants._scatter_mem_estimate`.

        Returns
        -------
        tuple
            The memory estimate.
        """
        estimates.append(estimate(*args))
        return estimates[-1]

    def record_plan(*args):
        """
        Record the planned batches.

        Parameters
        ----------
        *args : tuple
            Arguments of :func:`~martini._util._plan_segments`.

        Returns
        -------
        list
            The batches.
        """
        planned.append(plan(*args))
        return planned[-1]

    monkeypatch.setattr(module_interpolants, "_scatter_mem_estimate", record_estimate)
    monkeypatch.setattr(module_interpolants, "_plan_segments", record_plan)
    interpolant(**kwargs, mem_allowance_GB=np.inf)
    assert len(planned) == 0  # all at once
    return estimates[0] + (planned,)


class TestParticleInCell:
    """Test deposition of particles into the cells containing them."""

    def test_uniform_containing_cells(self):
        """Check containing cells on a uniform grid against a brute force search."""
        shape, size = (3, 2, 4), 2.0
        corners = np.array(list(itertools.product(*map(range, shape)))) * size - 0.5
        rng = np.random.default_rng(0)
        coords = np.vstack(
            (
                rng.uniform(-1.5, 8.5, (1000, 3)),
                # on cell boundaries, at the grid edges and outside the grid:
                [[1.5, 1.5, 1.5], [-0.5, -0.5, -0.5], [5.5, 1, 1], [-0.6, 1, 1]],
                [[np.nan, 0, 0], [1e300, 0, 0]],
            )
        )
        inside = np.all(
            (coords[:, np.newaxis] >= corners)
            & (coords[:, np.newaxis] < corners + size),
            axis=2,
        )
        expected = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
        assert np.array_equal(
            module_interpolants._uniform_containing_cells(coords, shape, size),
            expected,
        )

    def test_particle_in_cell(self):
        """Check deposited density, and mass-weighted velocity and temperature."""
        field_positions = (
            np.array(list(itertools.product((0.5, 1.5), repeat=3))) * U.pix
        )
        n_pos = len(field_positions)
        distortion_corrections = np.ones(n_pos) * U.dimensionless_unscaled
        distortion_corrections[0] = 2  # cell with corner at (0, 0, 0)
        # two particles in the first cell, one outside the grid:
        fields = module_interpolants.particle_in_cell(
            positions=np.array([[0.2, 0.3, 0.4], [0.7, 0.1, 0.9], [5, 5, 5]]) * U.pix,
            velocities=np.array([10, 30, 50]) * U.km / U.s,
            smoothing_lengths=np.ones(3) * U.pix,
            masses_HI=np.array([0.5, 1.5, 2.5]) * U.Msun,
            temperatures=np.array([100, 300, 500]) * U.km**2 / U.s**2,
            masses=np.array([1, 3, 5]) * U.Msun,
            kernel=DiracDeltaKernel(),
            cell_grid=SimpleNamespace(  # any object providing the cells will do
                field_positions=field_positions,
                field_volumes=np.ones(n_pos) * U.pix**3,
                field_distortion_corrections=distortion_corrections,
                uniform_grid=None,
                octree=Octree(field_positions.value, np.ones(n_pos)),
            ),
        )
        expected_densities = np.zeros(n_pos) * U.Msun / U.pix**3
        expected_densities[0] = 2.0 * U.Msun / (1 * U.pix**3 * 2)
        expected_velocities = np.zeros(n_pos) * U.km / U.s
        expected_velocities[0] = (1 * 10 + 3 * 30) / 4 * U.km / U.s
        expected_temperatures = np.zeros(n_pos) * U.km**2 / U.s**2
        expected_temperatures[0] = (1 * 100 + 3 * 300) / 4 * U.km**2 / U.s**2
        assert U.allclose(fields["masses_HI"], expected_densities)
        assert U.allclose(fields["velocities"], expected_velocities)
        assert U.allclose(fields["temperatures"], expected_temperatures)


class TestVoronoiMesh:
    """Test interpolation with a Voronoi mesh."""

    def test_grid_edges(self):
        """
        Check that particles anywhere inside the grid contribute.

        The grid extends half a cell beyond the outermost cell centres, and includes its
        lower edges but not its upper edges. Particles nearest to no cell centre must
        still be assigned to the cell containing them.
        """
        field_positions = (
            np.array(list(itertools.product((0.5, 1.5), repeat=3))) * U.pix
        )
        n_pos = len(field_positions)
        positions = np.vstack(
            (
                field_positions.to_value(U.pix),  # one particle at each cell centre
                [
                    [0.1, 0.5, 0.5],  # beyond the outermost cell centres, kept
                    [0.0, 1.5, 1.5],  # on a lower edge of the grid, kept
                    [1.95, 1.5, 0.5],  # beyond the outermost cell centres, kept
                    [2.0, 0.5, 1.5],  # on an upper edge of the grid, excluded
                ],
            )
        )
        masses_HI = np.r_[np.ones(n_pos), 2, 4, 16, 32] * U.Msun
        n = len(positions)
        fields = module_interpolants.voronoi_mesh(
            positions=positions * U.pix,
            velocities=np.zeros(n) * U.km / U.s,
            smoothing_lengths=np.ones(n) * U.pix,
            masses_HI=masses_HI,
            temperatures=np.ones(n) * U.km**2 / U.s**2,
            masses=masses_HI,
            kernel=DiracDeltaKernel(),
            cell_grid=SimpleNamespace(  # any object providing the cells will do
                field_positions=field_positions,
                field_volumes=np.ones(n_pos) * U.pix**3,
                field_distortion_corrections=np.ones(n_pos) * U.dimensionless_unscaled,
            ),
        )
        assert U.isclose(np.sum(fields["masses_HI"] * U.pix**3), masses_HI[:-1].sum())

    def test_shared_cell(self):
        """
        Check a cell shared by several particles nearest to no cell centre.

        Each particle sharing the cell is assigned an equal part of its volume, and the
        fields in the cell are the means over the particles.
        """
        field_positions = (
            np.array(list(itertools.product((0.5, 1.5), repeat=3))) * U.pix
        )
        n_pos = len(field_positions)
        positions = np.vstack(
            (
                field_positions.to_value(U.pix),  # one particle at each cell centre
                [[0.1, 0.5, 0.5], [0.2, 0.4, 0.5]],  # both in the first cell
            )
        )
        masses_HI = np.r_[np.ones(n_pos), 2, 4] * U.Msun
        velocities = np.r_[np.ones(n_pos), 4, 7] * U.km / U.s
        n = len(positions)
        fields = module_interpolants.voronoi_mesh(
            positions=positions * U.pix,
            velocities=velocities,
            smoothing_lengths=np.ones(n) * U.pix,
            masses_HI=masses_HI,
            temperatures=np.ones(n) * U.km**2 / U.s**2,
            masses=masses_HI,
            kernel=DiracDeltaKernel(),
            cell_grid=SimpleNamespace(  # any object providing the cells will do
                field_positions=field_positions,
                field_volumes=np.ones(n_pos) * U.pix**3,
                field_distortion_corrections=np.ones(n_pos) * U.dimensionless_unscaled,
            ),
        )
        assert U.isclose(np.sum(fields["masses_HI"] * U.pix**3), masses_HI.sum())
        assert U.isclose(fields["masses_HI"][0] * U.pix**3, (1 + 2 + 4) * U.Msun)
        assert U.isclose(fields["velocities"][0], (1 + 4 + 7) / 3 * U.km / U.s)


class TestMochi:
    """Main API test for MOCHI backend."""

    los_depth = 2.5 * U.kpc  # about 11.5px at 3Mpc for 15arcsec pixels

    @pytest.mark.parametrize(
        "shape", [(16, 16), (100, 100), (128, 96), (38, 250), (2, 50), (514, 514)]
    )
    def test_coarse_shape(self, shape):
        """
        Check coarse grid has power-of-two cells exactly covering the padded shape.

        Also checks that no more than 10% additional cells are added.
        """
        coarse_shape, cell_size, pad = _coarse_shape(shape)
        assert cell_size & (cell_size - 1) == 0  # power of two
        for n, c, p in zip(shape, coarse_shape, pad):
            assert n + 2 * p == c * cell_size
        assert np.prod(np.array(coarse_shape) * cell_size) <= 1.1 * np.prod(shape)

    def test_los_depth_required(self, many_particle_source, dc_zeros):
        """Check that the depth along the LoS must be given."""
        with pytest.raises(TypeError, match="los_depth"):
            Mochi(
                source=many_particle_source(),
                datacube=dc_zeros,
                sph_kernel=CubicSplineKernel(),
                spectral_model=DiracDeltaSpectrum(),
                interpolant=module_interpolants.sph,
            )

    @pytest.mark.parametrize(
        ("los_depth", "error"),
        (
            (0 * U.kpc, ValueError),
            (-1 * U.kpc, ValueError),
            (6 * U.Mpc, ValueError),  # twice the source distance
        ),
    )
    def test_invalid_los_depth_raises(
        self, many_particle_source, dc_zeros, los_depth, error
    ):
        """Check that the depth along the LoS must be a positive length within range."""
        with pytest.raises(error, match="los_depth|line-of-sight depth"):
            Mochi(
                source=many_particle_source(distance=3 * U.Mpc),
                datacube=dc_zeros,
                sph_kernel=CubicSplineKernel(),
                spectral_model=DiracDeltaSpectrum(),
                los_depth=los_depth,
                interpolant=module_interpolants.sph,
            )

    def test_los_region(self):
        """
        Check the edges of the region along the LoS in pixel coordinates.

        The edges map back to the source distance minus and plus half of the depth. The
        region is wider on the near side in pixel coordinates (logarithmic in distance).
        """
        distance, depth, px_size = 3 * U.Mpc, 500 * U.kpc, 10 * U.arcsec
        los_lower_bound, los_upper_bound = Mochi._los_region(depth, distance, px_size)
        px = px_size.to_value(U.rad)
        assert U.isclose(distance * np.exp(los_lower_bound * px), distance - depth / 2)
        assert U.isclose(distance * np.exp(los_upper_bound * px), distance + depth / 2)
        assert -los_lower_bound > los_upper_bound > 0
        for bad_depth in (0 * U.kpc, -depth, 2 * distance):
            with pytest.raises(ValueError, match="line-of-sight depth"):
                Mochi._los_region(bad_depth, distance, px_size)

    def test_prune_particles(self, dc_zeros):
        """
        Check that particles are pruned in RA, Dec and LoS distance only.

        Particles that cannot contribute to the region along the LoS or on the sky are
        pruned. Particles with zero HI mass or outside the spectral bandwidth are not.
        """
        # 1px is about 0.22kpc at 3Mpc, so the region is about 2.6kpc deep and (with
        # padding for the beam) about 9kpc wide on the sky
        xyz_g = U.Quantity(
            [
                [0, 0, 0],  # kept
                [5, 0, 0],  # beyond the region along the LoS, pruned
                [-5, 0, 0],  # in front of the region along the LoS, pruned
                [0, 0, 0],  # zero HI mass, kept
                [0, 0, 0],  # outside the spectral bandwidth, kept
                [0, 10, 0],  # outside the region on the sky, pruned
            ],
            U.kpc,
        )
        expected_kept = {0, 3, 4}
        n = xyz_g.shape[0]
        vxyz_g = np.zeros((n, 3)) * U.km / U.s
        vxyz_g[4, 0] = 1000 * U.km / U.s
        mHI_g = np.ones(n) * 1.0e4 * U.Msun
        mHI_g[3] = 0
        source = SPHSource(
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            T_g=(1.0e4 + np.arange(n)) * U.K,  # label particles to identify survivors
            hsm_g=np.ones(n) * 0.1 * U.kpc,
            mHI_g=mHI_g,
            distance=3 * U.Mpc,
        )
        m = Mochi(
            source=source,
            datacube=dc_zeros,
            beam=GaussianBeam(),
            sph_kernel=CubicSplineKernel(),
            spectral_model=GaussianSpectrum(),
            los_depth=self.los_depth,
            interpolant=module_interpolants.sph,
            quiet=True,
        )
        kept = np.rint(m.source.T_g.to_value(U.K) - 1.0e4).astype(int)
        assert set(kept) == expected_kept

    # 32px (42px with padding for the beam) and 2.5kpc: coarse cells of 4px need padding
    # on the sky; 38px (48px) and 4.5kpc: coarse cells of 8px extend the grid (24px)
    # beyond the symmetric extent of the region (22px) along the LoS
    @pytest.mark.parametrize(
        ("n_px", "los_depth"), ((32, los_depth), (38, 4.5 * U.kpc))
    )
    @pytest.mark.parametrize("adaptive_grid", (True, False))
    def test_grid_covers_region(
        self, many_particle_source, adaptive_grid, n_px, los_depth
    ):
        """
        Check that the cell grid covers the region, symmetrically about the source.

        The extent of the grid along the LoS is set by ``los_depth``, independently of
        the extent of the datacube on the sky. It is also the extent used to prune
        particles along the LoS, ``[-0.5, _n_px_z - 0.5]``, centred on the source.
        """
        source = many_particle_source()
        m = Mochi(
            source=source,
            datacube=DataCube(
                n_px_x=n_px,
                n_px_y=n_px,
                n_channels=16,
                channel_width=4 * U.km / U.s,
                spectral_centre=source.vsys,
                px_size=15 * U.arcsec,
            ),
            beam=GaussianBeam(),
            sph_kernel=CubicSplineKernel(),
            spectral_model=GaussianSpectrum(),
            los_depth=los_depth,
            interpolant=module_interpolants.sph,
            adaptive_grid=adaptive_grid,
            quiet=True,
        )
        grid = m._make_cell_grid()
        assert isinstance(grid, AdaptiveCellGrid) == adaptive_grid
        assert grid.los_region == m._los_bounds
        los_low = grid.cells["x"].min()
        los_high = (grid.cells["x"] + grid.cells["size"]).max()
        assert los_low == -0.5
        assert los_high == m._n_px_z - 0.5
        # the region is within the grid and maps back to the requested depth:
        los_centre = (m._n_px_z - 1) / 2
        lo, hi = m._los_bounds
        assert los_low <= lo < hi <= los_high
        px = m.datacube.px_size.to_value(U.rad)
        assert U.isclose(
            source.distance * np.exp((lo - los_centre) * px),
            source.distance - los_depth / 2,
        )
        assert U.isclose(
            source.distance * np.exp((hi - los_centre) * px),
            source.distance + los_depth / 2,
        )
        # the smallest whole number of cells covering the region symmetrically:
        cell_size = m.initial_cell_size if adaptive_grid else 1
        assert m._n_px_z - cell_size < 2 * (los_centre - lo) <= m._n_px_z
        # on the sky the grid covers the (padded) datacube:
        for axis, n_px in zip("yz", m.datacube.datacube_shape[:2]):
            assert grid.cells[axis].min() == -0.5
            assert (grid.cells[axis] + grid.cells["size"]).max() == n_px - 0.5

    # 32px (42px with padding for the beam) and 2.5kpc: coarse cells of 4px need padding
    # on the sky; 38px (48px) and 4.5kpc: coarse cells of 8px extend the grid (24px)
    # beyond the symmetric extent of the region (22px) along the LoS
    @pytest.mark.parametrize(
        ("n_px", "los_depth"), ((32, los_depth), (38, 4.5 * U.kpc))
    )
    @pytest.mark.parametrize("adaptive_grid", (True, False))
    def test_prune_against_grid(self, adaptive_grid, n_px, los_depth):
        """
        Check that particles are pruned against the extent of the cell grid.

        Particles that can only contribute to the parts of the grid beyond the region
        that contributes to the cube (beyond ``los_depth`` along the LoS, in the padding
        for the coarse grid on the sky) must be kept: cells straddling the edge of the
        region need their full density. Particles beyond the grid are pruned.
        """
        distance = 3 * U.Mpc
        px_size = 15 * U.arcsec
        px_kpc = (px_size * distance).to_value(U.kpc, U.dimensionless_angles())

        datacube_kwargs = {
            "n_px_x": n_px,
            "n_px_y": n_px,
            "n_channels": 16,
            "channel_width": 4 * U.km / U.s,
            "spectral_centre": distance * 70 * U.km / U.s / U.Mpc,
            "px_size": px_size,
        }

        m_reference = Mochi(
            source=SPHSource(
                xyz_g=np.zeros((1, 3)) * U.kpc,
                vxyz_g=np.zeros((1, 3)) * U.km / U.s,
                T_g=np.ones(1) * 1.0e4 * U.K,
                hsm_g=np.ones(1) * 1.0e-3 * U.kpc,
                mHI_g=np.ones(1) * 1.0e4 * U.Msun,
                distance=distance,
            ),
            datacube=DataCube(**datacube_kwargs),
            beam=GaussianBeam(),
            sph_kernel=CubicSplineKernel(),
            spectral_model=GaussianSpectrum(),
            los_depth=los_depth,
            interpolant=module_interpolants.sph,
            adaptive_grid=adaptive_grid,
            quiet=True,
        )
        n_px_z = m_reference._n_px_z
        # distance of the far edge of the grid along the LoS:
        grid_far = distance * np.exp(n_px_z / 2 * px_size.to_value(U.rad))
        region_far = distance + los_depth / 2
        assert grid_far > region_far  # the grid extends beyond the region
        x_overhang = ((region_far + grid_far) / 2 - distance).to_value(U.kpc)
        x_beyond = (grid_far - distance).to_value(U.kpc) + 0.05
        # on the sky the grid extends from -0.5 to n - 0.5 pixels, the source is at the
        # centre and pixel x increases with decreasing source y
        n_sky = m_reference.datacube.datacube_shape[0]
        sky_edge_px = n_sky / 2  # from the centre to the outer edge of the grid
        coarse_pad = m_reference.datacube.padx - m_reference.beam.needs_pad()[0]
        y_pad = -(sky_edge_px - coarse_pad / 2) * px_kpc  # in the coarse padding
        y_beyond = -(sky_edge_px + 0.2) * px_kpc
        xyz_g = U.Quantity(
            [
                [0, 0, 0],  # kept
                [x_overhang, 0, 0],  # beyond the region, within the grid, kept
                [x_beyond, 0, 0],  # beyond the grid, pruned
                [0, y_pad, 0],  # in the coarse padding (if any), kept
                [0, y_beyond, 0],  # beyond the grid on the sky, pruned
            ],
            U.kpc,
        )
        n = xyz_g.shape[0]
        m = Mochi(
            source=SPHSource(
                xyz_g=xyz_g,
                vxyz_g=np.zeros((n, 3)) * U.km / U.s,
                T_g=(1.0e4 + np.arange(n)) * U.K,  # label particles to identify them
                hsm_g=np.ones(n) * 1.0e-3 * U.kpc,
                mHI_g=np.ones(n) * 1.0e4 * U.Msun,
                distance=distance,
            ),
            datacube=DataCube(**datacube_kwargs),
            beam=GaussianBeam(),
            sph_kernel=CubicSplineKernel(),
            spectral_model=GaussianSpectrum(),
            los_depth=los_depth,
            interpolant=module_interpolants.sph,
            adaptive_grid=adaptive_grid,
            quiet=True,
        )
        assert m._n_px_z == n_px_z
        if adaptive_grid and n_px == 32:
            assert coarse_pad > 0  # check that the coarse padding is tested
        kept = np.rint(m.source.T_g.to_value(U.K) - 1.0e4).astype(int)
        assert set(kept) == {0, 1, 3}

    def test_preferred_kernel_ranges(self):
        """
        Check that kernel ranges are those of the preferred kernel of adaptive kernels.

        Mochi evaluates the preferred kernel over 3D cells for all particles, so
        particles that Martini's validation (for 2D pixels) assigns a fallback kernel
        (here a ``DiracDeltaKernel``, narrower than the cubic spline) must be pruned
        (and grids refined) with the reach of the preferred kernel.
        """
        distance = 3 * U.Mpc
        px_size = 15 * U.arcsec
        px_kpc = (px_size * distance).to_value(U.kpc, U.dimensionless_angles())
        hsm = 0.1  # kpc, about 0.46px: too small for the cubic spline in martini
        preferred, fallback = CubicSplineKernel().kernels[:2]
        assert isinstance(fallback, DiracDeltaKernel)
        assert fallback.size_in_fwhm < preferred.size_in_fwhm
        h_px = hsm / px_kpc

        def make_mochi(xyz_g):
            """
            Create a Mochi instance with particles at the given positions.

            Parameters
            ----------
            xyz_g : ~astropy.units.Quantity
                Particle positions.

            Returns
            -------
            ~martini.mochi.mochi.Mochi
                The Mochi instance.
            """
            n = xyz_g.shape[0]
            return Mochi(
                source=SPHSource(
                    xyz_g=xyz_g,
                    vxyz_g=np.zeros((n, 3)) * U.km / U.s,
                    T_g=(1.0e4 + np.arange(n))
                    * U.K,  # label particles to identify them
                    hsm_g=np.ones(n) * hsm * U.kpc,
                    mHI_g=np.ones(n) * 1.0e4 * U.Msun,
                    distance=distance,
                    coordinate_axis=1,
                ),
                datacube=DataCube(
                    n_px_x=32,
                    n_px_y=32,
                    n_channels=16,
                    channel_width=4 * U.km / U.s,
                    spectral_centre=distance * 70 * U.km / U.s / U.Mpc,
                    px_size=px_size,
                ),
                beam=GaussianBeam(),
                sph_kernel=CubicSplineKernel(),
                spectral_model=GaussianSpectrum(),
                los_depth=self.los_depth,
                interpolant=module_interpolants.sph,
                quiet=True,
            )

        # pixel x increases with decreasing source y, from the centre to the edge of the
        # grid is half of its (padded) width:
        sky_edge_px = (
            make_mochi(np.zeros((1, 3)) * U.kpc).datacube.datacube_shape[0] / 2
        )
        between = (fallback.size_in_fwhm + preferred.size_in_fwhm) / 2
        beyond = 1.2 * preferred.size_in_fwhm
        m = make_mochi(
            U.Quantity(
                [
                    [0, 0, 0],  # kept
                    [0, -(sky_edge_px + between * h_px) * px_kpc, 0],  # within reach
                    [0, -(sky_edge_px + beyond * h_px) * px_kpc, 0],  # out of reach
                ],
                U.kpc,
            )
        )
        kept = np.rint(m.source.T_g.to_value(U.K) - 1.0e4).astype(int)
        assert set(kept) == {0, 1}
        assert np.all(m.sph_kernel.kernel_indices == 1)  # martini falls back
        expected = m.sph_kernel.sm_lengths * preferred.size_in_fwhm
        assert U.allclose(m.sph_kernel.sm_ranges, expected)
        m.insert_source_in_cube()
        assert U.allclose(m.sph_kernel.sm_ranges, expected)

    def test_pixcoords_match_padded_datacube(self, many_particle_source, dc_zeros):
        """
        Check particle pixel coordinates after padding the datacube for the coarse grid.

        The datacube is padded for the coarse grid after the particle pixel coordinates
        are first evaluated, they must be consistent with the padded datacube.
        """
        beam = GaussianBeam()
        m = Mochi(
            source=many_particle_source(),
            datacube=dc_zeros,
            beam=beam,
            sph_kernel=CubicSplineKernel(),
            spectral_model=GaussianSpectrum(),
            los_depth=self.los_depth,
            interpolant=module_interpolants.sph,
            quiet=True,
        )
        # the coarse grid needs additional padding in this configuration:
        assert m.datacube.padx > beam.needs_pad()[0]
        skycoords = m.source.skycoords.transform_to(m.datacube.coordinate_frame)
        expected = m.datacube.wcs.celestial.wcs_world2pix(
            skycoords.ra.to_value(m.datacube.units[0]),
            skycoords.dec.to_value(m.datacube.units[1]),
            0,
        )
        assert np.allclose(m.source.pixcoords[0].to_value(U.pix), expected[0])
        assert np.allclose(m.source.pixcoords[1].to_value(U.pix), expected[1])

    @pytest.mark.parametrize("n_px", [(15, 16), (16, 15)])
    def test_odd_datacube_raises(self, many_particle_source, n_px):
        """Check that a datacube with an odd number of pixels is rejected."""
        datacube = DataCube(
            n_px_x=n_px[0],
            n_px_y=n_px[1],
            n_channels=16,
            px_size=10 * U.arcsec,
            channel_width=10 * U.km / U.s,
        )
        with pytest.raises(ValueError, match="even number of pixels"):
            Mochi(
                source=many_particle_source(),
                datacube=datacube,
                sph_kernel=CubicSplineKernel(),
                spectral_model=DiracDeltaSpectrum(),
                los_depth=self.los_depth,
                interpolant=module_interpolants.sph,
            )

    @pytest.mark.parametrize("interpolant", interpolants)
    @pytest.mark.parametrize("refinement_strategy", refinement_strategies)
    @pytest.mark.parametrize("spectral_model", spectral_models)
    @pytest.mark.parametrize("adaptive_grid", [True, False])
    def test_insert_source_in_cube(
        self,
        many_particle_source,
        dc_zeros,
        interpolant,
        refinement_strategy,
        spectral_model,
        adaptive_grid,
    ):
        """Check that API call resolves without error."""
        datacube = dc_zeros
        source = many_particle_source(hsm_g=0.5 * U.kpc)
        beam = GaussianBeam()
        spectral_model = spectral_model()
        sph_kernel = CubicSplineKernel()
        m = Mochi(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
            los_depth=self.los_depth,
            interpolant=interpolant,
            adaptive_grid=adaptive_grid,
            refinement_strategy=refinement_strategy,
        )
        m.insert_source_in_cube()

    @pytest.mark.parametrize("adaptive_grid", [True, False])
    def test_mem_lim(self, monkeypatch, many_particle_source, dc_zeros, adaptive_grid):
        """Check that the memory limit batches the interpolation onto the cell grid."""

        def make_mochi():
            return Mochi(
                source=many_particle_source(hsm_g=0.5 * U.kpc),
                datacube=dc_zeros.copy(),
                sph_kernel=CubicSplineKernel(),
                spectral_model=DiracDeltaSpectrum(),
                los_depth=self.los_depth,
                interpolant=module_interpolants.mfm,
                adaptive_grid=adaptive_grid,
                quiet=True,
            )

        m = make_mochi()
        with pytest.raises(RuntimeError, match="less than estimated baseline"):
            m.insert_source_in_cube(mem_lim_GB=m.baseline_mem_estimate / 2)
        m = make_mochi()
        estimates = []
        estimate = module_interpolants._scatter_mem_estimate

        def record_estimate(*args):
            estimates.append(estimate(*args))
            return estimates[-1]

        monkeypatch.setattr(
            module_interpolants, "_scatter_mem_estimate", record_estimate
        )
        m.insert_source_in_cube()  # default limit: all at once
        per_particle, per_batch, fixed = estimates[0]
        allowance = fixed + per_batch + max(per_particle.max(), per_particle.sum() / 2)
        m_batched = make_mochi()
        with pytest.warns(RuntimeWarning, match="Processing in batches"):
            m_batched.insert_source_in_cube(
                mem_lim_GB=m_batched.baseline_mem_estimate + allowance
            )
        assert U.allclose(
            m_batched.datacube._array,
            m.datacube._array,
            rtol=1e-10,
            atol=1e-12 * np.abs(m.datacube._array).max(),
        )


def check_inserted_mass_accuracy(m, expected_fraction=1.0):
    """
    Check that mass in cube after source insertion matches input mass by better than 1%.

    Unlike :func:`check_mass_accuracy` the mass in the padded region of the cube is
    included and the beam convolution and output steps are not checked.

    Parameters
    ----------
    m : ~martini.mochi.mochi.Mochi
        The mochi instance to test for accuracy.

    expected_fraction : float, optional
        The fraction of the input mass expected in the cube.
    """
    # flux in channels
    F = (m.datacube._array * m.datacube.px_size**2).sum((0, 1)).squeeze()  # Jy

    # channel width
    dv = np.abs(np.diff(m.datacube.velocity_channel_edges))

    # HI mass
    MHI = np.sum(
        2.36e5
        * U.Msun
        * m.source.distance.to_value(U.Mpc) ** 2
        * F.to_value(U.Jy)
        * dv.to_value(U.km / U.s)
    ).to(U.Msun)

    assert U.isclose(MHI, expected_fraction * m.source.mHI_g.sum(), rtol=1e-2)


class TestMochiMassAccuracy:
    """Check that the mass of a single particle is conserved by the MOCHI backend."""

    @pytest.mark.parametrize("sph_kernel", simple_kernels)
    @pytest.mark.parametrize("interpolant", interpolants, ids=interpolant_ids)
    @pytest.mark.parametrize("adaptive_grid", (True, False))
    @pytest.mark.parametrize("spectral_model", spectral_models)
    @pytest.mark.parametrize("out_mode", ("fits", "hdf5"))
    def test_mass_accuracy(
        self,
        dc_zeros,
        sph_kernel,
        interpolant,
        adaptive_grid,
        spectral_model,
        single_particle_source,
        out_mode,
        tmp_path_factory,
    ):
        """
        Check that input mass in particles ends up in the output cube.

        Check that the input mass in a single particle gives the correct total mass in
        the datacube, by checking the conversion back to total mass. Covers testing
        Mochi.insert_source_in_cube.
        """
        kernel_name = sph_kernel.__name__
        interpolant_name = interpolant_ids[interpolants.index(interpolant)]
        hsm_g = 0.1 * U.kpc if kernel_name == "DiracDeltaKernel" else 1.0 * U.kpc
        source = single_particle_source(hsm_g=hsm_g)
        # single_particle_source has a mass of 1E4Msun, temperature of 1E4K
        mochi_kwargs = dict(
            source=source,
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=spectral_model(),
            sph_kernel=sph_kernel(),
            los_depth=8 * U.kpc,  # deep enough to contain the particle's kernel
            interpolant=interpolant,
            adaptive_grid=adaptive_grid,
            quiet=True,
        )
        if kernel_name == "DiracDeltaKernel" and interpolant_name == "manual_sph":
            # manual_sph cannot sample a Dirac-delta kernel:
            with pytest.raises(ValueError, match="DiracDeltaKernel"):
                Mochi(**mochi_kwargs)
            return
        if kernel_name != "DiracDeltaKernel" and interpolant_name == "particle_in_cell":
            with pytest.warns(UserWarning, match="particle_in_cell"):
                m = Mochi(**mochi_kwargs)
        else:
            m = Mochi(**mochi_kwargs)
        m.insert_source_in_cube()
        if interpolant_name == "voronoi_mesh":
            # A single Voronoi cell fills the whole grid uniformly, including the padded
            # region that is cropped after beam convolution, so only check after
            # insertion. Along the LoS the grid extends beyond the region, and the mass
            # beyond the region is excluded.
            los_lo, los_hi = m._los_bounds
            check_inserted_mass_accuracy(
                m, expected_fraction=(los_hi - los_lo) / m._n_px_z
            )
        else:
            check_mass_accuracy(m, out_mode, tmp_path_factory.mktemp("cube"))
