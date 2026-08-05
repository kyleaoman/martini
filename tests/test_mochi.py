"""Tests of the mochi sub-module."""

from martini.mochi.mochi import Mochi
from martini.beams import GaussianBeam
from martini.spectral_models import GaussianSpectrum, DiracDeltaSpectrum
from martini.sph_kernels import CubicSplineKernel
import astropy.units as U
from martini.mochi.mochi import AdaptiveCellGrid
from martini.mochi import interpolants
from martini.mochi import refinement
from martini.mochi._dtypes import CELL_DTYPE
import pytest
import numpy as np
import itertools

interpolants = (
    interpolants.sph,
    interpolants.mfm,
    interpolants.voronoi_mesh,
    interpolants.manual_sph,
)
refinement_strategies = (
    refinement.refine_grid_to_half_particle_scale,
    refinement.refine_grid_to_single_occupancy,
)
spectral_models = (GaussianSpectrum, DiracDeltaSpectrum)


class TestAdaptiveCellGridUtils:
    """Test adaptive cell grid utils."""

    def test_refine_grid_bisect(self):
        """Check grid bisect operation for a (0,0,0,2) cell."""
        from martini.mochi.refinement import _refine_grid_bisect

        cell = np.array((0, 0, 0, 2), dtype=CELL_DTYPE)
        mask = np.ones(3, dtype=bool)
        incell = np.ones(3, dtype=bool)
        new_cells = []
        new_cells_over = []
        new_cells_masks = []
        _refine_grid_bisect(
            cell, mask, incell, new_cells, new_cells_over, new_cells_masks
        )
        new_cells_correct = [
            np.array([0, 0, 0, 1], dtype=float),
            np.array([0, 0, 1, 1], dtype=float),
            np.array([0, 1, 1, 1], dtype=float),
            np.array([0, 1, 0, 1], dtype=float),
            np.array([1, 0, 0, 1], dtype=float),
            np.array([1, 0, 1, 1], dtype=float),
            np.array([1, 1, 1, 1], dtype=float),
            np.array([1, 1, 0, 1], dtype=float),
        ]
        assert len(new_cells) == len(new_cells_correct)
        for cell in new_cells:
            cell_is_good = False
            for i in range(len(new_cells_correct)):
                correct_cell = new_cells_correct[i]
                if np.all(correct_cell == cell):
                    cell_is_good = True
                    del new_cells_correct[i]
                    break
            assert cell_is_good

    def test_pass_complete_cell(self):
        """Check that _pass_complete_cell correctly passes contents."""
        from martini.mochi.refinement import _pass_complete_cell

        cells_lists = [["a"], [1]]
        content_list = ["b", 2]
        _pass_complete_cell(cells_lists, content_list)
        assert cells_lists[0][-1] == content_list[0]
        assert cells_lists[1][-1] == content_list[1]

    def test_refine_grid(self):
        """Test that refine grid passes correctly when bisect condition is not met."""

        def in_cell_condition(mask, positions, radii, cell):
            return np.ones(np.count_nonzero(mask), dtype=bool)

        def bisect_condition(in_cell):
            return False

        from martini.mochi.refinement import _refine_grid

        input_cells = (np.array([(0, 0, 0, 2)], dtype=CELL_DTYPE),)  # cells

        cells = _refine_grid(
            in_cell_condition,
            bisect_condition,
            input_cells,
            np.array([[0, 0, 0], [1, 1, 1]], dtype=float),  # positions
            np.array([1, 1], dtype=float),  # radii
        )

        assert np.all(input_cells == cells)  # null case bc bisect condition is false
        """this test could be extended to a non-null case: ex test_refine_grid_bisect"""

    def test_occupancy_in_cell(self):
        """Check that only particles inside cell are selected."""
        cell = np.array((10.0, 100.0, 1000.0, 1.0), dtype=CELL_DTYPE)
        mask = [True, True]
        particles_pos = np.array([[10.5, 100.5, 1000.5], [10.5, 100.5, -1000.5]])
        particles_radii = np.array([1, 1e7])  # radii should not impact occupancy
        result = refinement._occupancy_in_cell(
            mask, particles_pos, particles_radii, cell
        )
        correct_result = np.array([True, False])
        assert np.all(correct_result == result)

    @pytest.mark.parametrize(
        "count, mask_count", list(itertools.product([10, 3, 2], [100, 4, 2, 0]))
    )
    def test_has_more_than(self, count, mask_count):
        """Check that _has_more_than for different counts and masks."""
        mask_size = 1000
        mask = np.zeros(mask_size, dtype=bool)
        mask[:mask_count] = True
        assert refinement._has_more_than(count, mask) == (count < mask_count)

    def test_intersect_in_cell(self):
        """Check intersect_in_cell selects for small particles intersecting."""
        cell = np.array((10.0, 100.0, 1000.0, 1.0), dtype=CELL_DTYPE)
        particles_pos = np.array(
            [
                [10.5, 100.5, 1000.5],
                [10.5, 100.5, -1000.5],
                [10.5, 100.5, -1000.5],
                [10.5, 100.5, 1001.1],
                [11, 101, 1001.11],  # intersect_in_cell approximates, using worst case
            ]
        )
        particles_radii = np.array([1, 1e7, 0.1, 0.25, 0.01])  # radii should impact
        mask = [
            True,
        ] * len(particles_radii)
        result = refinement._intersect_in_cell(
            0.5, mask, particles_pos, particles_radii, cell
        )
        correct_result = np.array([True, False, False, True, False])
        assert np.all(correct_result == result)

    def test_refine_grid_to_occupancy(self):
        """TBD."""
        raise NotImplementedError

    def test_refine_grid_to_particle_scale(self):
        """TBD."""
        raise NotImplementedError


class TestAdaptiveCellGrid:
    """Test functionality of adaptive cell grid class."""

    def test_init(self, many_particle_source, dc_zeros):
        """Check class initialization."""
        datacube = dc_zeros
        initial_grid_size = 3
        acg = AdaptiveCellGrid(datacube, initial_grid_size=initial_grid_size)
        # insert checks on acg.pix_range
        assert len(acg.initial_cells) == initial_grid_size**3
        # insert checks on cell locations & sizes

    def test_init_particle_locations(self, many_particle_source, dc_zeros):
        """Test actual pixel coordinates in source module tests."""
        datacube = dc_zeros
        source = many_particle_source()
        source._init_skycoords()
        source._init_pixcoords(datacube)
        source._init_los_pixcoords(datacube)
        sph_kernel = CubicSplineKernel()
        sph_kernel._init_sm_lengths(source, datacube)
        sph_kernel._init_sm_ranges()
        acg = AdaptiveCellGrid(datacube)
        acg.init_particle_locations(source, sph_kernel)
        assert acg.positions.shape == (source.hsm_g.size, 3)
        assert acg.radii.shape == (source.hsm_g.size,)
        assert acg.positions.unit == U.pix
        assert acg.radii.unit == U.pix

    def test_eval_grid_refinement(self):
        """TBD."""
        raise NotImplementedError

    def test_init_cell_centres(self):
        """TBD."""
        raise NotImplementedError

    def test_init_cell_volumes(self):
        """TBD."""
        raise NotImplementedError

    def test_interpolate_fields(self):
        """TBD."""
        raise NotImplementedError

    def test_create_regular_array(self):
        """TBD."""
        raise NotImplementedError

    def eval_radiative_transfer(self):
        """TBD."""
        raise NotImplementedError


class TestMochi:
    """Main API test for MOCHI backend."""

    # Testing *all* combinations here is overkill, simplify later.
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
            interpolant=interpolant,
            adaptive_grid=adaptive_grid,
            refinement_strategy=refinement_strategy,
        )
        m.insert_source_in_cube()
