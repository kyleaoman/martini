"""Tests of the mochi sub-module."""

from martini.mochi.mochi import Mochi
from martini.beams import GaussianBeam
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import CubicSplineKernel
import astropy.units as U
from martini.mochi.mochi import AdaptiveCellGrid
from martini.mochi import interpolants
from martini.mochi import radiative_transfer
from martini.mochi import refinement
import pytest

interpolants = (
    interpolants.sph,
    interpolants.mfm,
    interpolants.voronoi_mesh,
    interpolants.manual_sph,
)
rt_methods = (
    radiative_transfer.optically_thin,
    radiative_transfer.adaptive_optically_thin,
)
refinement_strategies = (
    refinement.refine_grid_to_particle_scale,
    refinement.refine_grid_to_occupancy,
)


class TestAdaptiveCellGridUtils:
    def test_refine_grid_bisect(self):
        raise NotImplementedError

    def test_pass_complete_cell(self):
        raise NotImplementedError

    def test_refine_grid(self):
        raise NotImplementedError

    def test_occupancy_in_cell(self):
        raise NotImplementedError

    def test_is_not_single_occupancy(self):
        raise NotImplementedError

    def test_refine_grid_to_occupancy(self):
        raise NotImplementedError

    def test_intersect_in_cell(self):
        raise NotImplementedError

    def test_is_any_particle_included(self):
        raise NotImplementedError

    def test_refine_grid_to_particle_scale(self):
        raise NotImplementedError


class TestAdaptiveCellGrid:
    def test_init(self, many_particle_source, dc_zeros):
        datacube = dc_zeros
        initial_grid_size = 3
        acg = AdaptiveCellGrid(datacube, initial_grid_size=initial_grid_size)
        # insert checks on acg.pix_range
        assert len(acg.initial_cells) == initial_grid_size**3
        # insert checks on cell locations & sizes

    def test_init_particle_locations(self, many_particle_source, dc_zeros):
        # test actual pixel coordinates in source module tests
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
        raise NotImplementedError

    def test_init_cell_centres(self):
        raise NotImplementedError

    def test_init_cell_volumes(self):
        raise NotImplementedError

    def test_interpolate_fields(self):
        raise NotImplementedError

    def test_create_regular_array(self):
        raise NotImplementedError

    def eval_radiative_transfer(self):
        raise NotImplementedError


class TestMochi:
    @pytest.mark.parametrize("interpolant", interpolants)
    @pytest.mark.parametrize("radiative_transfer_method", rt_methods)
    @pytest.mark.parametrize("refinement_strategy", refinement_strategies)
    def test_insert_source_in_cube(
        self,
        many_particle_source,
        dc_zeros,
        interpolant,
        radiative_transfer_method,
        refinement_strategy,
    ):
        datacube = dc_zeros
        source = many_particle_source(hsm_g=0.5 * U.kpc)
        beam = GaussianBeam()
        spectral_model = GaussianSpectrum()
        sph_kernel = CubicSplineKernel()
        m = Mochi(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
            interpolant=interpolant,
            radiative_transfer=radiative_transfer_method,
            refinement_strategy=refinement_strategy,
        )
        if radiative_transfer_method is radiative_transfer.optically_thin:
            # So far only adaptive cube is implemented, non-adaptive RT is not compatible:
            with pytest.raises(ValueError, match="cannot reshape array"):
                m.insert_source_in_cube()
            return
        m.insert_source_in_cube()
