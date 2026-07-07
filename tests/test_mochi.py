import matplotlib.pyplot as plt
import numpy as np
from martini.mochi.mochi import Mochi
from martini.martini import Martini
from martini.beams import GaussianBeam
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import CubicSplineKernel
import astropy.units as U
from copy import deepcopy
from martini.mochi.mochi import AdaptiveCellGrid


def test_scratch(many_particle_source, dc_zeros):
    # This will try to run with 4 variants of dc_zeros.
    # Use dc_zeros1 (stokes + vchannels) for now.
    martini_dc = deepcopy(dc_zeros)
    hsm_g = 0.3 * U.kpc
    mochi = Mochi(
        source=many_particle_source(
            hsm_g=hsm_g,
        ),  # refined cube changes by factors of 2 with this choice
        datacube=dc_zeros,
        beam=GaussianBeam(),
        noise=None,
        spectral_model=GaussianSpectrum(),
        sph_kernel=CubicSplineKernel(),
    )
    mochi.insert_source_in_cube()

    martini = Martini(
        source=many_particle_source(hsm_g=hsm_g),
        datacube=martini_dc,
        beam=GaussianBeam(),
        noise=None,
        spectral_model=GaussianSpectrum(),
        sph_kernel=CubicSplineKernel(),
    )
    martini.insert_source_in_cube()

    plt.imsave("martini.png", np.sum(martini.datacube._array, axis=(2, 3)))
    plt.imsave("mochi.png", np.sum(mochi.datacube._array, axis=(2, 3)))


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
    def test_insert_source_in_cube(self, many_particle_source, dc_zeros):
        # should test different threshold and refine_algorithm values
        datacube = dc_zeros
        source = many_particle_source()
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
        )
        m.insert_source_in_cube()
