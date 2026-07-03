from martini.martini import Mochi
from martini.beams import GaussianBeam
from martini.spectral_models import DiracDeltaSpectrum
from martini.sph_kernels import DiracDeltaKernel


class TestMochi:
    def test_scratch(self, single_particle_source, dc_zeros):
        m = Mochi(
            source=single_particle_source(),
            datacube=dc_zeros,
            beam=GaussianBeam(),
            noise=None,
            spectral_model=DiracDeltaSpectrum(),
            sph_kernel=DiracDeltaKernel(),
        )
        m.insert_source_in_cube()
