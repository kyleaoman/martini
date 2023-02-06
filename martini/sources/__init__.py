# generic source class:
from .sph_source import SPHSource

# debugging source classes:
from ._single_particle_source import _SingleParticleSource
from ._cross_source import _CrossSource

# toolkit-specific source classes:
from .so_source import SOSource
from .swiftgalaxy_source import SWIFTGalaxySource

# simulation-specific source classes:
from .colibre_source import ColibreSource
from .eagle_source import EAGLESource
from .magneticum_source import MagneticumSource
from .simba_source import SimbaSource
from .tng_source import TNGSource
