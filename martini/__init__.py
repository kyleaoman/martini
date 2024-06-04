from .martini import Martini, GlobalProfile
from .datacube import DataCube
from ._demo import demo, demo_source
from .__version__ import __version__

__doc__ = """
    Mock APERTIF-like Radio Telescope Interferometry of the Neutral ISM

    MARTINI is a modular package for the creation of synthetic resolved HI line
    observations (data cubes) of smoothed-particle hydrodynamics simulations of
    galaxies. The various aspects of the mock-observing process are divided
    logically into sub-modules handling the data cube, source, beam, noise,
    spectral model and SPH kernel. MARTINI is object-oriented: each sub-module
    provides a class (or classes) which can be configured as desired. For most
    sub-modules, base classes are provided to allow for straightforward
    customization. Instances of each sub-module class are then given as
    parameters to the Martini class. A mock observation is then constructed by
    calling a handful of functions to execute the desired steps in the
    mock-observing process.
"""
