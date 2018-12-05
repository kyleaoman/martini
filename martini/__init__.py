"""
Mock APERTIF-like Radio Telescope Interferometry of the Neutral ISM

MARTINI is a modular package for the creation of synthetic resolved HI line
observations (data cubes) of smoothed-particle hydrodynamics simulations of
galaxies. The various aspects of the mock-observing process are divided
logically into sub-modules handling the data cube, source, beam, noise,
spectral model and SPH kernel. MARTINI is object-oriented: each sub-module
provides a class (or classes) which can be configured as desired. For most
sub-modules, base classes are provided to allow for straightforward
customization. Instances of each sub-module class are then given as parameters
to the Martini class. A mock observation is then constructed by calling a
handful of functions to execute the desired steps in the mock-observing
process.

The package is functional and (an old version) has been used in a forthcoming
paper (https://ui.adsabs.harvard.edu/#abs/2017arXiv170607478O/abstract). It is
still undergoing active development and is not quite yet at the point of a
stable release version. Things will change, bugs will happen. Any feedback is
greatly appreciated.

See the help for martini.Martini for an example script to configure MARTINI
and create a datacube. This example can be run by doing:

from martini import demo
demo()

If your use of MARTINI leads to a publication, please acknowledge this and link
to the github page, ideally specifying the version used (git commit ID).

Suport available via koman@astro.rug.nl.
"""

from ._martini import Martini
from ._datacube import DataCube
from ._demo import demo
