.. image:: martini_banner.png
	   
Overview
========

MARTINI is a modular package for the creation of synthetic resolved HI line observations (data cubes) of smoothed-particle hydrodynamics simulations of galaxies. The various aspects of the mock-observing process are divided logically into sub-modules handling the data cube, source, beam, noise, spectral model and SPH kernel. MARTINI is object-oriented: each sub-module provides a class (or classes) which can be configured as desired. For most sub-modules, base classes are provided to allow for straightforward customization. Instances of each sub-module class are given as parameters to the Martini class; a mock observation is then constructed by calling a handful of functions to execute the desired steps in the mock-observing process.

The package is fully functional and (an old version) has been used in this paper_. Stable releases are available via PyPI (``pip install astromartini``) and the numbered branches on github. The github master branch is under active development: things will change, bugs will happen. Any feedback is greatly appreciated.

.. _paper: https://ui.adsabs.harvard.edu/#abs/2019MNRAS.482..821O/abstract

MARTINI does not support use with python2.

See the help for martini.Martini_ for an example script to configure MARTINI and create a datacube. This example can be run by doing::
  
  python -c "from martini import demo; demo()"

.. _martini.Martini: https://kyleaoman.github.io/martini/build/html/martini.html

Martini has (so far) been successfully run on the output of these simulations:

- EAGLE
- APOSTLE
- C-EAGLE/Hydrangea
- Illustris
- IllustrisTNG
- Auriga
- MaGICC (and therefore in principle NIHAO)
- Magneticum
- Simba

I attempt to support publicly available simulations with a customized source module. If your simulation is public and not supported, please contact me at the address below. Currently custom source modules exist for:

- EAGLE (martini.sources.EAGLESource_)
- IllustrisTNG (martini.sources.TNGSource_; also works with Illustris)
- Magneticum (martini.sources.MagneticumSource_)
- Simba (martini.sources.SimbaSource_)

.. _martini.sources.EAGLESource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.EAGLESource
.. _martini.sources.TNGSource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.TNGSource
.. _martini.sources.MagneticumSource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.MagneticumSource
.. _martini.sources.SimbaSource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.SimbaSource
   
If your use of MARTINI leads to a publication, please acknowledge this and link to the github page, ideally specifying the version used (git commit ID or version number). Suport available via kyle.a.oman@durham.ac.uk.

.. _kyle.a.oman@durham.ac.uk: mailto:kyle.a.oman@durham.ac.uk

Installation Notes
==================

The easiest way to install martini is from PyPI by doing ``python3 -m pip install astromartini``; python2 is not supported. Output to ``.fits`` files is supported by default; if output to ``.hdf5`` format is desired use ``python3 -m pip install astromartini[hdf5_output]`` instead. This will also handle the installation of the required dependencies. However, some optional features require additional dependencies hosted on github, and PyPI does not allow installing these automatically. In particular, EAGLE and Illustris/TNG users who wish to use the custom source modules for those simulations in Martini must install from github (see below) to automatically install the optional dependencies. Or, it is also possible to install from PyPI and then manually install the optional dependencies.

Installation by doing ``python setup.py install`` is not recommended.

Installing from github
----------------------

Choose a branch_. The numbered branches (e.g. 1.0.X) are stable, while the master branch is actively developed. The latest numbered branch is usually the best choice. From the branch page (e.g. ``https://github.com/kyleaoman/martini/tree/1.0.X``), click the green 'Clone or download' button and follow instructions to obtain the files. Unpack the zip file if necessary. You should then be able to do ``python3 -m pip install martini/[optional]``, where ``optional`` should be replaced by a comma separated list of optional dependencies. If this fails check that ``martini/`` is a path pointing to the directory containing the ``setup.py`` file for Martini. The currently available options are:

- ``hdf5_output``: Supports output to hdf5 files via the h5py package. Since h5py is hosted on PyPI, this option may be used when installing via PyPI.
- ``eaglesource``: Dependencies for the EAGLESource_ module, which greatly simplifies reading input from EAGLE simulation snapshots. Installs my Hdecompose_ package, providing implementations of the `Rahmati et al. (2013)`_ method for computing netural hydrogen fractions and the `Blitz & Rosolowsky (2006)`_ method for atomic/molecular fractions. Also installs `my python-only version`_ of John Helly's `read_eagle`_ package for quick extraction of particles in a simulation sub-volume. h5py is also required.
- ``tngsource``: Dependencies for the TNGSource_ module, which greatly simplifies reading input from IllustrisTNG (and original Illustris) snapshots. Installs my Hdecompose_ package, providing implementations of the `Rahmati et al. (2013)`_ method for computing netural hydrogen fractions and the `Blitz & Rosolowsky (2006)`_ method for atomic/molecular fractions.
- ``magneticumsource``: Dependencies for the MagneticumSource_ module, which supports the Magneticum simulations via `my fork`_ of the `g3t`_ package by Antonio Ragagnin.
- ``sosource``: Dependencies for the SOSource_ module, which provides unofficial support for several simulation datasets hosted on specific systems. This is intended mostly for my own use, but APOSTLE, C-EAGLE/Hydrangea and Auriga users may contact_ me for further information.

.. _branch: https://github.com/kyleaoman/martini/branches
.. _EAGLESource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.EAGLESource
.. _Hdecompose: https://github.com/kyleaoman/Hdecompose
.. _`Rahmati et al. (2013)`: https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.2427R/abstract
.. _`Blitz & Rosolowsky (2006)`: https://ui.adsabs.harvard.edu/abs/2006ApJ...650..933B/abstract
.. _`my python-only version`: https://github.com/kyleaoman/pyread_eagle
.. _`read_eagle`: https://github.com/jchelly/read_eagle
.. _TNGSource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.TNGSource
.. _MagneticumSource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.MagneticumSource
.. _`my fork`: https://github.com/kyleaoman/g3t
.. _`g3t`: https://gitlab.lrz.de/di29bop/g3t
.. _SOSource: https://kyleaoman.github.io/martini/build/html/source.html#martini.sources.SOSource
.. _contact: mailto:kyle.a.oman@durham.ac.uk
