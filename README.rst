.. image:: martini_banner.png

|Build status| |Documentation status|

.. |Build status| image:: https://github.com/kyleaoman/martini/actions/workflows/code_quality.yml/badge.svg
    :target: https://github.com/kyleaoman/martini/actions/workflows/code_quality.yml
    :alt: Build Status
.. |Documentation status| image:: https://readthedocs.org/projects/martini/badge/?version=latest
    :target: https://martini.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation status
	   
Overview
========

.. INTRO_START_LABEL

MARTINI is a modular package for the creation of synthetic resolved HI line observations (data cubes) of smoothed-particle hydrodynamics simulations of galaxies. The various aspects of the mock-observing process are divided logically into sub-modules handling the data cube, source, beam, noise, spectral model and SPH kernel. MARTINI is object-oriented: each sub-module provides a class (or classes) which can be configured as desired. For most sub-modules, base classes are provided to allow for straightforward customization. Instances of each sub-module class are given as parameters to the Martini class; a mock observation is then constructed by calling a handful of functions to execute the desired steps in the mock-observing process.

.. INTRO_END_LABEL

Citing MARTINI
--------------

.. CITING_START_LABEL
   
If your use of MARTINI leads to a publication, please acknowledge this and link to the github page, ideally specifying the version used (git commit ID or version number). Please cite the `MARTINI entry`_ in the ASCL_ (`indexed on ADS`_), and the `original paper`_:

.. code-block:: bibtex

   @MISC{2019ascl.soft11005O,
        author = {{Oman}, Kyle A.},
         title = "{MARTINI: Mock spatially resolved spectral line observations of simulated galaxies}",
      keywords = {Software},
  howpublished = {Astrophysics Source Code Library, record ascl:1911.005},
          year = 2019,
         month = nov,
           eid = {ascl:1911.005},
         pages = {ascl:1911.005},
 archivePrefix = {ascl},
        eprint = {1911.005},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2019ascl.soft11005O},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
   
   @ARTICLE{2019MNRAS.482..821O,
        author = {{Oman}, Kyle A. and {Marasco}, Antonino and {Navarro}, Julio F. and {Frenk}, Carlos S. and {Schaye}, Joop and {Ben{\'\i}tez-Llambay}, Alejandro},
         title = "{Non-circular motions and the diversity of dwarf galaxy rotation curves}",
       journal = {\mnras},
      keywords = {ISM: kinematics and dynamics, galaxies: haloes, galaxies: structure, dark matter, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics},
          year = 2019,
         month = jan,
        volume = {482},
        number = {1},
         pages = {821-847},
           doi = {10.1093/mnras/sty2687},
 archivePrefix = {arXiv},
        eprint = {1706.07478},
  primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2019MNRAS.482..821O},
       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


.. _MARTINI entry: https://ascl.net/1911.005
.. _ASCL: https://ascl.net
.. _indexed on ADS: https://ui.adsabs.harvard.edu/abs/2019ascl.soft11005O/abstract
.. _original paper: https://ui.adsabs.harvard.edu/abs/2019MNRAS.482..821O/abstract

.. CITING_END_LABEL

Installation Notes
==================

.. INSTALLATION_NOTES_START_LABEL

MARTINI works with ``python3`` (version ``3.7`` or higher), and does not support ``python2``.

Stable releases are available via PyPI_ (``pip install astromartini``) and the numbered releases (starting from 2.0.0) on github. The github main branch is actively developed: things will change, bugs will happen. Any feedback is greatly appreciated via github or kyle.a.oman@durham.ac.uk.

.. _PyPI: https://pypi.org/
.. _kyle.a.oman@durham.ac.uk: mailto:kyle.a.oman@durham.ac.uk

The easiest way to install martini is from PyPI by doing ``python3 -m pip install astromartini``. Output to ``.fits`` files is supported by default; if output to ``.hdf5`` format is desired use ``python3 -m pip install astromartini[hdf5_output]`` instead. This will also handle the installation of the required dependencies. Other optional features require additional dependencies hosted on PyPI. In particular, EAGLE, Illustris/TNG and Simba users who wish to use the custom source modules for those simulations in MARTINI can automatically install the optional dependencies with ``python3 -m pip install astromartini[eaglesource]``, or ``[simbasource]`` or ``[tngsource]``.

.. INSTALLATION_NOTES_END_LABEL

Installing from github
----------------------

.. GITHUB_INSTALLATION_NOTES_START_LABEL

Choose a branch_. Unless you want to install the latest (possibly buggy) main branch, installing with PyPI should usually be preferred. From the branch page (e.g. ``https://github.com/kyleaoman/martini/tree/2.0.X``), click the green 'Code' button and follow instructions to obtain the files. Unpack the zip file if necessary. You should then be able to do ``python3 -m pip install martini/[optional]``, where ``optional`` should be replaced by a comma separated list of optional dependencies. If this fails check that ``martini/`` is a path pointing to the directory containing the ``pyproject.toml`` file for MARTINI. The currently available options are:

- ``hdf5_output``: Supports output to hdf5 files via the h5py package. Since h5py is hosted on PyPI, this option may be used when installing via PyPI.
- ``eaglesource``: Dependencies for the |martini.sources.EAGLESource| module, which greatly simplifies reading input from EAGLE simulation snapshots. Installs my Hdecompose_ package, providing implementations of the `Rahmati et al. (2013)`_ method for computing netural hydrogen fractions and the `Blitz & Rosolowsky (2006)`_ method for atomic/molecular fractions. Also installs `my python-only version`_ of John Helly's `read_eagle`_ package for quick extraction of particles in a simulation sub-volume. h5py is also required.
- ``tngsource``: Dependencies for the |martini.sources.TNGSource| module, which greatly simplifies reading input from IllustrisTNG (and original Illustris) snapshots. Installs my Hdecompose_ package, providing implementations of the `Rahmati et al. (2013)`_ method for computing netural hydrogen fractions and the `Blitz & Rosolowsky (2006)`_ method for atomic/molecular fractions.
- ``magneticumsource``: Dependencies for the |martini.sources.MagneticumSource| module, which supports the Magneticum simulations via `my fork`_ of the `g3t`_ package by Antonio Ragagnin.
- ``sosource``: Dependencies for the |martini.sources.SOSource| module, which provides unofficial support for several simulation datasets hosted on specific systems. This is intended mostly for my own use, but APOSTLE, C-EAGLE/Hydrangea and Auriga users may contact_ me for further information.

.. _branch: https://github.com/kyleaoman/martini/branches
.. _Hdecompose: https://github.com/kyleaoman/Hdecompose
.. _`Rahmati et al. (2013)`: https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.2427R/abstract
.. _`Blitz & Rosolowsky (2006)`: https://ui.adsabs.harvard.edu/abs/2006ApJ...650..933B/abstract
.. _`my python-only version`: https://github.com/kyleaoman/pyread_eagle
.. _`read_eagle`: https://github.com/jchelly/read_eagle
.. _`my fork`: https://github.com/kyleaoman/g3t
.. _`g3t`: https://gitlab.lrz.de/di29bop/g3t
.. _contact: mailto:kyle.a.oman@durham.ac.uk

.. GITHUB_INSTALLATION_NOTES_END_LABEL

Getting started
===============

.. QUICKSTART_START_LABEL
   
See the help for |martini.Martini| for an example script to configure MARTINI and create a datacube. This example can be run by doing:

  ``python -c "from martini import demo; demo()"``

MARTINI has (so far) been successfully run on the output of these simulations:

* EAGLE
* APOSTLE
* C-EAGLE/Hydrangea
* Illustris
* IllustrisTNG
* Auriga
* MaGICC (and therefore in principle NIHAO)
* Magneticum
* Simba

I attempt to support publicly available simulations with a customized source module. If your simulation is public and not supported, please `contact me`_. Currently custom source modules exist for:

.. _contact me: mailto:kyle.a.oman@durham.ac.uk

* EAGLE (|martini.sources.EAGLESource|)
* IllustrisTNG (|martini.sources.TNGSource|; also works with Illustris)
* Magneticum (|martini.sources.MagneticumSource|)
* Simba (|martini.sources.SimbaSource|)

Example notebooks_ are available for supported simulations.

.. _notebooks: https://github.com/kyleaoman/martini/tree/main/examples

.. QUICKSTART_END_LABEL

.. |martini.Martini| replace:: martini.Martini_
.. _martini.Martini: https://martini.readthedocs.io/en/latest/modules/martini.martini.html#martini.martini.Martini

.. |martini.sources.EAGLESource| replace:: martini.sources.EAGLESource_
.. _martini.sources.EAGLESource: https://martini.readthedocs.io/en/latest/modules/martini.sources.eagle_source.html#martini.sources.eagle_source.EAGLESource

.. |martini.sources.TNGSource| replace:: martini.sources.TNGSource_
.. _martini.sources.TNGSource: https://martini.readthedocs.io/en/latest/modules/martini.sources.tng_source.html#martini.sources.tng_source.TNGSource

.. |martini.sources.MagneticumSource| replace:: martini.sources.MagneticumSource_
.. _martini.sources.MagneticumSource: https://martini.readthedocs.io/en/latest/modules/martini.sources.magneticum_source.html#martini.sources.magneticum_source.MagneticumSource

.. |martini.sources.SimbaSource| replace:: martini.sources.SimbaSource_
.. _martini.sources.SimbaSource: https://martini.readthedocs.io/en/latest/modules/martini.sources.simba_source.html#martini.sources.simba_source.SimbaSource

.. |martini.sources.SOSource| replace:: martini.sources.SOSource_
.. _martini.sources.SOSource: https://martini.readthedocs.io/en/latest/modules/martini.sources.so_source.html#martini.sources.so_source.SOSource
