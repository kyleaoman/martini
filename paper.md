---
title: "MARTINI: Mock Array Radio Telescope Interferometry of the Neutral ISM"
tags:
  - Python
  - astronomy
  - simulations
authors: 
  - name: Kyle A. Oman
    orcid: 0000-0001-9857-7788
    affiliation: "1, 2"
    corresponding: true
affiliations:
  - name: Institute for Computational Cosmology, Physics Department, Durham University
    index: 1
  - name: Centre for Extragalactic Astronomy, Physics Department, Durham University
    index: 2
date: 13 March 2024
codeRepository: https://github.com/kyleaoman/martini
license: LGPLv3
bibliography: bibliography.bib
---

# Summary

MARTINI is a modular Python package that takes smoothed-particle hydrodynamics (SPH) simulations of galaxies as input and creates synthetic spatially- and/or spectrally-resolved observations of the 21-cm radio emission line of atomic hydrogen (data cubes). The various aspects of the mock-observing process are divided logically into sub-modules handling the data cube, source galaxy, telescope beam pattern, noise, spectral model and SPH kernel. MARTINI is object-oriented: each sub-module provides a class (or classes) which can be configured as desired. For most sub-modules, base classes are provided to allow for straightforward customization. Instances of each sub-module class are given as parameters to an instance of a main "Martini" class; a mock observation is then constructed by calling a handful of functions to execute the desired steps in the mock-observing process.

# Background

The primordial Universe contained predominantly hydrogen, some helium, and trace amount of heavier elements. Hydrogen remains the most abundant element, occuring ubiquitously in stars and interstellar/intergalactic gas. Atomic hydrogen (i.e. with an electron bound to the nucleus) is much more abundant in galaxies where the gravitational field of dark matter allows gas to collect and cool. Hydrogen atoms in their lowest energy state exhibit a transition between the state where the electron spin is aligned and that where it is anti-aligned with the nuclear spin. The very close ("hyperfine") spacing in energy between these two states means that the photon emitted/absorbed by a transition between the states has a correspondingly low energy, or equivalently, long wavelength of about 21 cm. The decay from the excited (spins aligned) to the ground (spins anti-aligned) state is very slow, with a mean lifetime of about 11 million years, but the ubiquity of hydrogen in the Universe makes the 21 cm, or "HI", line readily observable in emission (and absorption, but this is not the focus of MARTINI) using radio telescopes. The fact that the 21 cm radiation originates from a spectral line means that the precise frequency in the rest frame is known from laboratory measurements, so the observed frequency can be used to measure a Doppler shift and therefore a measure of the kinematics of the emitting gas. These properties lead to a wealth of scientific applications of HI line emission observations, including in extragalactic astrophysics where MARTINI focuses. An more detailed overview of 21 cm radio astronomy is available in [cite].

Another powerful tool for the study of galaxies, this time coming from the theory perspective, are simulations of galaxy formation. These come in several flavours, one of which is "hydrodynamical" simulations where gas is discretized as either a set of particles or a set of cells in a mesh. Where particles are used, the most common formulation of the relevant equations is "smoothed particle hydrodynamics" (SPH) [cite]. Observations of the cosmic microwave background [cite] enable us to formulate initial conditions for a simulation of a representative region of the early Universe, before the first galaxies formed. By iteratively solving Poisson's equation for gravity, the hydrodynamic equations, and additional equations and approximations needed to capture relevant astrophysical processes, the simulated universe can be evolved forward in time and make predictions for the structure and kinematics of galaxies, including their atomic hydrogen gas.

# Statement of need

A SPH simulation of galaxies and radio astronomical observations of galaxies produce qualitatively different data sets. The output of a simulation consists of tabulated lists of particles positions, velocities, masses, chemical abundances, etc. A common high-level data product from a radio observatory is a cubic data structure where the axes of the cube correspond to right ascension (longitude on the sky), declination (latitude) and frequency. Each cell in such a cube records the intensity of radio emission with corresponding right ascension, declination and frequency. Since two of the cube axes are spatial coordinates and the third is a spectral coordinate, one can equivalently think of these data cubes as images where each pixel contains a discretized spectrum at that position instead of a single-valued intensity. Testing the theoretical predictions of simulations against these kinds of observations is therefore challenging because the data are organized into fundamentally different structures. MARTINI provides a tool to mimick the process of observing a simulated galaxy with a radio telescope, producing as output the same kinds of "data cubes" and thus enabling much more robust tests of the theoretical predictions made by galaxy formation simulations.

MARTINI is hosted on [GitHub](https://github.com/kyleaoman/martini) and
has documentation available through
[ReadTheDocs](https://martini.readthedocs.io).

# Acknowledgements

KAO acknowledges support by the Royal Society trhough Dorothy Hodgkin Fellowship DHF/R1/231105, by STFC through grant ST/T000244/1, by the European Research Council (ERC) through an Advanced Investigator Grant to C. S. Frenk, DMIDAS (GA 786910), and by the Netherlands Foundation for Scientific Research (NWO) through VICI grant 016.130.338 to M. Verheijen. This work has made use of NASA's Astrophysics Data System.

# References
