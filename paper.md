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

MARTINI is a modular Python package for the creation of synthetic spatially-resolved observations of the 21-cm emission line of atomic hydrogen (data cubes), using smoothed-particle hydrodynamics simulations of galaxies as input. The various aspects of the mock-observing process are divided logically into sub-modules handling the data cube, source galaxy, telescope beam pattern, noise, spectral model and SPH kernel. MARTINI is object-oriented: each sub-module provides a class (or classes) which can be configured as desired. For most sub-modules, base classes are provided to allow for straightforward customization. Instances of each sub-module class are given as parameters to an instance of a main "Martini" class; a mock observation is then constructed by calling a handful of functions to execute the desired steps in the mock-observing process.

# Background


# Why MARTINI?


`MARTINI` is hosted on [GitHub](https://github.com/kyleaoman/martini) and
has documentation available through
[ReadTheDocs](https://martini.readthedocs.io).

# Acknowledgements

KAO acknowledges support by the Royal Society trhough Dorothy Hodgkin Fellowship DHF/R1/231105, by STFC through grant ST/T000244/1, by the European Research Council (ERC) through an Advanced Investigator Grant to C. S. Frenk, DMIDAS (GA 786910), and by the Netherlands Foundation for Scientific Research (NWO) through VICI grant 016.130.338 to M. Verheijen. This work has made use of NASA's Astrophysics Data System.

# References
