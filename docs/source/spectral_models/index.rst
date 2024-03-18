Spectral Models
===============

The spectral model module defines the shape of the line emission that arises from the HI gas particles of the input simulation.

Spectral models in MARTINI
--------------------------

MARTINI provides a module with classes representing two line profiles:

 - The :class:`~martini.spectral_models.GaussianSpectrum` (in most cases this is the best option)
 - The :class:`~martini.spectral_models.DiracDeltaSpectrum`

These can be imported as :class:`martini.spectral_models.GaussianSpectrum` and :class:`martini.spectral_models.DiracDeltaSpectrum`, respectively. In most cases the :class:`~martini.spectral_models.GaussianSpectrum` is the best option.

The :class:`~martini.spectral_models.GaussianSpectrum` models the line emission as a Gaussian function. The width (sigma, not e.g. FWHM) of the Gaussian can be set to a constant, for example:

.. code-block:: python

   import astropy.units as U
   GaussianSpectrum(sigma=10 * U.km / U.s)

The default is a constant 7 km/s line width, corresponding to a FWHM of about 16 km/s. Alternatively, the width of the Gaussian emitted by each particle can be tied to the gas temperature as :math:`'sqrt{k_B T_g / m_p}`, which is the 1D velocity dispersion of an ideal gas of protons, where :math:`k_B` is Boltzmann's constant, :math:`T_g` is the gas particle temperature and :math:`m_p` is the proton mass.

The :class:`~martini.spectral_models.DiracDeltaSpectrum` instead assigns all of the emission of a particle to the channel containing the particle's line-of-sight velocity. This is mostly intended as a debugging tool but can sometimes be useful to speed up a calculation in cases where a delta function is a good approximation (for instance if the channels are very wide).

Using MARTINI's spectral model classes
--------------------------------------

Simply choose the class corresponding to your preferred line shape and initialise it, then pass it to the main :class:`~martini.martini.Martini` class, for example:

.. code-block:: python

    from martini.sph_kernels import GaussianSpectrum
    spectral_model = GaussianSpectrum(sigma="thermal")
    M = Martini(spectral_model=spectral_model, ...)

Parallelization
+++++++++++++++

The spectra of all input particles are computed when a :class:`~martini.martini.Martini` object is initialized, before the particles are inserted into the data cube (which is triggered by calling :meth:`~martini.martini.Martini.insert_source_in_cube`). After the main source insertion loop, calculation of the spectra is often the most computationally expensive step, although typically this is only noticeable for very large numbers of particles. If initializing the :class:`~martini.martini.Martini` class is found to be slow, it may be significantly faster to calculate the spectra in parallel. For lower particle counts, however, running in parallel is usually significantly slower because of the overheads involved in the parallel implementation chosen for the calculation. Parallel execution is specified when the spectral model module is initialized, for example:

.. code-block:: python

    spectral_model = GaussianSpectrum(ncpu=4)

There is also a parallel mode for :meth:`~martini.martini.Martini.insert_source_in_cube`. Optimization of the two parts of the calculation should be considered separately: while it is almost always faster to run the source insertion in parallel, the dependence of the calculation of the spectra on the number of particles means that parallel execution should not be turned on blindly for this step. Some testing by users for their specific use cases is recommended.
    
Memory usage and data type of spectra (advanced usage)
++++++++++++++++++++++++++++++++++++++++++++++++++++++

The spectra are stored in a 2-dimensional array whose size is the product of the number of particles in the source and the number of channels in the data cube. For large numbers of particles (or channels) this can consume a lot of memory. The data type of this array can be controlled to help mitigate this, if less precision is acceptable. By default the spectra are stored with :class:`~numpy.float64`. This can be changed, for example:

.. code-block:: python

    spectral_model = GaussianSpectrum(spec_dtype=np.float32)

This array is often the most memory-intensive component of MARTINI (although the data cube array can also dominate if there are less particles than there are pixels), and the memory cost can remain high even with low precision. There are plans to offer more options to manage memory usage in future version of MARTINI; requests can be submitted in the existing github issue, or new issues created, if this is hindering use of the code.
