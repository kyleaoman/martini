Global profile mode
===================

Sometimes all you want is a spatially-integrated spectrum of a source. The :class:`~martini.martini.GlobalProfile` class offers a simplified setup of MARTINI 

Assumptions and limitations
---------------------------

In order to offer a simpler and faster way to produce a spectrum, the :class:`~martini.martini.GlobalProfile` class makes some assumptions:

 - No spatial aperture is assumed. Every particle in the source contributes to the spectrum (unless it falls outside of the spectral bandwidth).
 - The positions of particles are still used to calculate the line-of-sight vector and the velocity along this direction.

There is therefore no need or way to specify a beam or SPH kernel as with the main :class:`~martini.martini.Martini` class. It is also not possible to use MARTINI's noise modules with this class. If these restrictions are found to be too limiting, the best course of action is to produce a spatially-resolved mock observation and derive the spectrum from those data as would be done with "real" observations. For example, if the spectrum within a spatial mask defined by a signal-to-noise or other cut is desired, or if spatially-dependent effects like primary beam attenuation are relevant, then the :class:`~martini.martini.GlobalProfile` class should not be used. The :class:`~martini.martini.GlobalProfile` class is mainly intended to efficiently provide a "quick look" at the spectrum, or a reference "ideal" spectrum.

Usage
-----

The :doc:`source </sources/index>` and :doc:`spectral model </spectral_models/index>` modules should be set up as for a full MARTINI mock observation. The :doc:`beam </beams/index>`, :doc:`noise </noise/index>` and :doc:`sph_kernel </sph_kernels/index>` modules are not relevant. The :doc:`datacube </datacube/index>` module is not used, but a subset of its configuration options are instead given directly to the :class:`~martini.martini.GlobalProfile`. Schematically, an example initialisation looks like:

.. code-block:: python

    from martini import GlobalProfile
    from martini.sources import SPHSource
    from martini.spectral_models import GaussianSpectrum
		
    source = SPHSource(...)
    spectral_model = GaussianSpectrum(...)

    gp = GlobalProfile(
        source=source,
	spectral_model=spectral_model,
	n_channels=64,
	channel_width=10 * U.km * U.s**-1,
	velocity_centre=source.vsys,
	channels="velocity",
    )

The arguments to the other modules are omitted here (replaced with ``...``), check the documentation pages of each module for details. Here the spectrum will be centred on the source systemic velocity, but an explicit frequency or Doppler velocity value could be given instead. The ``channels`` argument determines whether the resulting spectrum will have channel edges in velocity or frequency units. The units (frequency or velocity) of the ``channel_width`` and ``velocity_centre``, and whether ``channels`` is set to ``"frequency"`` or ``"velocity"``, can be mixed in any combination.

Inserting the source
--------------------

Bla.

Parallelization
+++++++++++++++

The core loop in the source insertion function is a loop over pixels. Since parallelization is implemented for this loop, and for a :class:`~martini.martini.GlobalProfile` there is a single pixel, parallelization is not available in this mode.
