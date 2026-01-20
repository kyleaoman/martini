Noise
=====

Real telescopes inevitably have some level of noise that is recorded in their
measurements. This could arise from actual noise signals received by the detector
(radio-frequency interference, for instance), or as instrumental noise imposed by the
limitations of electronics.

Noise in MARTINI
----------------

MARTINI includes some functionality to simulate a simple form of noise: Gaussian white
noise. The interface for this is provided by the :class:`~martini.noise.GaussianNoise`
class.

Using MARTINI's noise classes
-----------------------------

The :class:`~martini.noise.GaussianNoise` class is straightforward to use. It has an
``rms`` argument that expects the desired noise level in the final mock observation (after
convolution with the beam). There is also a ``seed`` argument for the random number
generator seed that defaults to ``0``, so random results will be consistent and
predictable (at least on a given system). Initializing the
:class:`~martini.noise.GaussianNoise` class looks like:

.. code-block:: python

    import astropy.units as U
    from martini.noise import GaussianNoise

    GaussianNoise(rms=1.0 * U.Jy / U.beam, seed=0)

The root mean square (RMS) noise level is approximate because the noise needs to be
generated before convolution with the beam, and the smoothing effect of the convolution
reduces the RMS. Convolution with a Gaussian beam with axis lengths ``bmaj`` and ``bmin``
reduces the RMS by a factor of approximately
:math:`(2\pi\sqrt{\sigma_\mathrm{maj}\sigma_\mathrm{min}})^{-1}` where :math:`\sigma` are
the widths of the Gaussians (recall that MARTINI defines ``bmaj`` and ``bmin`` as FWHM
values) *in pixels*. Empirical testing reveals that this approximation is too low by about
10%, so the factor is adjusted slightly.

MARTINI then creates a Gaussian white noise array with RMS that is higher than the final
desired level by the factor above such that when the convolution with the beam is done the
result has (approximately) the target noise level.

Other noise models (advanced usage)
+++++++++++++++++++++++++++++++++++

Other, perhaps more intricate models for noise can be implemented into MARTINI. A class
created to do this should inherit from the :class:`martini.noise._BaseNoise` class. Note
that classes beginning in underscores are intentionally not documented in the online API
documentation as most users are unlikely to need them. Refer to the docstrings in the
source code for technical documentation. The new class needs to implement a
:meth:`~martini.noise._BaseNoise.generate` method that initializes an array of the same
shape as a given datacube containing the noise. The
:meth:`~martini.noise._BaseNoise.generate` method receives the ``datacube`` and ``beam``
members of a :class:`~martini.martini.Martini` object, making their properties available
within the method. Other parameters are expected to be passed to the noise class on
initialization, analogous to the ``rms`` argument of the
:meth:`~martini.noise.GaussianNoise.__init__` method of
:class:`~martini.noise.GaussianNoise`.
