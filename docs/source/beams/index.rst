Beams
=====

The primary beam is the analogue of the point spread function (PSF) and plays a central role in setting the spatial resolution of a mock observation.

Beams in MARTINI
----------------

MARTINI provides the :class:`~martini.beams.GaussianBeam` class as a possible approximation to the primary beam of any telescope. The major & minor axis lengths and position angle of the beam can be set.

Using MARTINI's beam classes
----------------------------

The :class:`~martini.beams.GaussianBeam` class accepts the full width at half-maximum (FWHM) angular size of the beam via the ``bmaj`` and ``bmin`` keyword arguments. These should be specified with :mod:`astropy.units`. Unequal ``bmaj`` and ``bmin`` results in an elliptical Gaussian beam. Make sure to always specify both the major and minor axis lengths (otherwise a default value may be used). The position angle of the ellipse (East of North) can be set with the ``bpa`` keyword argument. An example initialization looks like:

.. code-block:: python

    from martini.beams import GaussianBeam

    beam = GaussianBeam(
        bmaj=1 * U.arcmin,
	bmin=0.5 * U.arcmin,
	bpa=45 * U.deg,
    )

.. note::

   The angular sizes expect full-width at half-maximum (FWHM) values.
    
There is one further keyword argument ``truncate``. At angular offsets more than this number of FWHM the beam image amplitude is set to zero. The default value of ``truncate=4.0`` should be a reasonable choice for most use cases.
    
Custom beam images (advanced usage)
+++++++++++++++++++++++++++++++++++

For users wanting to use a beam image more specific than a generic Gaussian beam, a base class :class:`martini.beams._BaseBeam` is available to inherit from. Classes beginning with an underscore are deliberately not included in the online documentation as they are either for internal functionality or advanced use cases - refer to the docstrings in the source code for technical documentation. The main requirement is to provide a function via the :meth:`martini.beams._BaseBeam.f_kernel` abstract method (i.e. your class inheriting from :class:`~martini.beams._BaseBeam` should implement this) that returns the beam amplitude as a function of the angular offsets (RA and Dec) from the beam centre. This could be achieved, for example, by reading a beam image from a file and interpolating to arbitrary offset. A few other abstract methods need to be implemented by beam classes, refer to the docstrings in the source code for further information. The :class:`~martini.beams.GaussianBeam` class implements all of these methods and can be used as a loose example of what each needs to do.
