Data cubes
==========

A data cube contains the flux distribution measured in an observation as a function of position on the sky (RA and Dec) and frequency (or velocity). The main purpose of MARTINI is to take a :doc:`source </sources/index>` as input and determine how its flux is distributed in an output data cube.

Data cubes in MARTINI
---------------------

MARTINI uses the :class:`~martini.datacube.DataCube` class to represent the data cube that will be output as the end product of creating a mock observation. This defines the extent of the region to be observed, both in position (angular apertures around a central RA and Dec) and in frequency or velocity (bandwidth around a central frequency, or velocity offset around a central velocity). The resolution is also defined by the number of pixels and channels that subdivide the region to be observed.

Using MARTINI's DataCube class
------------------------------

When using MARTINI, most of the time the only interaction with the :class:`~martini.datacube.DataCube` class is to initialize it and pass it to the main :class:`~martini.martini.Martini` class. This section therefore focuses on the parameters that can be set at initialization, but at the bottom there is a short mention of writing the state of a data cube to disk (and restoring from this).

Initialization parameters
+++++++++++++++++++++++++

The footprint of the mock observation on the sky is set by the five parameters: ``ra``, ``dec``, ``px_size``, ``n_px_x``, ``n_px_y``. It is recommended to choose even numbers of pixels along each axis for ``n_px_x`` (RA axis) and ``n_px_y`` (Dec axis). The ``ra`` and ``dec`` parameters then control the location of the boundary between the two centremost pixels along each axis, the the boundaries of pixels going out from this centre are each offset in angle by ``px_size`` from their neighbour. For users accustomed to the FITS_ standard, the ``px_size`` corresponds to `CDELT` and ``ra`` and ``dec`` correspond to `CRVAL`, with `CRPIX` set to the boundary between the two central pixels (or the central pixel if an odd number is chosen). The ``ra``, ``dec`` and ``px_size`` parameters should be provided with :mod:`astropy.units` with dimensions of angle.

.. _FITS: https://fits.gsfc.nasa.gov/fits_standard.html

In the spectral direction, the number of channels is set by ``n_channels`` and the location of the boundary between the two central channels (assuming an even number of channels) is set by ``velocity_centre`` (similar to `CRVAL`). The ``velocity_centre`` can be specified either as a velocity (:mod:`astropy.units` with dimensions of speed) or as a frequency (dimensions of inverse time). Similarly, the spacing between channels, set by ``channel_width`` (similar to `CDELT`), can be either a frequency spacing or a velocity spacing. Frequencies and velocities can be mixed freely, so a ``velocity_centre`` in Hz could be given along with a ``channel_width`` in km/s. Regardless of how the input is provided, internally the channels are spaced evenly in velocity. Since the frequency interval corresponding to a velocity interval is frequency-dependent, if a frequency width is specified it is converted to a velocity width assuming the frequency at the central channel (i.e. at the ``velocity_centre``).

If a Stokes' axis is desired, this can be enabled with ``stokes_axis=True``.

.. note::

   The size of the array stored by a :class:`~martini.datacube.DataCube` typically changes during the process of using MARTINI because some padding is applied to ensure accuracy when convolving with a beam, but will return to its original size after convolution or before writing out a mock observation. See the :doc:`core routines </martini/index>` section for an explanation.

A possible initialization of a :class:`~martini.datacube.DataCube` looks like:

.. code-block:: python

    import astropy.units as U
    from martini import DataCube

    datacube = DataCube(
        n_px_x=256,
	n_px_y=256,
	n_channels=64,
	px_size=15 * U.arcsec,
	channel_width=4 * U.km / U.s,
	velocity_centre=1000 * U.km / U.s,
	ra=45 * U.deg,
	dec=-30 * U.deg,
	stokes_axis=False,
    )

It often makes sense to place the source centre (defined by its RA, Dec and systemic velocity) in the centre of the data cube. A convenient way to do this looks like (omitting the particle data in the source initialization):

.. code-block:: python

    import astropy.units as U
    from martini.sources import SPHSource
    from martini import DataCube

    source = SPHSource(
        distance=10 * U.Mpc,
	vpeculiar=-75 * U.km / U.s,
	ra=45 * U.deg,
	dec=-30 * U.deg,
	h=0.7,
	T_g=...,
	mHI_g=...,
	xyz_g=...,
	vxyz_g=...,
	hsm_g=...,
    )
    # the source defines a property called vsys
    # defined as h * 100km/s * distance + vpeculiar
    datacube = DataCube(
        n_px_x=256,
	n_px_y=256,
	n_channels=64,
	px_size=15 * U.arcsec,
	channel_width=4 * U.km / U.s,
	velocity_centre=source.vsys,
	ra=source.ra,
	dec=source.dec,
	stokes_axis=False,
    )


Saving, loading & copying the datacube state
++++++++++++++++++++++++++++++++++++++++++++

Because some operations that modify :class:`~martini.datacube.DataCube` objects are computationally expensive, especially :meth:`~martini.martini.Martini.insert_source_in_cube`, some functionality to load/save/copy the state of a datacube object is provided. For instance, the result of inserting the source in the cube could be cached and the source insertion step skipped if the cache file exists like this:

.. code-block:: python

    import os
    from martini import Martini, DataCube
    from martini.sources import SPHSource
    from martini.beams import GaussianBeam
    from martini.noise import GaussianNoise
    from martini.sph_kernels import CubicSplineKernel
    from martini.spectral_models import GaussianSpectrum

    # initialization parameters omitted for this schematic example:
    source = SPHSource(...)
    datacube = DataCube(...)
    beam = GaussianBeam(...)
    noise = GaussianNoise(...)
    sph_kernel = CubicSplineKernel(...)
    spectral_model = GaussianSpectrum(...)

    m = Martini(
        source=source,
	datacube=datacube,
	beam=beam,
	noise=noise,
	sph_kernel=sph_kernel,
	spectral_model=spectral_model,
    )
    cache_filename = "cache.hdf5"  # note h5py must be installed
    if not os.path.isfile(cache_filename):
        m.insert_source_in_cube()  # expensive step
	m.datacube.save_state(
	    cache_filename,
	    overwrite=False,  # set to True to allow overwriting existing files
	)
    else:
        m.datacube.load_state(cache_filename)  # avoid expensive step
    m.add_noise()
    m.convolve_beam()
    m.write_fits("my_mock.fits")

.. warning::

   The :meth:`~martini.datacube.DataCube.save_state` method is not intended to save the result of a mock observation. Use the :class:`~martini.martini.Martini` class's :meth:`~martini.martini.Martini.write_fits` or :meth:`~martini.martini.Martini.write_hdf5` methods for this purpose.

Another possible workflow is to copy a :class:`~martini.datacube.DataCube` to create two mock observations that share some common initial steps and then differ later:

.. code-block:: python

    import astropy.units as U
    from martini import Martini, DataCube
    from martini.sources import SPHSource
    from martini.beams import GaussianBeam
    from martini.noise import GaussianNoise
    from martini.sph_kernels import CubicSplineKernel
    from martini.spectral_models import GaussianSpectrum

    # initialization parameters omitted for this schematic example:
    source = SPHSource(...)
    datacube1 = DataCube(...)
    datacube2 = datacube1.copy()  # placeholder, we'll replace it below
    beam1 = GaussianBeam(bmaj=30 * U.arcsec, bmin=30 * U.arcsec)
    beam2 = GaussianBeam(bmaj=15 * U.arcsec, bmin=15 * U.arcsec)
    noise = GaussianNoise(...)
    sph_kernel = CubicSplineKernel(...)
    spectral_model = GaussianSpectrum(...)

    m1 = Martini(
        source=source,
	datacube=datacube1,
	beam=beam1,
	noise=noise,
	sph_kernel=sph_kernel,
	spectral_model=spectral_model,
    )
    m2 = Martini(
        source=source,
	datacube=datacube2,
	beam=beam2,
	noise=noise,
	sph_kernel=sph_kernel,
	spectral_model=spectral_model,
    )
    m1.insert_source_in_cube()  # expensive step
    m2.datacube = m1.datacube.copy()  # bypass expensive step
    m1.add_noise()
    m2.add_noise()
    m1.convolve_beam()
    m2.convolve_beam()
    m1.write_fits("my_mock1.fits")
    m2.write_fits("my_mock2.fits")

.. warning::

   The example using :meth:`~martini.datacube.DataCube.copy` has a subtle potential pitfall. Because of the padding applied to the datacube when creating the :class:`~martini.martini.Martini` object in preparation for convolution with the beam (see :doc:`core routines </martini/index>` section), the two datacubes have different dimensions once ``m1`` and ``m2`` are initialized. A given beam requires a *minimum* pad size, so this example has been carefully constructed to copy the datacube associated with ``m1`` (with the larger pad associated with the larger beam) into ``m2`` (that requires a smaller pad because of the smaller beam). Trying to swap which :class:`~martini.datacube.DataCube` is copied results in a pad that is too small when the :meth:`~martini.martini.Martini.convolve_beam` step is reached and raises an error similar to:

   .. code-block::

       ValueError: datacube padding insufficient for beam convolution (perhaps you loaded a datacube state with datacube.load_state that was previously initialized by martini with a smaller beam?)

   It is a known issue that this kind of workflow is quite a fragile construct. Some streamlining and simplification is planned for future code development.
