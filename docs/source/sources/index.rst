Sources
=======

In the context of mock observations of a simulation, the source is the collection of particles that produce the emission to be "observed".

Sources in MARTINI
------------------

MARTINI has a collection of classes used to store and manipulate the properties of the source particles. All of these inherit from the :class:`martini.sources.SPHSource` class. This base class can be created by providing arrays containing the particle coordinates, velocities, masses, etc. The other source classes are tailored to specific simulations or data formats and will typically expect one or more filenames as input, along with an identifier specifying a group of particles (as defined in a group catalogue, for example). These classes will then take care of reading data from the input files, annotating them with units, doing any required calculations or conversions, and so on. There are source classes for the EAGLE_, IllustrisTNG, Simba, Colibre and Magneticum simulations. The SWIFTGalaxy_ interface to `SWIFT`_-based simulations with a variety of halo catalogue formats is also supported.

.. _EAGLE: https://icc.dur.ac.uk/Eagle
.. _IllustrisTNG: https://www.tng-project.org
.. _Simba: http://simba.roe.ac.uk
.. _SWIFTGalaxy: https://github.com/SWIFTSIM/swiftgalaxy
.. _SWIFT: https://github.com/SWIFTSIM/SWIFT

Units in MARTINI
----------------

MARTINI adopts the astropy_ system of :mod:`astropy.units`. A very brief introduction is given here for users unfamiliar with this module.

.. _astropy: https://www.astropy.org

The most common unit operation in MARTINI is to attach units to scalars or arrays. This is intuitively achieved by multiplying (or dividing) the scalar/array by the unit. Units can also be raised to powers.

.. code-block :: python

    import numpy as np
    import astropy.units as U
    
    mass = 1 * U.Msun
    speeds = np.array([1000, 1001, 1002]) * U.km / U.s
    density = 1 * U.Msun / U.kpc ** 3

These :class:`astropy.units.Quantity` objects can be used in most of the same ways as ordinary scalars or arrays, but will check unit consistency and/or propagate units during calculations. For example, attempting to add two quantities with incompatible dimensions will raise an exception, and dividing a quantity with dimensions of length by a quantity with dimensions of time will return a quantity with dimensions of speed. Conversion to different units with the same dimensions can be achieved with the :meth:`~astropy.units.Quantity.to` method, while the :meth:`~astropy.units.Quantity.to_value` method will return the (array or scalar) value in the specified units, without units attached.

.. code-block :: python

    from astropy.units import UnitConversionError

    speeds.to(U.Mpc / U.Gyr)  # returns a Quantity object
    (mass * speeds).to_value(U.kg * U.m / U.s)  # returns a numpy array of momenta in SI units
    try:
        mass + density  # incompatible units raise exception!
    except UnitConversionError:
        pass

MARTINI functions that expect :class:`~astropy.units.Quantity` inputs accept any units with the correct dimensions, so a mass could be specified in kg, Msun, or other mass units. Any required unit conversions will then happen internally.

The module :mod:`astropy.constants` provides a wide range of pre-defined physical constants compatible with :mod:`astropy.units`.

Using MARTINI's source classes
------------------------------

Quick overview here (including preview), details below

Distance, peculiar velocity, right ascension and declination
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

distance (h), vpeculiar, ra, dec

Rotation and translation
++++++++++++++++++++++++

axis_angle, rotmat, L_coords :meth:`~martini.sources.SPHSource.boost` :meth:`~martini.sources.SPHSource.rotate` :meth:`~martini.sources.SPHSource.translate` :meth:`~martini.sources.SPHSource.save_current_rotation`

Particle arrays
+++++++++++++++

T_g, mHI_g, xyz_g, vxyz_g, hsm_g (FWHM, see smoothing lengths)

Masking
+++++++

:meth:`~martini.sources.SPHSource.apply_mask`

Preview
+++++++

:meth:`~martini.sources.SPHSource.preview`
