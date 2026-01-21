SPH kernels
===========

.. warning::

   There are significant changes to this module in ``v2.0.3``. Consider upgrading your
   installation, or refer to documentation for the specific version that you are using.
   This page was also added in ``v2.0.3`` and does not exist in earlier versions of the
   documentation. The best references for earlier versions are the notebooks in the
   ``examples/`` directory on github.

In smoothed-particle hydrodynamics (SPH) simulations, gas particles are not point-like but
are instead smoothed around there position according to a kernel function. The
hydrodynamic properties of the gas are then defined at every point in space. For instance,
the gas temperature at a given point is a weighted combination of the temperatures of all
particles whose smoothing kernels overlap the point of interest, with the weights given by
the kernel function evaluated at the distance of each particle from the point of interest.
The classical SPH smoothing kernel is a cubic spline. Some modern simulations use the
cubic spline, but others choose different kernels to optimise for different aspects of
accuracy or performance.

SPH kernels in MARTINI
----------------------

MARTINI provides a module with classes corresponding to the most common SPH kernels used
in simulations. These are:

 - The WendlandC2 kernel (e.g. EAGLE simulations).
 - The WendlandC6 kernel (e.g. Magneticum simulations).
 - The cubic spline kernel (e.g. original Gadget2 code, but not necessarily simulations
   based on Gadget2).
 - The quartic spline kernel (e.g. Colibre simulations).

These can be imported as :class:`martini.sph_kernels.WendlandC2Kernel`,
:class:`martini.sph_kernels.WendlandC6Kernel`,
:class:`martini.sph_kernels.CubicSplineKernel` and
:class:`martini.sph_kernels.QuarticSplineKernel`, respectively. The IllustrisTNG
simulations are conspicuously not mentioned - see below for details.

MARTINI also provides a :class:`martini.sph_kernels.GaussianKernel`. Gaussian kernels are
not commonly used in SPH schemes, but they *are* sometimes convenient approximations when
projecting a spherical kernel onto a 2-dimensional image, which is the calculation needed
in MARTINI.

Finally, there is a :class:`martini.sph_kernels.DiracDeltaKernel`. This can be useful when
the SPH smoothing lengths are much smaller than the desired pixel scale: in this case any
smoothing is ignored and each particle simply contributes flux to the pixel that its
centre is enclosed in.

The line integrals through the kernel functions commonly used in simulations that need to
be evaluated to calculate the flux contributed by a particle to a pixel are in practice
hideously complicated. SPH kernels are generally chosen for very nice mathematical
properties in 3-dimensional space, their 2-dimensional projections are not a
consideration. In MARTINI, rather than use (slow) numerical evaluation of the required
integrals, the approach implemented is to use approximations of the integrals and
guarantee that these will always result in a total flux from a particle within 1% of the
exact calculation. This means that a mock data cube will always contain at least 99% of
the flux (within the field of view and bandpass) that should be emitted by the simulation
particles. In practice it is usually more than this: for most particles, the
approximations used are better than 1% accurate.

Adaptive kernels in MARTINI (advanced usage)
++++++++++++++++++++++++++++++++++++++++++++

.. note::

   New users can skip this subsection - it's enough to know that MARTINI guarantees that
   its kernel integral approximations are accurate within 1%.

The way that MARTINI achieves this depends on the size of the smoothing lengths compared
to the size of the pixels. The smoothing lengths are usually given in physical length
(like kpc), while the pixel size is usually given in angular units (arcsec or similar), so
the comparison also involves the distance from the observer. There are 3 regimes:

 - When the smoothing scale is much larger than a pixel (so a particle's flux will be
   distributed across many pixels) the approximate integrals are very accurate, so these
   are preferentially used.
 - When the smoothing scale is less than half a pixel (so a particle's flux will on
   average land entirely in one pixel), MARTINI assigns all the flux from a particle to
   the pixel that its centre is enclosed in.
 - In the intermediate case MARTINI substitutes the actual SPH kernel with a Gaussian
   kernel truncated at 6 standard deviations. The line integral through this kernel is
   evaluated accurately enough to guarantee the promised accuracy for smoothing length to
   pixel size ratios down to 0.5, below which the flux is instead assigned to a single
   pixel as explained in the previous bullet point. The change in shape should usually be
   barely noticeable since the kernel is so sparsely sampled in this regime.

All of this is implemented in the :class:`martini.sph_kernels._AdaptiveKernel` class, from
which the kernels listed above inherit. For advanced use, this class can be initialised
with a list of kernels in decreasing order of priority. MARTINI will try each one in turn
for each particle until it finds one that will achieve at least 1% flux accuracy. If this
adaptive behaviour is not wanted, the adaptive kernel classes each have a non-adaptive
counterpart prefixed with an underscore, e.g.
:class:`martini.sph_kernels.WendlandC2Kernel` becomes
:class:`martini.sph_kernels._WendlandC2Kernel` (note that
:class:`martini.sph_kernels.DiracDeltaKernel` is not adaptive).

.. note::

   MARTINI's online documentation pages omit classes starting with an underscore - this is
   intentional as most users will not need them. They are fully documented in the source
   code docstrings, accessible for instance by browsing the source code in the online help
   pages or on github, or by using ``help()`` in an interactive python session.

Smoothing lengths in MARTINI
----------------------------

There are many definitions in the literature for the smoothing length, even that of a
single kernel. For instance, sometimes the *diameter* where the kernel's amplitude is 0.5
of its peak value (FWHM) is used, while elsewhere the *radius* where the kernel amplitude
reaches 0 might be used. To avoid confusion, MARTINI requires that smoothing lengths
always be provided as FWHM values (keep in mind that this is a diameter, not a radius!).
In general these are not the smoothing lengths recorded in snapshot files and you need to
convert them yourself. Refer to the documentation of your simulations or simulation code
to find out what the values recorded in snapshots represent.

.. note::

   If you are using one of MARTINI's source classes for a specific simulation, such as
   :class:`~martini.sources.eagle_source.EAGLESource`, then any necessary conversion of
   smoothing lengths is already implemented in that class.

MARTINI with moving-mesh simulations
++++++++++++++++++++++++++++++++++++

Moving-mesh simulations (e.g. run with the AREPO code, such as Illustris, IllustrisTNG,
Auriga) are similar to SPH in some respects, but have no concept of a smoothing length.
For these simulations it is often not unreasonable to derive radii for the Voronoi cells
by taking spheres with a volume equal to the cells and calculating their radii. A
reasonable choice for a smoothing length (FWHM) is 2.5 times these cell radii in
combination with a cubic spline kernel. This is the implementation in the
:class:`~martini.sources.tng_source.TNGSource` class.

Using MARTINI's SPH kernel classes
----------------------------------

Simply choose the class corresponding to your preferred SPH kernel (e.g. the one used in
the simulation your are 'observing') and initialise it, then pass it to the main
:class:`~martini.martini.Martini` class, for example:

.. code-block:: python

    from martini.sph_kernels import WendlandC2Kernel
    sph_kernel = WendlandC2Kernel()
    M = Martini(sph_kernel=sph_kernel, ...)

Although generally not needed for routine use of MARTINI, there are functions that provide
the kernel function directly (see the documentation of these functions for the definition
of the function evaluated), for instance:

.. code-block:: python

    sph_kernel.kernel(np.linspace(0, 1, 200))

returns a finely-sampled smoothing kernel. Note that this function expects a dimensionless
array (radius normalized by the radius of compact support, that is the radius where the
kernel function reaches 0; in other words the kernel function is non-zero between 0 and 1)
as input.

There are also functions that evaluate the kernel at a given radius for a given smoothing
length (FWHM). These are available as:

.. code-block:: python

    import astropy.units as U
    sph_kernel.eval_kernel(1 * U.kpc, 3 * U.kpc)  # (radius, smoothing length)


This function will accept either scalars or arrays in any combination, and the two
arguments can have any units (or no units), provided that they have the same dimensions.
