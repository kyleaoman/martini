import numpy as np
from .sph_source import SPHSource
from astropy import units as U


class SWIFTGalaxySource(SPHSource):
    """
    Class abstracting HI sources designed to work with the :mod:`swiftsimio` and
    :mod:`swiftgalaxy` modules.

    Parameters
    ----------
    galaxy: ~swiftgalaxy.reader.SWIFTGalaxy
        Instance of a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    distance : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: ``3 * U.Mpc``)

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: ``0 * U.km * U.s**-1``)

    rotation : dict, optional
        Must have a single key, which must be one of ``axis_angle``, ``rotmat`` or
        ``L_coords``. Note that the 'y-z' plane will be the one eventually placed in the
        plane of the "sky". The corresponding value must be:

        - ``axis_angle`` : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element a :class:`~astropy.units.Quantity` with \
        dimensions of angle, indicating the angle to rotate through.
        - ``rotmat`` : A (3, 3) :class:`~numpy.ndarray` specifying a rotation.
        - ``L_coords`` : A 2-tuple containing an inclination and an azimuthal \
        angle (both :class:`~astropy.units.Quantity` instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about \
        'y'). A 3-tuple may be provided instead, in which case the third \
        value specifies the position angle on the sky (second rotation about 'x'). \
        The default position angle is 270 degrees.

        (Default: ``np.eye(3)``)

    ra : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Right ascension for the source centroid. (Default: ``0 * U.deg``)

    dec : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Declination for the source centroid. (Default: ``0 * U.deg``)

    """

    def __init__(
        self,
        galaxy,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    ):
        h = galaxy.metadata.cosmology.h
        particles = dict(
            xyz_g=galaxy.gas.coordinates.to_astropy(),
            vxyz_g=galaxy.gas.velocities.to_astropy(),
            T_g=galaxy.gas.temperatures.to_astropy(),
            # SWIFT guarantees smoothing lengths are kernel FWHM, use directly:
            hsm_g=galaxy.gas.smoothing_lengths.to_astropy(),
            mHI_g=galaxy.gas.atomic_hydrogen_masses.to_astropy(),
        )

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            **particles,
        )
        return
