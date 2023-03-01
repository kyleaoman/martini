import numpy as np
from astropy import units as U
from .swiftgalaxy_source import SWIFTGalaxySource


class ColibreSource(SWIFTGalaxySource):
    """
    Class abstracting HI sources designed to work with Colibre simulations. Uses the
    :mod:`swiftsimio` and :mod:`swiftgalaxy` modules.

    Parameters
    ----------
    galaxy: SWIFTGalaxy
        Instance of a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    distance : Quantity, with dimensions of length, optional
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: 3 Mpc.)

    vpeculiar : Quantity, with dimensions of velocity, optional
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: 0 km/s.)

    rotation : dict, optional
        Must have a single key, which must be one of `axis_angle`, `rotmat` or
        `L_coords`. Note that the 'y-z' plane will be the one eventually placed in the
        plane of the "sky". The corresponding value must be:

        - `axis_angle` : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element a Quantity with \
        dimensions of angle, indicating the angle to rotate through.
        - `rotmat` : A (3, 3) numpy.array specifying a rotation.
        - `L_coords` : A 2-tuple containing an inclination and an azimuthal \
        angle (both Quantity instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about \
        'y'). A 3-tuple may be provided instead, in which case the third \
        value specifies the position angle on the sky (rotation about 'x'). \
        The default position angle is 270 degrees.

        (Default: identity rotation matrix.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension for the source centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination for the source centroid. (Default: 0 deg.)

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
        # No special functionality wanted/needed:
        super().__init__(
            galaxy,
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
        )
        return
