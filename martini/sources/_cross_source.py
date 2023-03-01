import numpy as np
import astropy.units as U
from .sph_source import SPHSource


class _CrossSource(SPHSource):
    """
    Creates a source consisting of 4 particles arrayed in an asymmetric cross.

    A simple test source consisting of four particles will be created. Each has
    a mass of 10^4 Msun, a SPH smoothing length of 1 kpc, a temperature of
    10^4 K, and will be placed in the Hubble flow assuming h=.7 and a distance
    of 3 Mpc. Particle coordinates in kpc are
    [[0,  1,  0],
    [0,  0,  2],
    [0, -3,  0],
    [0,  0, -4]]
    and velocities in km/s are
    [[0,  0,  1],
    [0, -1,  0],
    [0,  0, -1],
    [0,  1,  0]]

    Parameters
    ----------
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

    Returns
    -------
    out : CrossSource
        An appropriately initialized CrossSource object.
    """

    def __init__(
        self,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    ):

        xyz_g = np.array([[0, 1, 0], [0, 0, 2], [0, -3, 0], [0, 0, -4]]) * U.kpc

        vxyz_g = (
            np.array([[0, 0, 1], [0, -1, 0], [0, 0, -1], [0, 1, 0]]) * U.km * U.s**-1
        )

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=0.7,
            T_g=np.ones(4) * 1.0e4 * U.K,
            mHI_g=np.ones(4) * 1.0e4 * U.Msun,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=np.ones(4) * U.kpc,
        )
        return
