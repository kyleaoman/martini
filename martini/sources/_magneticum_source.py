import numpy as np
import astropy.units as U
from ._sph_source import SPHSource


class MagneticumSource(SPHSource):
    """
    Class abstracting HI sources designed to work with Magneticum snapshot
    + group fies.

    Parameters
    ----------
    snapBase : str
        Path to snapshot file, omitting the portion numbering the snapshot
        pieces, e.g. /path/snap_136.0 becomes /path/snap_136

    haloPosition : array_like, with shape (3, )
        Location of source centre in simulation units. Provide either
        arguments haloPosition, haloVelocity and haloRadius, or arguments
        groupFile, (haloID | subhaloID), not both.

    haloVelocity : array_like, with shape (3, )
        Velocity of halo in the simulation box frame, in simulation units.
        Provide either arguments haloPosition, haloVelocity and
        haloRadius, or arguments groupFile, (haloID | subhaloID), not
        both.

    haloRadius : float
        Aperture within which to select particles around the source
        centre, in simulation units. Provide either arguments
        haloPosition, haloVelocity and haloRadius, or arguments groupFile,
        (haloID | subhaloID), not both.

    groupFile : str
        Path to group file (e.g. /path/to/groups_136). Provide either
        arguments haloPosition, haloVelocity and haloRadius, or arguments
        groupFile, (haloID | subhaloID), not both.

    haloID : int
        ID of FOF group to use as source. Provice either arguments
        haloPosition, haloVelocity and haloRadius, or arguments groupFile,
        (haloID | subhaloID), not both.

    subhaloID : int
        ID of subhalo to use as source. Provide either arguments haloPostion,
        haloVelocity and haloRadius, or arguments groupFile, (haloID |
        subhaloID), not both.

    xH : float
        Primordial hydrogen fraction (default: 0.76).

    Lbox : astropy.units.Quantity, with dimensions of length
        Comoving box side length, without factor h.

    internal_units : dict
        Specify the system of units used in the snapshot file. The dict keys
        should be 'L' (length), 'M' (mass), 'V' (velocity), 'T' (temperature).
        The values should use astropy.units.Quantity. (Default:
        dict(L=U.kpc, M=1E10 * U.Msun, V=U.km/U.s, T=U.K).)

    rescaleRadius : float
        Factor by which to multiply the haloRadius to define the aperture
        within which particles are selected. Useful in conjunction with
        arguments groupFile, (haloID | subhaloID): by default the aperture
        will be the halo virial radius, use this argument to adjust as needed.

    distance : Quantity, with dimensions of length, optional
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: 3 Mpc.)

    vpeculiar : Quantity, with dimensions of velocity, optional
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: 0 km/s.)

    rotation : dict, optional
        Keys may be any combination of `axis_angle`, `rotmat` and/or
        `L_coords`. These will be applied in this order. Note that the 'y-z'
        plane will be the one eventually placed in the plane of the "sky". The
        corresponding values:

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
        momentum pole (rotation about 'x'), and inclined (rotation about 'y').

        (Default: rotmat with the identity rotation.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension for the source centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination for the source centroid. (Default: 0 deg.)
    """
    def __init__(
            self,
            snapBase=None,
            haloPosition=None,
            haloVelocity=None,
            haloRadius=None,
            groupFile=None,
            haloID=None,
            subhaloID=None,
            rescaleRadius=1.0,
            xH=0.76,  # not in header
            Lbox=100*U.Mpc,  # what is it, actually?
            internal_units=dict(
                L=U.kpc,
                M=1E10 * U.Msun,
                V=U.km/U.s,
                T=U.K
            ),
            distance=3*U.Mpc,
            vpeculiar=0*U.km/U.s,
            rotation={'L_coords': (60.*U.deg, 0.*U.deg)},
            ra=0*U.deg,
            dec=0*U.deg
    ):

        from g3t.stable.g3read import GadgetFile, read_particles_in_box

        # I guess I should allow rescaling of radius to get fore/background

        if (haloID is not None) or (subhaloID is not None) \
           or (groupFile is not None):
            if (haloPosition is not None) or (haloVelocity is not None) \
               or (haloRadius is not None):
                raise
        else:
            if (haloID is not None) and (subhaloID is not None):
                raise

        if (haloID is not None) or (subhaloID is not None):
            f = GadgetFile(groupFile)
            data_sub = f.read_new(blocks=["SPOS", "SVEL", "GRNR"], ptypes=[1])
            data_fof = f.read_new(blocks=["RVIR", "FSUB"], ptypes=[0])
            xyz = data_sub["SPOS"]
            vxyz = data_sub["SVEL"]
            fsub = data_fof["FSUB"]
            rvir = data_fof["RVIR"]
            grnr = data_sub["GRNR"]
            if subhaloID is None:
                subhaloID = fsub[haloID]
            if haloID is None:
                haloID = grnr[subhaloID]
            haloPosition = xyz[fsub[haloID]]
            haloVelocity = vxyz[fsub[haloID]]
            haloRadius = rvir[haloID]

        haloRadius *= rescaleRadius

        particles = {}

        # Here all is still in code units
        header = GadgetFile(snapBase + '.0').header

        a = header.time
        h = header.HubbleParam

        l_unit = internal_units['L'] * a / h
        m_unit = internal_units['M'] / h
        v_unit = internal_units['V'] * np.sqrt(a)
        T_unit = internal_units['T']

        f_gas = read_particles_in_box(
            snapBase,
            haloPosition,
            haloRadius,
            ['POS ', 'VEL ', 'MASS', 'TEMP', 'NH  ', 'HSML'],
            [0]
        )

        particles['xyz_g'] = f_gas['POS '] * l_unit
        particles['vxyz_g'] = f_gas['VEL '] * v_unit
        particles['hsm_g'] = f_gas['HSML'] * l_unit
        particles['T_g'] = f_gas['TEMP'] * T_unit
        particles['mHI_g'] = f_gas['NH  '] * xH * f_gas['MASS'] * m_unit

        particles['xyz_g'] -= haloPosition * l_unit
        particles['xyz_g'][particles['xyz_g'] > Lbox * a / 2.] \
            -= Lbox.to(U.kpc) * a
        particles['xyz_g'][particles['xyz_g'] < -Lbox * a / 2.] \
            += Lbox.to(U.kpc) * a
        particles['vxyz_g'] -= haloVelocity * v_unit

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            **particles
        )
        return
