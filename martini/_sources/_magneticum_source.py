import numpy as np
import astropy.units as U
from ._sph_source import SPHSource
from g3t.stable.g3read import GadgetFile, read_particles_in_box


class MagneticumSource(SPHSource):
    """
    Class abstracting HI sources designed to work with Magneticum snapshot 
    + group fies.

    Parameters
    ----------
    snapBase : string
        x

    haloPosition : numpy.ndarray with shape (3, )
        x

    haloVelocity : numpy.ndarray with shape (3, )
        x

    haloRadius : float
        x

    groupFile : string
        x

    haloID : int
        x

    subhaloID : int
        x

    xH : float
        x

    Lbox : astropy.units.Quantity, with dimensions of length
        x

    internal_units : dict
        x

    distance : astropy.units.Quantity, with dimensions of length
        Source distance, also used to set the velocity offset via Hubble's law.

    vpeculiar : astropy.units.Quantity, with dimensions of velocity
        Source peculiar velocity, added to the velocity from Hubble's law.

    rotation : dict
        Keys may be any combination of 'axis_angle', 'rotmat' and/or
        'L_coords'. These will be applied in this order. Note that the 'y-z'
        pane will be the one eventually placed in the plane of the "sky". The
        corresponding values:
        - 'axis_angle' : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element an astropy.units.Quantity with \
        dimensions of angle, indicating the angle to rotate through.
        - 'rotmat' : A (3, 3) numpy.array specifying a rotation.
        - 'L_coords' : A 2-tuple containing an inclination and an azimuthal \
        angle (both astropy.units.Quantity instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane wil then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about 'y').

    ra : astropy.units.Quantity, with dimensions of angle
        Right ascension for the source centroid.

    dec : astropy.units.Quantity, with dimensions of angle 
        Declination for the source centroid.

    Returns
    -------
    out : MagneticumSource
        An appropriately initialized MagneticumSource object.
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
            f = g3read.GadgetFile(groupFile)
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
            haloVelocity = vx[fsub[haloID]]
            haloRadius = rvir[haloID]

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
        # full kernel width - CHECK IF NEED .5?
        particles['hsm_g'] = f_gas['HSML'] * l_unit
        particles['T_g'] = f_gas['TEMP'] * T_unit
        particles['mHI_g'] = f_gas['NH  '] * xH * f_gas['MASS'] * m_unit

        particles['xyz_g'] -= haloPosition * l_unit
        particles['xyz_g'][particles['xyz_g'] > Lbox / 2.] -= Lbox.to(U.kpc)
        particles['xyz_g'][particles['xyz_g'] < -Lbox / 2.] += Lbox.to(U.kpc)
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
