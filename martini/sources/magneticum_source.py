import numpy as np
import astropy.units as U
from astropy.coordinates import ICRS
from ..sph_kernels import _WendlandC6Kernel, find_fwhm
from .sph_source import SPHSource


class MagneticumSource(SPHSource):
    """
    Class abstracting HI sources designed to work with Magneticum snapshot
    + group fies.

    Provide either:

     - ``haloPosition``, ``haloVelocity`` and ``haloRadius``;
     - or ``groupFile`` and ``haloID`` or ``subhaloID`` (not both).

    Parameters
    ----------
    snapBase : str
        Path to snapshot file, omitting the portion numbering the snapshot
        pieces, e.g. ``/path/snap_136.0`` becomes ``/path/snap_136``.

    haloPosition : ~numpy.typing.ArrayLike
        Array with shape ``(3, )``.
        Location of source centre in simulation units.

    haloVelocity : ~numpy.typing.ArrayLike
        Array with shape ``(3, )``.
        Velocity of halo in the simulation box frame, in simulation units.

    haloRadius : float
        Aperture within which to select particles around the source
        centre, in simulation units.

    groupFile : str
        Path to group file (e.g. ``/path/to/groups_136``).

    haloID : int
        ID of FOF group to use as source.

    subhaloID : int
        ID of subhalo to use as source.

    xH : float
        Primordial hydrogen fraction. (Default: ``0.76``)

    Lbox : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Comoving box side length, without factor h.

    internal_units : dict
        Specify the system of units used in the snapshot file. The dict keys
        should be ``L`` (length), ``M`` (mass), ``V`` (velocity), ``T`` (temperature).
        The values should use :class:`~astropy.units.Quantity`.
        (Default: ``dict(L=U.kpc, M=1E10 * U.Msun, V=U.km/U.s, T=U.K)``)

    rescaleRadius : float
        Factor by which to multiply the haloRadius to define the aperture
        within which particles are selected. Useful in conjunction with
        arguments ``groupFile`` and ``haloID`` or ``subhaloID``: by default the aperture
        will be the halo virial radius, use this argument to adjust as needed.

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

    coordinate_frame : ~astropy.coordinates.builtin_frames.baseradec.BaseRADecFrame, \
    optional
        The coordinate frame assumed in converting particle coordinates to RA and Dec, and
        for transforming coordinates and velocities to the data cube frame. The frame
        needs to have a well-defined velocity as well as spatial origin. Recommended
        frames are :class:`~astropy.coordinates.GCRS`, :class:`~astropy.coordinates.ICRS`,
        :class:`~astropy.coordinates.HCRS`, :class:`~astropy.coordinates.LSRK`,
        :class:`~astropy.coordinates.LSRD` or :class:`~astropy.coordinates.LSR`. The frame
        should be passed initialized, e.g. ``ICRS()`` (not just ``ICRS``).
        (Default: ``astropy.coordinates.ICRS()``)
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
        Lbox=100 * U.Mpc,  # what is it, actually?
        internal_units=dict(L=U.kpc, M=1e10 * U.Msun, V=U.km / U.s, T=U.K),
        distance=3 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0 * U.deg,
        dec=0 * U.deg,
        coordinate_frame=ICRS(),
    ):
        from g3t.stable.g3read import GadgetFile, read_particles_in_box

        # I guess I should allow rescaling of radius to get fore/background

        if (haloID is not None) or (subhaloID is not None) or (groupFile is not None):
            if (
                (haloPosition is not None)
                or (haloVelocity is not None)
                or (haloRadius is not None)
            ):
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
        header = GadgetFile(snapBase + ".0").header

        a = header.time
        h = header.HubbleParam

        l_unit = internal_units["L"] * a / h
        m_unit = internal_units["M"] / h
        v_unit = internal_units["V"] * np.sqrt(a)
        T_unit = internal_units["T"]

        f_gas = read_particles_in_box(
            snapBase,
            haloPosition,
            haloRadius,
            ["POS ", "VEL ", "MASS", "TEMP", "NH  ", "HSML"],
            [0],
        )

        particles["xyz_g"] = f_gas["POS "] * l_unit
        particles["vxyz_g"] = f_gas["VEL "] * v_unit
        particles["hsm_g"] = (
            f_gas["HSML"] * l_unit * find_fwhm(_WendlandC6Kernel().kernel)
        )
        particles["T_g"] = f_gas["TEMP"] * T_unit
        particles["mHI_g"] = f_gas["NH  "] * xH * f_gas["MASS"] * m_unit

        particles["xyz_g"] -= haloPosition * l_unit
        particles["xyz_g"][particles["xyz_g"] > Lbox * a / 2.0] -= Lbox.to(U.kpc) * a
        particles["xyz_g"][particles["xyz_g"] < -Lbox * a / 2.0] += Lbox.to(U.kpc) * a
        particles["vxyz_g"] -= haloVelocity * v_unit

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            coordinate_frame=coordinate_frame,
            **particles,
        )
        return
