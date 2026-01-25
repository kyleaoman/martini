"""
Provides the :class:`~martini.sources.magneticum_source.MagneticumSource` class.

Facilitates working with Magneticum simulations as input.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
import astropy.units as U
from astropy.coordinates import ICRS
from ..sph_kernels import _WendlandC6Kernel, find_fwhm
from ..L_coords import L_coords
from .sph_source import SPHSource

if TYPE_CHECKING:
    from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame


class MagneticumSource(SPHSource):
    """
    Class abstracting HI sources for use with Magneticum snapshot and group files.

    Provide either:

     - ``haloPosition``, ``haloVelocity`` and ``haloRadius``;
     - or ``groupFile`` and ``haloID`` or ``subhaloID`` (not both).

    Parameters
    ----------
    snapBase : str
        Path to snapshot file, omitting the portion numbering the snapshot
        pieces, e.g. ``/path/snap_136.0`` becomes ``/path/snap_136``.

    haloPosition : ~numpy.ndarray
        Array with shape ``(3, )``.
        Location of source centre in simulation units.

    haloVelocity : ~numpy.ndarray
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

    rescaleRadius : float
        Factor by which to multiply the haloRadius to define the aperture
        within which particles are selected. Useful in conjunction with
        arguments ``groupFile`` and ``haloID`` or ``subhaloID``: by default the aperture
        will be the halo virial radius, use this argument to adjust as needed.

    xH : float
        Primordial hydrogen fraction.

    Lbox : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Comoving box side length, without factor h.

    internal_units : dict
        Specify the system of units used in the snapshot file. The dict keys
        should be ``L`` (length), ``M`` (mass), ``V`` (velocity), ``T`` (temperature).
        The values should use :class:`~astropy.units.Quantity`.

    distance : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity along the direction to the source centre.

    rotation : ~scipy.spatial.transform.Rotation, optional
        A rotation to apply to the source particles, specified using the
        :class:`~scipy.spatial.transform.Rotation` class. That class supports many ways to
        specify a rotation (Euler angle, rotation matrices, quaternions, etc.). Refer to
        the :mod:`scipy` documentation for details. Note that the ``y-z`` plane will be
        the one eventually placed in the plane of the "sky". Cannot be used at the same
        time as ``L_coords``.

    L_coords : ~martini.L_coords.L_coords, optional
        A named tuple specifying 3 angles. Import it as ``from martini import L_coords``.
        The angles are used to orient the galaxy relative to its angular momentum vector,
        "L". The routine will first identify a preferred plane based on the angular
        momenta of the central 1/3 of HI gas. This plane will then be rotated to lie in
        the plane of the "sky" (``y-z`` plane), rotated by an angle ``az_rot`` around the
        angular momentum vector (rotation around ``x``), then inclined by ``incl`` towards
        or away from the line of sight (rotation around ``y``) and finally rotated on the
        sky to set the position angle ``pa`` (second rotation around ``x``). All rotations
        are extrinsic. The position angle refers to the receding side of the galaxy
        measured East of North. The angles should be specified using syntax like:
        ``L_coords=L_coords(incl=0 * U.deg, pa=270 * U.deg, az_rot=0 * U.deg)``. These
        example values are the defaults. Cannot be used at the same time as ``rotation``.

    ra : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Right ascension for the source centroid.

    dec : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Declination for the source centroid.

    coordinate_frame : ~astropy.coordinates.builtin_frames.baseradec.BaseRADecFrame, \
    optional
        The coordinate frame assumed in converting particle coordinates to RA and Dec, and
        for transforming coordinates and velocities to the data cube frame. The frame
        needs to have a well-defined velocity as well as spatial origin. Recommended
        frames are :class:`~astropy.coordinates.GCRS`, :class:`~astropy.coordinates.ICRS`,
        :class:`~astropy.coordinates.HCRS`, :class:`~astropy.coordinates.LSRK`,
        :class:`~astropy.coordinates.LSRD` or :class:`~astropy.coordinates.LSR`. The frame
        should be passed initialized, e.g. ``ICRS()`` (not just ``ICRS``).
    """

    def __init__(
        self,
        *,
        snapBase: str,
        haloPosition: np.ndarray,
        haloVelocity: np.ndarray,
        haloRadius: float,
        groupFile: str,
        haloID: int,
        subhaloID: int,
        rescaleRadius: float = 1.0,
        xH: float = 0.76,  # not in header
        Lbox: U.Quantity[U.Mpc] = 100 * U.Mpc,  # what is it, actually?
        internal_units: dict = {
            "L": U.kpc,
            "M": 1e10 * U.Msun,
            "V": U.km / U.s,
            "T": U.K,
        },
        distance: U.Quantity[U.Mpc],
        vpeculiar: U.Quantity[U.km / U.s] = 0 * U.km / U.s,
        rotation: Rotation | None = Rotation.identity(),
        L_coords: L_coords | None = None,
        ra: U.Quantity[U.deg] = 0 * U.deg,
        dec: U.Quantity[U.deg] = 0 * U.deg,
        coordinate_frame: "BaseRADecFrame" = ICRS(),
    ) -> None:
        from g3read import GadgetFile, read_particles_in_box

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
            L_coords=L_coords,
            ra=ra,
            dec=dec,
            h=h,
            coordinate_frame=coordinate_frame,
            **particles,
        )
        return
