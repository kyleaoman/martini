"""
Provides the :class:`~martini.sources.simba_source.SimbaSource` class.

Facilitates working with Simba simulations as input.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
from .sph_source import SPHSource
from ..sph_kernels import _CubicSplineKernel, find_fwhm
from ..L_coords import L_coords
from os.path import join
from astropy import units as U, constants as C
from astropy.coordinates import ICRS

if TYPE_CHECKING:
    from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame


class SimbaSource(SPHSource):
    """
    Class abstracting HI sources designed to work with SIMBA snapshot and group data.

    For file access, enquire with R. Davé (rad@roe.ac.uk).

    Parameters
    ----------
    snapPath : str
        Directory containing snapshot files.

    snapName : str
        Filename of snapshot file.

    groupPath : str
        Directory containing group catalogue files.

    groupName : str
        Group catalogue filename.

    groupID : int
        Identifier in the GroupID column of group catalogue.

    aperture : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Radial extent of a region to load around the object of interest,
        in physical (not comoving, no little h) units. Using larger values
        will include more foreground/background, which may be desirable, but
        will also slow down execution and impair the automatic routine used
        to find a disc plane.

    distance : ~astropy.units.Quantity, optional
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
        snapPath: str,
        snapName: str,
        groupPath: str,
        groupName: str,
        groupID: int,
        *,
        aperture: U.Quantity[U.kpc] = 50.0 * U.kpc,
        distance: U.Quantity[U.Mpc] = 3.0 * U.Mpc,
        vpeculiar: U.Quantity[U.km / U.s] = 0 * U.km / U.s,
        rotation: Rotation | None = None,
        L_coords: L_coords | None = None,
        ra: U.Quantity[U.deg] = 0.0 * U.deg,
        dec: U.Quantity[U.deg] = 0.0 * U.deg,
        coordinate_frame: "BaseRADecFrame" = ICRS(),
    ) -> None:
        if snapPath is None:
            raise ValueError("Provide snapPath argument to SimbaSource.")
        if snapName is None:
            raise ValueError("Provide snapName argument to SimbaSource.")
        if groupPath is None:
            raise ValueError("Provide groupPath argument to SimbaSource.")
        if groupName is None:
            raise ValueError("Provide groupName argument to SimbaSource.")
        if groupID is None:
            raise ValueError("Provide groupID argument to SimbaSource.")

        # optional dependencies for this source class
        import h5py

        snapFile = join(snapPath, snapName)
        groupFile = join(groupPath, groupName)

        gamma = 5 / 3

        with h5py.File(snapFile, "r") as f:
            a = f["Header"].attrs["Time"]
            h = f["Header"].attrs["HubbleParam"]
            lbox = f["Header"].attrs["BoxSize"] / h * U.kpc
            gas = f["PartType0"]
            fZ = gas["Metallicity"][()][:, 0]
            fHe = gas["Metallicity"][()][:, 1]
            fH = 1 - fHe - fZ
            xe = gas["ElectronAbundance"][()]
            particles = {
                "xyz_g": gas["Coordinates"][()] * a / h * U.kpc,
                "vxyz_g": gas["Velocities"][()] * np.sqrt(a) * U.km / U.s,
                "T_g": (
                    (4 / (1 + 3 * fH + 4 * fH * xe))
                    * C.m_p
                    * (gamma - 1)
                    * gas["InternalEnergy"][()]
                    * (U.km / U.s) ** 2
                    / C.k_B
                ).to(U.K),
                "hsm_g": gas["SmoothingLength"][()]
                * a
                / h
                * U.kpc
                * find_fwhm(_CubicSplineKernel().kernel),
                "mHI_g": gas["Masses"][()]
                * fH
                * gas["GrackleHI"][()]
                * 1e10
                / h
                * U.Msun,
            }
            del fH, fHe, xe

        with h5py.File(groupFile, "r") as f:
            groupIDs = f["galaxy_data/GroupID"][()]
            gmask = groupID == groupIDs
            # no h^-1 on minpotpos, not sure about comoving yet
            cop = f["galaxy_data/minpotpos"][()][gmask][0] * a * U.kpc
            vcent = f["galaxy_data/vel"][()][gmask][0] * np.sqrt(a) * U.km / U.s

        particles["xyz_g"] -= cop
        particles["xyz_g"][particles["xyz_g"] > lbox / 2.0] -= lbox
        particles["xyz_g"][particles["xyz_g"] < -lbox / 2.0] += lbox
        particles["vxyz_g"] -= vcent

        mask = np.zeros(particles["xyz_g"].shape[0], dtype=bool)
        outer_cube = (np.abs(particles["xyz_g"]) < aperture).all(axis=1)
        inner_cube = np.zeros(particles["xyz_g"].shape[0], dtype=bool)
        inner_cube[outer_cube] = (
            np.abs(particles["xyz_g"][outer_cube]) < aperture / np.sqrt(3)
        ).all(axis=1)
        need_distance = np.logical_and(outer_cube, np.logical_not(inner_cube))
        mask[inner_cube] = True
        mask[need_distance] = np.sum(
            np.power(particles["xyz_g"][need_distance], 2), axis=1
        ) < np.power(aperture, 2)

        for k, v in particles.items():
            particles[k] = v[mask]

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
