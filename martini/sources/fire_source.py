"""
Provides the :class:`~martini.sources.magneticum_source.MagneticumSource` class.

Facilitates working with FIRE simulations.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
from ..sph_kernels import _CubicSplineKernel, find_fwhm
from ..L_coords import L_coords
from astropy import units as U, constants as C
from astropy.coordinates import ICRS
from .sph_source import SPHSource

if TYPE_CHECKING:
    from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame


class FIRESource(SPHSource):
    """
    Class abstracting HI sources from publicly available FIRE snapshot and group data.

    For file access, see https://flathub.flatironinstitute.org/fire

    The authors of the :mod:`gizmo_analysis` package request the folowing: "If you use
    this package in work that you publish, please cite it, along the lines of: 'This work
    used GizmoAnalysis (http://ascl.net/2002.015), which first was used in Wetzel et al.
    2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W).'"

    Parameters
    ----------
    simulation_directory : str, optional
        Base directory containing FIRE simulation output.

    snapshot_directory : str, optional
        Directory within ``simulation_directory`` containing snapshots. If ``None``,
        the default defined by :mod:`gizmo_analysis` is assumed.

    snapshot : tuple, optional
        A 2-tuple specifying the snapshot. The first element is the type of identifier,
        a string chosen from ``"index"``, ``"redshift"`` or ``"scalefactor"``. The
        second element is the desired value of the identifier. For example setting
        ``snapshot=("scalefactor", 0.2)`` will select the closest snapshot to
        :math:`a=0.2`.

    host_number : int, optional
        Galaxy ("host") position in the catalogue, indexed from 1.

    assign_hosts : str, optional
        Method to compute galaxy centres. Iterative zoom-in based methods are
        recommended, options include ``"mass"`` (mass-weighted), ``"potential"``
        (potential-weighted) and ``"massfraction.metals"`` (metallicity-weighted).
        Other options are ``"track"`` (time-interpolated position) and ``"halo"``
        (from Rockstar halo catalogue). Setting ``True`` will attempt to find a
        sensible default by first trying ``"track"`` then ``"mass"``.

    convert_float32 : bool, optional
        If ``True``, convert floating point values to 32 bit precision to reduce
        memory usage.

    gizmo_io_verbose : bool, optional
        If ``True``, allow the gizmo.io module to print progress and diagnostic messages.

    distance : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity, added to the velocity from Hubble's law.

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
        simulation_directory: str = ".",
        snapshot_directory: str,
        snapshot: tuple[str, float] = ("redshift", 0.0),
        host_number: int = 1,
        assign_hosts: str = "mass",
        convert_float32: bool = False,
        gizmo_io_verbose: bool = False,
        distance: U.Quantity[U.Mpc],
        vpeculiar: U.Quantity[U.km / U.s] = 0 * U.km / U.s,
        rotation: Rotation | None = None,
        L_coords: L_coords | None = None,
        ra: U.Quantity[U.deg] = 0.0 * U.deg,
        dec: U.Quantity[U.deg] = 0.0 * U.deg,
        coordinate_frame: "BaseRADecFrame" = ICRS(),
    ) -> None:
        import gizmo_analysis as gizmo

        gizmo_read_kwargs = {
            "species": ["gas", "star"],
            "properties": [
                "position",
                "velocity",
                "temperature",
                "size",
                "density",
                "mass",
                "potential",
                "massfraction.metals.hydrogen",
                "hydrogen.neutral.fraction",
            ],
            "simulation_directory": simulation_directory,
            "snapshot_value_kind": snapshot[0],
            "snapshot_values": snapshot[1],
            "assign_hosts": assign_hosts,
            "host_number": host_number,
            "convert_float32": convert_float32,
            "verbose": gizmo_io_verbose,
        }
        if snapshot_directory is not None:
            gizmo_read_kwargs["snapshot_directory"] = snapshot_directory
        gizmo_snap = gizmo.io.Read.read_snapshots(
            **gizmo_read_kwargs,
        )
        particles = {
            "xyz_g": (
                gizmo_snap["gas"]["position"]
                - gizmo_snap.host["position"][host_number - 1]
            )
            * gizmo_snap.snapshot["scalefactor"]
            * U.kpc,
            "vxyz_g": (
                gizmo_snap["gas"]["velocity"]
                - gizmo_snap.host["velocity"][host_number - 1]
            )
            * U.km
            * U.s**-1,
            "T_g": gizmo_snap["gas"]["temperature"] * U.K,
            # see doi:10.1093/mnras/sty1241 Appendix B for molecular partition:
            "mHI_g": np.where(
                np.logical_and(
                    gizmo_snap["gas"]["temperature"] * U.K < 300 * U.K,
                    gizmo_snap["gas"]["density"] * U.Msun * U.kpc**-3
                    > C.m_p * 10 * U.cm**-3,
                ),
                0,
                gizmo_snap["gas"].prop("mass.hydrogen.neutral"),
            )
            * U.Msun,
            # per A. Wetzel, size / 0.5077 is radius of cpmpact support
            "hsm_g": gizmo_snap["gas"]["size"]
            / 0.5077
            * U.kpc
            * find_fwhm(_CubicSplineKernel().kernel),
        }
        super().__init__(
            **particles,
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            L_coords=L_coords,
            ra=ra,
            dec=dec,
            h=gizmo_snap.info["hubble"],
            coordinate_frame=coordinate_frame,
        )
        return
