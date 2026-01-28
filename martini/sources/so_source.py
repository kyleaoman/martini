"""
Provides the :class:`~martini.sources.so_source.SOSource` class.

Enables using the :mod:`simobj` interface to simulations.
"""

from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
import astropy.units as U
from astropy.coordinates import ICRS
from .sph_source import SPHSource
from ..L_coords import L_coords

if TYPE_CHECKING:
    from simobj import SimObj
    from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame


class SOSource(SPHSource):
    """
    Load HI sources using the :mod:`simobj` package for interface to simulation data.

    This class accesses simulation data via the :mod:`simobj` package
    (https://github.com/kyleaoman/simobj); see the documentation of that package for
    further configuration instructions.

    Parameters
    ----------
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

    coordinate_frame : ~astropy.coordinates.builtin_frames.baseradec.BaseRADecFrame
        Optional. The coordinate frame assumed in converting particle coordinates to RA
        and Dec, and for transforming coordinates and velocities to the data cube frame.
        The frame needs to have a well-defined velocity as well as spatial origin.
        Recommended frames are :class:`~astropy.coordinates.GCRS`,
        :class:`~astropy.coordinates.ICRS`, :class:`~astropy.coordinates.HCRS`,
        :class:`~astropy.coordinates.LSRK`, :class:`~astropy.coordinates.LSRD` or
        :class:`~astropy.coordinates.LSR`. The frame should be passed initialized, e.g.
        ``ICRS()`` (not just ``ICRS``).

    SO_args : dict, optional
        Dictionary of keyword arguments to pass to a call to
        :class:`simobj.simobj.SimObj`.
        Arguments are: ``obj_id``, ``snap_id``, ``mask_type``, ``mask_args``,
        ``mask_kwargs``, ``configfile``, ``simfiles_configfile``, ``ncpu``.
        See :mod:`simobj` package documentation for details. Provide ``SO_args`` or
        ``SO_instance``, not both.

    SO_instance : simobj.simobj.SimObj, optional
        An initialized :class:`simobj.simobj.SimObj` object. Provide ``SO_instance``
        or ``SO_args``, not both.

    rescale_hsm_g : float
        Factor by which to multiply the smoothing lengths returned by the
        :class:`simobj.simobj.SimObj` class to obtain FWHM smoothing lenghts.
    """

    def __init__(
        self,
        *,
        distance: U.Quantity[U.Mpc],
        vpeculiar: U.Quantity[U.km / U.s] = 0 * U.km / U.s,
        rotation: Rotation | None = None,
        L_coords: L_coords | None = None,
        ra: U.Quantity[U.deg] = 0.0 * U.deg,
        dec: U.Quantity[U.deg] = 0.0 * U.deg,
        coordinate_frame: "BaseRADecFrame" = ICRS(),
        SO_args: dict | None = None,
        SO_instance: "SimObj | None" = None,
        rescale_hsm_g: float = 1.0,
    ) -> None:
        from simobj import SimObj  # optional dependency for this source class

        self._SO_args = SO_args
        self.rescale_hsm_g = rescale_hsm_g
        if (SO_args is not None) and (SO_instance is not None):
            raise ValueError(
                "martini.source.SOSource: Provide SO_args or SO_instance, not both."
            )
        elif SO_args is not None:
            with SimObj(**self._SO_args) as SO:
                particles = {
                    "T_g": SO.T_g,
                    "mHI_g": SO.mHI_g,
                    "xyz_g": SO.xyz_g,
                    "vxyz_g": SO.vxyz_g,
                    "hsm_g": SO.hsm_g * self.rescale_hsm_g,
                }
                super().__init__(
                    distance=distance,
                    rotation=rotation,
                    L_coords=L_coords,
                    ra=ra,
                    dec=dec,
                    h=SO.h,
                    **particles,
                )
        elif SO_instance is not None:
            particles = {
                "T_g": SO_instance.T_g,
                "mHI_g": SO_instance.mHI_g,
                "xyz_g": SO_instance.xyz_g,
                "vxyz_g": SO_instance.vxyz_g,
                "hsm_g": SO_instance.hsm_g,
            }
            super().__init__(
                distance=distance,
                vpeculiar=vpeculiar,
                rotation=rotation,
                L_coords=L_coords,
                ra=ra,
                dec=dec,
                h=SO_instance.h,
                coordinate_frame=coordinate_frame,
                **particles,
            )
        else:
            raise ValueError(
                "martini.sources.SOSource: Provide one of SO_args or SO_instance."
            )
        return
