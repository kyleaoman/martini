"""
Provides the :class:`~martini.sources.swiftgalaxy_source.SWIFTGalaxySource` class.

Facilitates working with SWIFT simulations as input.
"""

from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
from .sph_source import SPHSource
from ..L_coords import L_coords
from astropy import units as U
from astropy.coordinates import ICRS

if TYPE_CHECKING:
    from swiftsimio import cosmo_array
    from swiftgalaxy import SWIFTGalaxy
    from astropy.coordinates.builtin_frames.baseradecframe import BaseRADecFrame


class SWIFTGalaxySource(SPHSource):
    """
    Load HI sources from SWIFT simulations via :mod:`swiftsimio` and :mod:`swiftgalaxy`.

    Parameters
    ----------
    galaxy : ~swiftgalaxy.reader.SWIFTGalaxy
        Instance of a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

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

    _mHI_g : ~swiftsimio.objects.cosmo_array, optional
        If the ``galaxy`` does not provide ``galaxy.gas.atomic_hydrogen_masses``, provide
        the particle HI masses here.
    """

    def __init__(
        self,
        galaxy: "SWIFTGalaxy",
        *,
        distance: U.Quantity[U.Mpc],
        vpeculiar: U.Quantity[U.km / U.s] = 0 * U.km / U.s,
        rotation: Rotation | None = Rotation.identity(),
        L_coords: L_coords | None = None,
        ra: U.Quantity[U.deg] = 0.0 * U.deg,
        dec: U.Quantity[U.deg] = 0.0 * U.deg,
        coordinate_frame: "BaseRADecFrame" = ICRS(),
        _mHI_g: "cosmo_array" = None,
    ) -> None:
        h = galaxy.metadata.cosmology.h
        mHI_g = (
            galaxy.gas.atomic_hydrogen_masses.to_astropy() if _mHI_g is None else _mHI_g
        )
        # SWIFT guarantees smoothing lengths are 2x kernel std
        # We should convert the 2x std of the kernel intrinsically used in the simulation
        # to the FWHM of the same kernel. For this we need to detect which kernel was
        # used.
        kernel_function = galaxy.metadata.hydro_scheme["Kernel function"].decode()
        compact_support_per_h = {
            "Quartic spline (M5)": 2.018932,
            "Quintic spline (M6)": 2.195775,
            "Cubic spline (M4)": 1.825742,
            "Wendland C2": 1.936492,
            "Wendland C4": 2.207940,
            "Wendland C6": 2.449490,
        }[kernel_function]
        fwhm_per_compact_support = {
            "Quartic spline (M5)": 0.637756,
            "Quintic spline (M6)": 0.577395,
            "Cubic spline (M4)": 0.722352,
            "Wendland C2": 0.627620,
            "Wendland C4": 0.560649,
            "Wendland C6": 0.504964,
        }[kernel_function]
        hsm_g = (
            galaxy.gas.smoothing_lengths.to_astropy()
            * compact_support_per_h
            * fwhm_per_compact_support
        )
        particles = {
            "xyz_g": galaxy.gas.coordinates.to_astropy(),
            "vxyz_g": galaxy.gas.velocities.to_astropy(),
            "T_g": galaxy.gas.temperatures.to_astropy(),
            "hsm_g": hsm_g,
            "mHI_g": mHI_g,
        }
        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            L_coords=L_coords,
            ra=ra,
            dec=dec,
            h=h,
            coordinate_frame=coordinate_frame,
            coordinate_axis=1,
            **particles,
        )
        return
