"""
Provides the :class:`~martini.sources.colibre_source.ColibreSource` class.

Facilitates working with Colibre simulations as input.
"""

from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
from astropy import units as U
from astropy.coordinates import ICRS
from .swiftgalaxy_source import SWIFTGalaxySource
from ..L_coords import L_coords

if TYPE_CHECKING:
    from swiftgalaxy import SWIFTGalaxy
    from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame


class ColibreSource(SWIFTGalaxySource):
    """
    Class abstracting HI sources designed to work with Colibre simulations.

    Uses the :mod:`swiftsimio` and :mod:`swiftgalaxy` modules.

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
    ) -> None:
        # No special functionality wanted/needed:
        super().__init__(
            galaxy,
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            L_coords=L_coords,
            ra=ra,
            dec=dec,
            coordinate_frame=coordinate_frame,
            _mHI_g=galaxy.gas.masses.to_astropy()
            * galaxy.gas.element_mass_fractions.hydrogen.to_astropy()
            * galaxy.gas.species_fractions.HI.to_astropy(),
        )
        return
