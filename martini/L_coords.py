"""Provide a specification for orienting a source relative to its angular momentum."""

from dataclasses import dataclass
from astropy import units as U


@dataclass(frozen=True)
class L_coords:
    """
    Provide an unambiguous way to specify a source orientation based on angular momentum.

    The orientation is defined as follows. First the angular momentum vector of the
    central 1/3 of particles weighted by HI mass is calculated. The angular momentum
    vector is then rotated to point along the ``x`` axis (the line of sight). The disc is
    now face-on. The L_coords arguments are then applied. First the source can be rotated
    around its angular momentum vector by an angle ``az_rot``. Then it can be rotated to
    incline it to the line of sight by an angle ``incl``. Finally it can be rotated in the
    plane of the sky by an angle ``pa``. All rotations are right-handed.

    Parameters
    ----------
    incl : ~astropy.units.Quantity`, optional
        The inclination with units of angle, defaults to 0 degrees (face-on).
    az_rot : ~astropy.units.Quantity, optional
        The rotation about the angular momentum axis with units of angle, defaults to 0
        degrees.
    pa : ~astropy.units.Quantity, optional
        The position angle on the sky with units of angle, defaults to 270 degrees.
    """

    incl: U.Quantity[U.deg] = 0.0 * U.deg
    az_rot: U.Quantity[U.deg] = 0.0 * U.deg
    pa: U.Quantity[U.deg] = 270.0 * U.deg

    @U.quantity_input
    def __init__(
        self,
        incl: U.Quantity[U.deg] = 0.0 * U.deg,
        az_rot: U.Quantity[U.deg] = 0.0 * U.deg,
        pa: U.Quantity[U.deg] = 270.0 * U.deg,
    ) -> None:
        object.__setattr__(self, "incl", incl)
        object.__setattr__(self, "az_rot", az_rot)
        object.__setattr__(self, "pa", pa)
        return
