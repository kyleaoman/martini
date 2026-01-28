"""Provide a specification for orienting a source relative to its angular momentum."""

from typing import NamedTuple
from astropy import units as U


class L_coords(NamedTuple):
    """
    Provide an unambiguous way to specify a source orientation based on angular momentum.

    The orientation is defined as follows. First the angular momentum vector of the
    central 1/3 of particles weighted by HI mass is calculated. The angular momentum
    vector is then rotated to point along the ``x`` axis (the line of sight). The disc is
    now face-on. The L_coords arguments are then applied. First the source can be rotated
    around its angular momentum vector by an angle ``az_rot``. Then it can be rotated to
    incline it to the line of sight by an angle ``incl``. Finally it can be rotated in the
    plane of the sky by an angle ``pa``. All rotations are right-handed.

    The initialisation arguments are:

     - ``incl`` (:class:`~astropy.units.Quantity`, optional): The inclination with units
       of angle, defaults to 0 degrees (face-on).
     - ``az_rot`` (:class:`~astropy.units.Quantity`, optional): The rotation about the
       angular momentum axis with units of angle, defaults to 0 degrees.
     - ``pa`` (:class:`~astropy.units.Quantity`, optional): The position angle on the sky
       with units of angle, defaults to 270 degrees.
    """

    incl: U.Quantity[U.deg] = 0.0 * U.deg
    az_rot: U.Quantity[U.deg] = 0.0 * U.deg
    pa: U.Quantity[U.deg] = 270.0 * U.deg
