"""Provide utilities for unit conversion."""

import numpy as np
import astropy.units as U


def MHI_to_Jy_inplace(x: U.Quantity[U.Msun]) -> None:
    """
    Apply the HI mass to flux density conversion, with no memory overhead.

    The conversion is:
    M_HI/Msun = 2.36x10^5 * (D/Mpc)^2 * (S_21/Jy km s^-1)

    Parameters
    ----------
    x : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of
        mass / length^2 / velocity.
    """
    input_units = U.Msun * U.Mpc**-2 * (U.km * U.s**-1) ** -1
    np.divide(x, 2.36e5, out=x)
    x *= U.Jy / input_units
    return
