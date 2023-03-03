from numpy import ndarray
import astropy.units as U
import typing as T

def L_align(
    xyz: U.Quantity[U.kpc],
    vxyz: U.Quantity[U.kpc],
    m: U.Quantity[U.Msun],
    frac: float = ...,
    saverot: T.Optional[str] = ...,
    Laxis: str = ...,
) -> ndarray: ...
