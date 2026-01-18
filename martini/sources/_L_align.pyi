from numpy import ndarray
import astropy.units as U

def L_align(
    xyz: U.Quantity[U.kpc],
    vxyz: U.Quantity[U.kpc],
    m: U.Quantity[U.Msun],
    frac: float = ...,
    saverot: str | None = ...,
    Laxis: str = ...,
) -> ndarray: ...
