import typing as T
import astropy.units as U
from astropy.wcs.wcs import WCS

HIfreq: U.Quantity[U.Hz]

class DataCube:
    px_size: U.Quantity[U.arcsec]
    arcsec2_to_pix: T.Tuple[
        U.Quantity[U.Jy * U.pix**-2],
        U.Quantity[U.Jy * U.arcsec**-2],
        T.Callable[[U.Quantity[U.Jy * U.pix**-2]], U.Quantity[U.Jy * U.arcsec**-2]],
        T.Callable[[U.Quantity[U.Jy * U.arcsec**-2]], U.Quantity[U.Jy * U.pix**-2]],
    ]
    channel_width: U.Quantity[U.km / U.s]
    velocity_centre: U.Quantity[U.km / U.s]
    ra: U.Quantity[U.deg]
    dec: U.Quantity[U.deg]
    padx: int
    pady: int
    _array: (
        U.Quantity[U.Jy * U.pix**-2]
        | U.Quantity[U.Jy * U.arcsec**-2]
        | U.Quantity[U.Jy * U.beam**-1]
    )
    wcs: WCS
    units: T.Tuple[U.Quantity[U.deg], U.Quantity[U.deg], U.Quantity[U.m / U.s]]
    n_px_x: int
    n_px_y: int
    n_channels: int
    stokes_axis: bool

    def __init__(
        self,
        n_px_x: int = ...,
        n_px_y: int = ...,
        n_channels: int = ...,
        px_size: U.Quantity[U.arcsec] = ...,
        channel_width: U.Quantity[U.arcsec] = ...,
        velocity_centre: U.Quantity[U.km / U.s] = ...,
        ra: U.Quantity[U.deg] = ...,
        dec: U.Quantity[U.deg] = ...,
        stokes_axis: bool = ...,
    ) -> None: ...
    def spatial_slices(self) -> T.Iterator[U.Quantity]: ...
    def spectra(self) -> T.Iterator[U.Quantity]: ...
    def freq_channels(self) -> None: ...
    def velocity_channels(self) -> None: ...
    def add_pad(self, pad: T.Tuple[int, int]) -> None: ...
    def drop_pad(self) -> None: ...
    def copy(self) -> "DataCube": ...
    def save_state(self, filename: str, overwrite: bool = ...) -> None: ...
    @classmethod
    def load_state(cls, filename: str) -> "DataCube": ...
