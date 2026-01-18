from typing import Self
from collections.abc import Callable, Iterator
import astropy.units as U
from astropy.wcs.wcs import WCS
from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame

HIfreq: U.Quantity[U.Hz]

class DataCube:
    px_size: U.Quantity[U.arcsec]
    arcsec2_to_pix: tuple[
        U.Quantity[U.Jy * U.pix**-2],
        U.Quantity[U.Jy * U.arcsec**-2],
        Callable[[U.Quantity[U.Jy * U.pix**-2]], U.Quantity[U.Jy * U.arcsec**-2]],
        Callable[[U.Quantity[U.Jy * U.arcsec**-2]], U.Quantity[U.Jy * U.pix**-2]],
    ]
    channel_width: U.Quantity[U.km / U.s]
    spectral_centre: U.Quantity[U.km / U.s]
    ra: U.Quantity[U.deg]
    dec: U.Quantity[U.deg]
    padx: int
    pady: int
    _array: (
        U.Quantity[U.Jy * U.pix**-2]
        | U.Quantity[U.Jy * U.arcsec**-2]
        | U.Quantity[U.Jy * U.beam**-1]
    )
    _wcs: WCS | None
    n_px_x: int
    n_px_y: int
    n_channels: int
    stokes_axis: bool
    coordinate_frame: BaseRADecFrame
    specsys: str
    _freq_channel_mode: bool
    _channel_edges: U.Quantity[U.Hz] | U.Quantity[U.m / U.s] | None
    _channel_mids: U.Quantity[U.Hz] | U.Quantity[U.m / U.s] | None

    def __init__(
        self,
        n_px_x: int = ...,
        n_px_y: int = ...,
        n_channels: int = ...,
        px_size: U.Quantity[U.arcsec] = ...,
        channel_width: U.Quantity[U.arcsec] = ...,
        spectral_centre: U.Quantity[U.km / U.s] = ...,
        ra: U.Quantity[U.deg] = ...,
        dec: U.Quantity[U.deg] = ...,
        stokes_axis: bool = ...,
        coordinate_frame: BaseRADecFrame = ...,
        specsys: str = ...,
        velocity_centre: None = ...,  # deprecated
    ) -> None: ...
    @classmethod
    def from_wcs(cls, input_wcs: WCS, specsys=str | None) -> Self: ...
    @property
    def units(
        self,
    ) -> (
        tuple[
            U.Quantity[U.deg],
            U.Quantity[U.deg],
            U.Quantity[U.Hz] | U.Quantity[U.m / U.s],
        ]
        | tuple[
            U.Quantity[U.deg],
            U.Quantity[U.deg],
            U.Quantity[U.Hz] | U.Quantity[U.m / U.s],
            U.Quantity[U.dimensionless_unscaled],
        ]
    ): ...
    def velocity_channels(self) -> None: ...  # deprecated
    def freq_channels(self) -> None: ...  # deprecated
    @property
    def wcs(self) -> WCS: ...
    @property
    def channel_mids(self) -> U.Quantity[U.Hz] | U.Quantity[U.m / U.s]: ...
    @property
    def channel_edges(self) -> U.Quantity[U.Hz] | U.Quantity[U.m / U.s]: ...
    @property
    def velocity_channel_mids(self) -> U.Quantity[U.m / U.s]: ...
    @property
    def velocity_channel_edges(self) -> U.Quantity[U.m / U.s]: ...
    @property
    def frequency_channel_mids(self) -> U.Quantity[U.Hz]: ...
    @property
    def frequency_channel_edges(self) -> U.Quantity[U.Hz]: ...
    @property
    def _stokes_index(self) -> int | None: ...
    @property
    def channel_maps(self) -> Iterator[U.Quantity]: ...
    @property
    def spatial_slices(self) -> Iterator[U.Quantity]: ...
    @property
    def spectra(self) -> Iterator[U.Quantity]: ...
    def add_pad(self, pad: tuple[int, int]) -> None: ...
    def drop_pad(self) -> None: ...
    def copy(self) -> Self: ...
    def save_state(self, filename: str, overwrite: bool = ...) -> None: ...
    @classmethod
    def load_state(cls, filename: str) -> Self: ...
    def __repr__(self) -> str: ...

class _GlobalProfileDataCube(DataCube):
    def __init__(
        self,
        n_channels: int = ...,
        channel_width: U.Quantity[U.arcsec] = ...,
        spectral_centre: U.Quantity[U.km / U.s] = ...,
        specsys: str = ...,
        velocity_centre: None = ...,  # deprecated
    ) -> None: ...
