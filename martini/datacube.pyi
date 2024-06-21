import typing as T
import astropy.units as U
from astropy.wcs.wcs import WCS
from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame

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
    _wcs: T.Optional[WCS]
    n_px_x: int
    n_px_y: int
    n_channels: int
    stokes_axis: bool
    coordinate_frame: BaseRADecFrame
    specsys: str
    _freq_channel_mode: bool
    _channel_edges: T.Optional[T.Union[U.Quantity[U.Hz], U.Quantity[U.m / U.s]]]
    _channel_mids: T.Optional[T.Union[U.Quantity[U.Hz], U.Quantity[U.m / U.s]]]

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
    def from_wcs(cls, input_wcs: WCS, specsys=T.Optional[str]) -> T.Self: ...
    @property
    def units(
        self,
    ) -> T.Union[
        T.Tuple[
            U.Quantity[U.deg],
            U.Quantity[U.deg],
            T.Union[U.Quantity[U.Hz], U.Quantity[U.m / U.s]],
        ],
        T.Tuple[
            U.Quantity[U.deg],
            U.Quantity[U.deg],
            T.Union[U.Quantity[U.Hz], U.Quantity[U.m / U.s]],
            U.Quantity[U.dimensionless_unscaled],
        ],
    ]: ...
    def velocity_channels(self) -> None: ...  # deprecated
    def freq_channels(self) -> None: ...  # deprecated
    @property
    def wcs(self) -> WCS: ...
    @property
    def channel_mids(self) -> T.Union[U.Quantity[U.Hz], U.Quantity[U.m / U.s]]: ...
    @property
    def channel_edges(self) -> T.Union[U.Quantity[U.Hz], U.Quantity[U.m / U.s]]: ...
    @property
    def velocity_channel_mids(self) -> U.Quantity[U.m / U.s]: ...
    @property
    def velocity_channel_edges(self) -> U.Quantity[U.m / U.s]: ...
    @property
    def frequency_channel_mids(self) -> U.Quantity[U.Hz]: ...
    @property
    def frequency_channel_edges(self) -> U.Quantity[U.Hz]: ...
    @property
    def _stokes_index(self) -> T.Optional[int]: ...
    @property
    def channel_maps(self) -> T.Iterator[U.Quantity]: ...
    @property
    def spatial_slices(self) -> T.Iterator[U.Quantity]: ...
    @property
    def spectra(self) -> T.Iterator[U.Quantity]: ...
    def add_pad(self, pad: T.Tuple[int, int]) -> None: ...
    def drop_pad(self) -> None: ...
    def copy(self) -> T.Self: ...
    def save_state(self, filename: str, overwrite: bool = ...) -> None: ...
    @classmethod
    def load_state(cls, filename: str) -> T.Self: ...
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
