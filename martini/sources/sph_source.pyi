import astropy.units as U
from astropy.coordinates import ICRS
from martini.datacube import DataCube
from numpy import ndarray
import typing as T
from matplotlib.figure import Figure
from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame

class SPHSource:
    h: float
    T_g: U.Quantity[U.K]
    mHI_g: U.Quantity[U.Msun]
    coordinates_g: U.Quantity[U.kpc]
    hsm_g: U.Quantity[U.kpc]
    npart: int
    ra: U.Quantity[U.deg]
    dec: U.Quantity[U.deg]
    distance: U.Quantity[U.Mpc]
    vpeculiar: U.Quantity[U.km / U.s]
    current_rotation: ndarray
    vhubble: U.Quantity[U.km / U.s]
    vsys: U.Quantity[U.km / U.s]
    sky_coordinates: ICRS
    coordinate_frame: BaseRADecFrame

    def __init__(
        self,
        distance: U.Quantity[U.Mpc] = ...,
        vpeculiar: U.Quantity[U.km / U.s] = ...,
        rotation: T.Dict[
            str,
            ndarray
            | T.Tuple[U.Quantity[U.deg], U.Quantity[U.deg]]
            | T.Tuple[U.Quantity[U.deg], U.Quantity[U.deg], U.Quantity[U.deg]]
            | T.Tuple[str, U.Quantity[U.deg]],
        ] = ...,
        ra: U.Quantity[U.deg] = ...,
        dec: U.Quantity[U.deg] = ...,
        h: float = ...,
        T_g: T.Optional[U.Quantity[U.K]] = ...,
        mHI_g: T.Optional[U.Quantity[U.Msun]] = ...,
        xyz_g: T.Optional[U.Quantity[U.kpc]] = ...,
        vxyz_g: T.Optional[U.Quantity[U.km / U.s]] = ...,
        hsm_g: T.Optional[U.Quantity[U.kpc]] = ...,
        coordinate_axis: T.Optional[int] = ...,
        coordinate_frame: BaseRADecFrame = ...,
    ) -> None: ...
    def _init_skycoords(self, _reset: bool = ...) -> None: ...
    def _init_pixcoords(self, datacube: DataCube, origin: int = ...) -> None: ...
    def apply_mask(self, mask: ndarray) -> None: ...
    def rotate(
        self,
        axis_angle: T.Optional[T.Tuple[str, U.Quantity[U.deg]]] = ...,
        rotmat: T.Optional[ndarray] = ...,
        L_coords: T.Optional[
            T.Tuple[U.Quantity[U.deg], U.Quantity[U.deg]]
            | T.Tuple[U.Quantity[U.deg], U.Quantity[U.deg], U.Quantity[U.deg]]
        ] = ...,
    ) -> None: ...
    def translate(self, translation_vector: U.Quantity[U.kpc]) -> None: ...
    def boost(self, boost_vector: U.Quantity[U.km / U.s]) -> None: ...
    def save_current_rotation(self, fname: str) -> None: ...
    def preview(
        self,
        max_points: int = ...,
        fig: int = ...,
        lim: T.Optional[U.Quantity[U.kpc]] = ...,
        vlim: T.Optional[U.Quantity[U.km / U.s]] = ...,
        point_scaling: str = ...,
        title: str = ...,
        save: T.Optional[str] = ...,
    ) -> Figure: ...
