import astropy.units as U
from astropy.coordinates import ICRS, SkyCoord
from martini.datacube import DataCube
from numpy import ndarray
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
    pixcoords: U.Quantity[U.pix]
    input_mass: U.Quantity[U.Msun]
    skycoords: SkyCoord | None

    def __init__(
        self,
        distance: U.Quantity[U.Mpc] = ...,
        vpeculiar: U.Quantity[U.km / U.s] = ...,
        rotation: dict[
            str,
            ndarray
            | tuple[U.Quantity[U.deg], U.Quantity[U.deg]]
            | tuple[U.Quantity[U.deg], U.Quantity[U.deg], U.Quantity[U.deg]]
            | tuple[str, U.Quantity[U.deg]],
        ] = ...,
        ra: U.Quantity[U.deg] = ...,
        dec: U.Quantity[U.deg] = ...,
        h: float = ...,
        T_g: U.Quantity[U.K] | None = ...,
        mHI_g: U.Quantity[U.Msun] | None = ...,
        xyz_g: U.Quantity[U.kpc] | None = ...,
        vxyz_g: U.Quantity[U.km / U.s] | None = ...,
        hsm_g: U.Quantity[U.kpc] | None = ...,
        coordinate_axis: int | None = ...,
        coordinate_frame: BaseRADecFrame = ...,
    ) -> None: ...
    def _init_skycoords(self, _reset: bool = ...) -> None: ...
    def _init_pixcoords(self, datacube: DataCube, origin: int = ...) -> None: ...
    def apply_mask(self, mask: ndarray) -> None: ...
    def rotate(
        self,
        axis_angle: tuple[str, U.Quantity[U.deg]] | None = ...,
        rotmat: ndarray | None = ...,
        L_coords: tuple[U.Quantity[U.deg], U.Quantity[U.deg]]
        | tuple[U.Quantity[U.deg], U.Quantity[U.deg], U.Quantity[U.deg]]
        | None = ...,
    ) -> None: ...
    def translate(self, translation_vector: U.Quantity[U.kpc]) -> None: ...
    def boost(self, boost_vector: U.Quantity[U.km / U.s]) -> None: ...
    def save_current_rotation(self, fname: str) -> None: ...
    def preview(
        self,
        max_points: int = ...,
        fig: int = ...,
        lim: U.Quantity[U.kpc] | None = ...,
        vlim: U.Quantity[U.km / U.s] | None = ...,
        point_scaling: str = ...,
        title: str = ...,
        save: str | None = ...,
    ) -> Figure: ...
