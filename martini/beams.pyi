from _typeshed import Incomplete
import astropy.units as U
import abc
from collections.abc import Callable
import numpy as np
from abc import abstractmethod
from martini.datacube import DataCube as DataCube

class _BaseBeam(metaclass=abc.ABCMeta):
    __metaclass__: Incomplete
    bmaj: U.Quantity[U.arcsec]
    bmin: U.Quantity[U.arcsec]
    bpa: U.Quantity[U.deg]
    px_size: U.Quantity[U.arcsec] | None
    kernel: U.Quantity[U.dimensionless_unscaled] | None
    area: U.Quantity[U.arcsec**2]

    def __init__(
        self,
        bmaj: U.Quantity[U.arcsec] = ...,
        bmin: U.Quantity[U.arcsec] = ...,
        bpa: U.Quantity[U.deg] = ...,
    ) -> None: ...
    def needs_pad(self) -> tuple[int, int]: ...

    vel: U.Quantity[U.km / U.s]
    ra: U.Quantity[U.deg]
    dec: U.Quantity[U.deg]

    def init_kernel(self, datacube: DataCube) -> None: ...
    @abstractmethod
    def f_kernel(self): ...
    @abstractmethod
    def kernel_size_px(self): ...
    @abstractmethod
    def init_beam_header(self): ...

class GaussianBeam(_BaseBeam, metaclass=abc.ABCMeta):
    truncate: float

    def __init__(
        self,
        bmaj: U.Quantity[U.arcsec] = ...,
        bmin: U.Quantity[U.arcsec] = ...,
        bpa: U.Quantity[U.deg] = ...,
        truncate: float = ...,
    ) -> None: ...
    def f_kernel(
        self,
    ) -> Callable[[float | np.ndarray, float | np.ndarray], U.Quantity]: ...
    def kernel_size_px(self) -> tuple[int, int]: ...
