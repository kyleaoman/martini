import typing as T
import astropy.units as U
import abc
from _typeshed import Incomplete
from abc import abstractmethod
from martini.datacube import DataCube as DataCube
from martini.beams import _BaseBeam
from numpy.random._generator import Generator

class _BaseNoise(metaclass=abc.ABCMeta):
    __metaclass__: Incomplete
    seed: int
    rng: Generator

    def __init__(self, seed: T.Optional[int] = ...) -> None: ...
    @abstractmethod
    def generate(self, datacube: DataCube, beam: _BaseBeam): ...
    def reset_rng(self) -> None: ...

class GaussianNoise(_BaseNoise):
    rms: U.Quantity[U.Jy * U.beam**-1]

    def __init__(
        self, rms: U.Quantity[U.Jy * U.beam**-1] = ..., seed: T.Optional[int] = ...
    ) -> None: ...
    def generate(
        self, datacube: DataCube, beam: _BaseBeam
    ) -> U.Quantity[U.Jy * U.arcsec**-2]: ...
