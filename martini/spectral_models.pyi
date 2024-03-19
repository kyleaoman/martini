import abc
import numpy as np
from _typeshed import Incomplete
from abc import abstractmethod
from martini.datacube import DataCube as DataCube
from martini.sources.sph_source import SPHSource as SPHSource
import typing as T
import astropy.units as U
from types import EllipsisType

class _BaseSpectrum(metaclass=abc.ABCMeta):
    __metaclass__: Incomplete
    spectra: T.Optional[U.Quantity[U.Jy]]
    ncpu: int
    spec_dtype: type
    spectral_function_extra_data: T.Optional[T.Dict[str, T.Any]]

    def __init__(self, ncpu: T.Optional[int] = ..., spec_dtype: type = ...) -> None: ...
    def init_spectra(self, source: SPHSource, datacube: DataCube) -> None: ...
    def evaluate_spectra(
        self,
        source: SPHSource,
        datacube: DataCube,
        mask: T.Union[slice, EllipsisType] = ...,
    ) -> None: ...
    @abstractmethod
    def half_width(self, source: SPHSource) -> U.Quantity[U.km / U.s]: ...
    @abstractmethod
    def spectral_function(
        self,
        a: U.Quantity[U.km / U.s],
        b: U.Quantity[U.km / U.s],
        vmids: U.Quantity[U.km / U.s],
    ) -> U.Quantity[U.dimensionless_unscaled]: ...
    def init_spectral_function_extra_data(
        self,
        source: SPHSource,
        datacube: DataCube,
        mask: T.Union[slice, EllipsisType] = ...,
    ) -> None: ...

class GaussianSpectrum(_BaseSpectrum):
    sigma_mode: T.Union[str, U.Quantity[U.km / U.s]]

    def __init__(
        self,
        sigma: T.Union[str, U.Quantity[U.km / U.s]] = ...,
        ncpu: T.Optional[int] = None,
        spec_dtype: type = ...,
    ) -> None: ...
    def spectral_function(
        self,
        a: U.Quantity[U.km / U.s],
        b: U.Quantity[U.km / U.s],
        vmids: U.Quantity[U.km / U.s],
    ) -> U.Quantity[U.dimensionless_unscaled]: ...
    def init_spectral_function_extra_data(
        self,
        source: SPHSource,
        datacube: DataCube,
        mask: T.Union[slice, EllipsisType] = ...,
    ) -> None: ...
    def half_width(self, source: SPHSource) -> U.Quantity[U.km / U.s]: ...

class DiracDeltaSpectrum(_BaseSpectrum):
    def __init__(self, ncpu: T.Optional[int] = ..., spec_dtype: type = ...) -> None: ...
    def spectral_function(
        self,
        a: U.Quantity[U.km / U.s],
        b: U.Quantity[U.km / U.s],
        vmids: U.Quantity[U.km / U.s],
    ) -> U.Quantity[U.dimensionless_unscaled]: ...
    def half_width(self, source: SPHSource) -> U.Quantity[U.km / U.s]: ...
