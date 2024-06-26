import abc
from _typeshed import Incomplete
from abc import abstractmethod
import typing as T
from types import EllipsisType
from numpy import ndarray
import astropy.units as U
from martini.datacube import DataCube as DataCube
from martini.sources.sph_source import SPHSource as SPHSource

def find_fwhm(f: T.Callable[[ndarray], float]) -> float: ...

class _BaseSPHKernel(metaclass=abc.ABCMeta):
    __metaclass__: Incomplete

    def __init__(self) -> None: ...
    def _px_weight(
        self, dij: U.Quantity[U.pix], mask: T.Optional[ndarray] = ...
    ) -> U.Quantity[U.pix**2]: ...
    def _confirm_validation(self, noraise: bool = ..., quiet: bool = ...) -> bool: ...
    def _validate_error(
        self,
        msg: str,
        sm_lengths: U.Quantity[U.pix],
        valid: ndarray,
        noraise: bool = ...,
        quiet: bool = ...,
    ) -> None: ...
    def eval_kernel(
        self, r: ndarray | U.Quantity[U.arcsec], h: ndarray | U.Quantity[U.arcsec]
    ) -> U.Quantity | ndarray | float: ...

    sm_lengths: U.Quantity[U.arcsec]
    sm_ranges: ndarray

    def _apply_mask(self, mask: ndarray) -> None: ...
    def _init_sm_lengths(
        self, source: T.Optional[SPHSource] = ..., datacube: T.Optional[DataCube] = ...
    ) -> None: ...
    @abstractmethod
    def kernel(self, q) -> ndarray: ...
    @abstractmethod
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    @abstractmethod
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class _WendlandC2Kernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class _WendlandC6Kernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class _CubicSplineKernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class _GaussianKernel(_BaseSPHKernel):
    no6sigwarn: bool
    truncate: float
    lims: T.Tuple[int, int]
    norm: float
    size_in_fwhm: float

    def __init__(self, truncate: float = ...) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class DiracDeltaKernel(_BaseSPHKernel):
    max_valid_size: float
    size_in_fwhm: float

    def __init__(self, size_in_fwhm: float = ...) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class _AdaptiveKernel(_BaseSPHKernel):
    kernels: T.Tuple[_BaseSPHKernel]
    size_in_fwhm: T.Optional[T.Tuple[float]]

    def __init__(self, kernels: T.Tuple[_BaseSPHKernel]) -> None: ...

    kernel_indices: ndarray

    def _init_sm_lengths(
        self, source: T.Optional[SPHSource] = ..., datacube: T.Optional[DataCube] = ...
    ) -> None: ...
    def _apply_mask(self, mask: ndarray) -> None: ...
    def eval_kernel(
        self, r: ndarray | U.Quantity[U.arcsec], h: ndarray | U.Quantity[U.arcsec]
    ) -> U.Quantity | ndarray | float: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class _QuarticSplineKernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def _kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def _validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class WendlandC2Kernel(_AdaptiveKernel):
    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...

class WendlandC6Kernel(_AdaptiveKernel):
    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...

class CubicSplineKernel(_AdaptiveKernel):
    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...

class GaussianKernel(_AdaptiveKernel):
    def __init__(self, truncate: float = ...) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...

class QuarticSplineKernel(_AdaptiveKernel):
    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...

class AdaptiveKernel(object):
    def __init__(self, *args, **kwargs) -> None: ...
