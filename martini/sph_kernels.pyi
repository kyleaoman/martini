import abc
from _typeshed import Incomplete
from abc import abstractmethod
import typing as T
from types import EllipsisType
from numpy import ndarray
import astropy.units as U

def find_fwhm(f: T.Callable[[ndarray], float]) -> float: ...

class _BaseSPHKernel(metaclass=abc.ABCMeta):
    __metaclass__: Incomplete

    def __init__(self) -> None: ...
    def px_weight(
        self, dij: U.Quantity[U.pix], mask: T.Optional[ndarray] = ...
    ) -> U.Quantity[U.pix**2]: ...
    def confirm_validation(self, noraise: bool = ..., quiet: bool = ...) -> bool: ...
    def eval_kernel(
        self, r: ndarray | U.Quantity[U.arcsec], h: ndarray | U.Quantity[U.arcsec]
    ) -> U.Quantity | ndarray | float: ...

    sm_lengths: U.Quantity[U.arcsec]
    sm_ranges: ndarray

    def apply_mask(self, mask: ndarray) -> None: ...
    @abstractmethod
    def kernel(self, q) -> ndarray: ...
    @abstractmethod
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    @abstractmethod
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class WendlandC2Kernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class WendlandC6Kernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class CubicSplineKernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class GaussianKernel(_BaseSPHKernel):
    no6sigwarn: bool
    truncate: float
    lims: T.Tuple[int, int]
    norm: float
    size_in_fwhm: float

    def __init__(self, truncate: float = ...) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class DiracDeltaKernel(_BaseSPHKernel):
    max_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class AdaptiveKernel(_BaseSPHKernel):
    kernels: T.Tuple[_BaseSPHKernel]
    size_in_fwhm: T.Optional[T.Tuple[float]]

    def __init__(self, kernels: T.Tuple[_BaseSPHKernel]) -> None: ...

    kernel_indices: ndarray

    def apply_mask(self, mask: ndarray) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...

class QuarticSplineKernel(_BaseSPHKernel):
    min_valid_size: float
    size_in_fwhm: float

    def __init__(self) -> None: ...
    def kernel(self, q: ndarray) -> ndarray: ...
    def kernel_integral(
        self,
        dij: U.Quantity[U.pix],
        h: U.Quantity[U.pix],
        mask: ndarray | EllipsisType | slice = ...,
    ) -> ndarray: ...
    def validate(
        self, sm_lengths: U.Quantity[U.pix], noraise: bool = ..., quiet: bool = ...
    ) -> ndarray: ...
