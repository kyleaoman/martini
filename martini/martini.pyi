import typing as T
from martini.beams import _BaseBeam
from martini.datacube import DataCube as DataCube
from martini.noise import _BaseNoise
from martini.sources.sph_source import SPHSource as SPHSource
from martini.spectral_models import _BaseSpectrum
from martini.sph_kernels import _BaseSPHKernel

gc: bytes

class Martini:
    source: SPHSource
    datacube: DataCube
    beam: _BaseBeam
    noise: _BaseNoise
    sph_kernel: _BaseSPHKernel
    spectral_model: _BaseSpectrum
    quiet: bool

    def __init__(
        self,
        source: T.Optional[SPHSource] = ...,
        datacube: T.Optional[DataCube] = ...,
        beam: T.Optional[_BaseBeam] = ...,
        noise: T.Optional[_BaseNoise] = ...,
        sph_kernel: T.Optional[_BaseSPHKernel] = ...,
        spectral_model: T.Optional[_BaseSpectrum] = ...,
        quiet: T.Optional[bool] = ...,
    ) -> None: ...
    def convolve_beam(self) -> None: ...
    def add_noise(self) -> None: ...
    def insert_source_in_cube(
        self,
        skip_validation: bool = ...,
        progressbar: T.Optional[bool] = ...,
        ncpu: int = ...,
    ) -> None: ...
    def write_fits(
        self, filename: str, channels: str = ..., overwrite: bool = ...
    ) -> None: ...
    def write_beam_fits(
        self, filename: str, channels: str = ..., overwrite: bool = ...
    ) -> None: ...
    def write_hdf5(
        self,
        filename: str,
        channels: str = ...,
        overwrite: bool = ...,
        memmap: bool = ...,
        compact: bool = ...,
    ) -> None: ...
    def reset(self) -> None: ...
