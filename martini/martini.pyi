from numpy import ndarray
from martini.beams import _BaseBeam
from martini.datacube import DataCube as DataCube
from martini.noise import _BaseNoise
from martini.sources.sph_source import SPHSource as SPHSource
from martini.spectral_models import _BaseSpectrum
from martini.sph_kernels import _BaseSPHKernel
from matplotlib.figure import Figure
import astropy.units as U

gc: bytes
martini_version: str

class _BaseMartini:
    source: SPHSource
    _datacube: DataCube
    beam: _BaseBeam
    noise: _BaseNoise
    sph_kernel: _BaseSPHKernel
    spectral_model: _BaseSpectrum
    quiet: bool

    def __init__(
        self,
        source: SPHSource | None = ...,
        datacube: DataCube | None = ...,
        beam: _BaseBeam | None = ...,
        noise: _BaseNoise | None = ...,
        sph_kernel: _BaseSPHKernel | None = ...,
        spectral_model: _BaseSpectrum | None = ...,
        quiet: bool | None = ...,
        _prune_kwargs: dict[str, bool | str] = ...,
    ) -> None: ...
    def init_spectra(self) -> None: ...
    def _prune_particles(
        self,
        spatial: bool = ...,
        spectral: bool = ...,
        mass: bool = ...,
        obj_type_str: str = ...,
    ) -> None: ...
    def _evaluate_pixel_spectrum(
        self,
        ij_px: tuple[int, int],
    ) -> tuple[slice, U.Quantity[U.Jy / U.arcsec**2]]: ...
    def _insert_pixel(
        self, insertion_slice: int | tuple | slice, insertion_data: ndarray
    ) -> None: ...
    def _insert_source_in_cube(
        self,
        skip_validation: bool = ...,
        progressbar: bool | None = ...,
        ncpu: int = ...,
        quiet: bool | None = ...,
    ) -> None: ...
    def reset(self) -> None: ...
    def preview(
        self,
        max_points: int = ...,
        fig: int = ...,
        lim: str | U.Quantity[U.kpc] | None = ...,
        vlim: str | U.Quantity[U.km / U.s] | None = ...,
        point_scaling: str = ...,
        title: str = ...,
        save: str | None = ...,
    ) -> Figure: ...

class Martini:
    def __init__(
        self,
        source: SPHSource | None = ...,
        datacube: DataCube | None = ...,
        beam: _BaseBeam | None = ...,
        noise: _BaseNoise | None = ...,
        sph_kernel: _BaseSPHKernel | None = ...,
        spectral_model: _BaseSpectrum | None = ...,
        quiet: bool | None = ...,
    ) -> None: ...
    @property
    def datacube(self) -> DataCube: ...
    def insert_source_in_cube(
        self,
        skip_validation: bool = ...,
        progressbar: bool | None = ...,
        ncpu: int = ...,
    ) -> None: ...
    def convolve_beam(self) -> None: ...
    def add_noise(self) -> None: ...
    def write_fits(
        self,
        filename: str,
        overwrite: bool = ...,
        obj_name: str = ...,
        channels: None = ...,
    ) -> None: ...
    def write_beam_fits(
        self, filename: str, overwrite: bool = ..., channels: None = ...
    ) -> None: ...
    def write_hdf5(
        self,
        filename: str,
        overwrite: bool = ...,
        memmap: bool = ...,
        compact: bool = ...,
        channels: None = ...,
    ) -> None: ...

class GlobalProfile(_BaseMartini):
    def __init__(
        self,
        source: SPHSource | None = ...,
        spectral_model: _BaseSpectrum | None = ...,
        n_channels: int = ...,
        channel_width: U.Quantity[U.km / U.s] = ...,
        spectral_centre: U.Quantity[U.km / U.s] = ...,
        quiet: bool = ...,
        channels: None = ...,
    ) -> None: ...
    def insert_source_in_spectrum(self) -> None: ...
    @property
    def spectrum(self) -> U.Quantity[U.Jy]: ...
    @property
    def channel_edges(self) -> U.Quantity[U.Hz] | U.Quantity[U.km / U.s]: ...
    @property
    def channel_mids(self) -> U.Quantity[U.Hz] | U.Quantity[U.km / U.s]: ...
    @property
    def velocity_channel_edges(self) -> U.Quantity[U.km / U.s]: ...
    @property
    def velocity_channel_mids(self) -> U.Quantity[U.km / U.s]: ...
    @property
    def frequency_channel_edges(self) -> U.Quantity[U.Hz]: ...
    @property
    def frequency_channel_mids(self) -> U.Quantity[U.Hz]: ...
    @property
    def channel_width(self) -> U.Quantity[U.Hz] | U.Quantity[U.km / U.s]: ...
    def reset(self) -> None: ...
    def plot_spectrum(
        self,
        fig: int = ...,
        title: str = ...,
        channels: str = ...,
        show_vsys: bool = ...,
        save: str | None = ...,
    ) -> Figure: ...
