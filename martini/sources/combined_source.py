"""
Provides a :class:`~martini.sources.combined_source.CombinedSource` class.

This can be used to join multiple individual sources together for use in a single mock
observation.
"""

import numpy as np
from astropy import units as U
from martini.sources.sph_source import SPHSource
from martini.datacube import DataCube
from ..L_coords import L_coords
from scipy.spatial.transform import Rotation


class CombinedSource(SPHSource):
    """
    Can combine multiple sources into a single source for a mock observation.

    It is often convenient to load source objects using separate source classes, for
    example perhaps multiple galaxies each loaded with an
    :class:`~martini.sources.eagle_source.EAGLESource`. However, it would be inefficient
    to run :mod:`martini`'s source insertion routine multiple times, so it is better to
    combined the multiple sources into one, first, using this class.

    Each source will retain its respective RA, Dec, distance, peculiar velocity, etc.
    These should all be set when creating the individual sources.

    Parameters
    ----------
    sources : list
        List of source objects to be combined (of type
        :class:`~martini.sources.sph_source.SPHSource` or derived classes).

    Examples
    --------
    .. code-block:: python

        from astropy import units as U
        from martini.souces import SPHSource, CombinedSource
        from scipy.spatial.transform import Rotation

        n = 1000
        r = np.random.rand(n) * 5 * U.kpc
        v = 100 * U.km / U.s
        t = np.random.rand(n) * 2 * np.pi * U.rad

        source1 = SPHSource(
            distance=10 * U.Mpc,
            vpeculiar=30 * U.km / U.s,
            rotation=Rotation.from_euler("y", (30 * U.deg).to_value(U.rad)),
            ra=49.95 * U.deg,
            dec=29.95 * U.deg,
            T_g=np.ones(n) * U.K,
            mHI_g=np.ones(n) * 1e7 * U.Msun,
            xyz_g=U.Quantity(
                [
                    np.zeros(n) * U.kpc,
                    r * np.cos(t),
                    r * np.sin(t),
                ]
            ),
            vxyz_g=U.Quantity(
                [
                    np.zeros(n) * U.km / U.s,
                    -v * np.sin(t),
                    v * np.cos(t),
                ]
            ),
            hsm_g=np.ones(n) * 0.3 * U.kpc,
        )
        source2 = SPHSource(
            distance=12 * U.Mpc,
            vpeculiar=-50 * U.km / U.s,
            rotation=Rotation.from_euler("z", (60 * U.deg).to_value(U.rad)),
            ra=50.05 * U.deg,
            dec=30.05 * U.deg,
            T_g=np.ones(n) * U.K,
            mHI_g=np.ones(n) * 1e7 * U.Msun,
            xyz_g=U.Quantity(
                [
                    np.zeros(n) * U.kpc,
                    r * np.cos(t),
                    r * np.sin(t),
                ]
            ),
            vxyz_g=U.Quantity(
                [
                    np.zeros(n) * U.km / U.s,
                    -v * np.sin(t),
                    v * np.cos(t),
                ]
            ),
            hsm_g=np.ones(n) * 0.3 * U.kpc,
        )
        source = CombinedSource([source1, source2])
        datacube = DataCube(
            n_px_x=128,
            n_px_y=128,
            n_channels=128,
            px_size=5 * U.arcsec,
            channel_width=3 * U.km / U.s,
            spectral_centre=760 * U.km / U.s,
            ra=50 * U.deg,
            dec=30 * U.deg,
        )
        beam = GaussianBeam(
            bmaj=15 * U.arcsec,
            bmin=15 * U.arcsec,
        )
        sph_kernel = GaussianKernel()
        spectral_model = GaussianSpectrum(sigma="thermal")
        m = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=None,
            sph_kernel=sph_kernel,
            spectral_model=spectral_model,
        )
        m.insert_source_in_cube()
        m.convolve_beam()
        # can write to file as usual
    """

    def __init__(self, sources: list[SPHSource]) -> None:
        self._distance = None
        self._ra = None
        self._dec = None
        self._vsys = None
        self._T_g = None
        self._mHI_g = None
        self._hsm_g = None
        self._coordinates_g = None
        self._skycoords = None
        self._spectralcoords = None
        self._pixcoords = None
        for source in sources:
            if not isinstance(source, SPHSource):
                raise ValueError(
                    "Pass a list of `SPHSource` (or derived class) objects as `sources`"
                    " argument."
                )
        self.coordinate_frame = sources[0].coordinate_frame
        for source in sources:
            if source.coordinate_frame != self.coordinate_frame:
                raise ValueError("All sources must have the same `coordinate_frame`.")
        self.sources = sources
        self.npart = np.sum([source.npart for source in self.sources])
        self.input_mass = np.sum(
            U.Quantity([source.input_mass for source in self.sources])
        )
        return

    def _init_skycoords(self, _reset: bool = True) -> None:
        # can save a bit of memory by copying these one at a time, I guess
        for source in self.sources:
            source._init_skycoords(_reset=_reset)
        # self.skycoords = np.concatenate([source.skycoords for source in self.sources])
        # for source in self.sources:
        #     del source.skycoords
        # self.spectralcoords = np.concatenate(
        #     [source.spectralcoords for source in self.sources]
        # )
        # for source in self.sources:
        #     del source.spectralcoords

    def _init_pixcoords(self, datacube: DataCube, origin: int = 0) -> None:
        for source in self.sources:
            source._init_pixcoords(datacube=datacube, origin=origin)
        # self.pixcoords = np.vstack([source.pixcoords for source in self.sources])

    @property
    def distance(self):
        if self._distance is None:
            self._distance = np.mean(  # not happy with this, but just used for messages
                U.Quantity([source.distance for source in self.sources])
            )
        return self._distance

    @property
    def ra(self):
        if self._ra is None:
            self._ra = np.mean(  # not happy with this, but just used for preview
                U.Quantity([source.ra for source in self.sources])
            )
        return self._ra

    @property
    def dec(self):
        if self._dec is None:
            self._dec = np.mean(  # not happy with this, but just used for preview
                U.Quantity([source.dec for source in self.sources])
            )
        return self._dec

    @property
    def vsys(self):
        if self._vsys is None:
            self._vsys = np.mean(  # not happy with this, but just used for preview
                U.Quantity([source.vsys for source in self.sources])
            )
        return self._vsys

    @property
    def T_g(self):
        if self._T_g is None:
            self._T_g = np.concatenate([source.T_g for source in self.sources])
        return self._T_g

    @property
    def mHI_g(self):
        if self._mHI_g is None:
            self._mHI_g = np.concatenate([source.mHI_g for source in self.sources])
        return self._mHI_g

    @property
    def hsm_g(self):
        if self._hsm_g is None:
            self._hsm_g = np.concatenate([source.hsm_g for source in self.sources])
        return self._hsm_g

    @property
    def coordinates_g(self):
        if self._coordinates_g is None:
            self._coordinates_g = np.concatenate(
                [source.coordinates_g for source in self.sources]
            )
        return self._coordinates_g

    @property
    def skycoords(self):
        if self._skycoords is None:
            self._skycoords = np.concatenate(
                [source.skycoords for source in self.sources]
            )
        return self._skycoords

    @property
    def spectralcoords(self):
        if self._spectralcoords is None:
            self._spectralcoords = np.concatenate(
                [source.spectralcoords for source in self.sources]
            )
        return self._spectralcoords

    @property
    def pixcoords(self):
        if self._pixcoords is None:
            self._pixcoords = np.hstack([source.pixcoords for source in self.sources])
        return self._pixcoords

    def apply_mask(self, mask: np.ndarray) -> None:
        count = 0
        # check that the mask is the right size here
        for source in self.sources:
            source.apply_mask(mask[count : count + source.npart])
        if self._T_g is not None:
            self._T_g = self._T_g[mask]
        if self._mHI_g is not None:
            self._mHI_g = self._mHI_g[mask]
        if self._hsm_g is not None:
            self._hsm_g = self._hsm_g[mask]
        if self._coordinates_g is not None:
            self._coordinates_g = self._coordinates_g[mask]
        if self._skycoords is not None:
            self._skycoords = self._skycoords[mask]
        if self._spectralcoords is not None:
            self._spectralcoords = self._spectralcoords[mask]
        if self._pixcoords is not None:
            self._pixcoords = self._pixcoords[:, mask]

    def rotate(
        self,
        rotation: Rotation | None = None,
        *,
        L_coords: L_coords | None = None,
    ) -> None:
        raise NotImplementedError("Rotate individual sources, not CombinedSource.")

    def translate(self, translation_vector: U.Quantity[U.kpc]) -> None:
        raise NotImplementedError("Translate individual sources, not CombinedSource.")

    def boost(self, boost_vector: U.Quantity[U.km / U.s]) -> None:
        raise NotImplementedError("Translate individual sources, not CombinedSource.")

    @property
    def curent_rotation(self) -> None:
        raise NotImplementedError(
            "Current rotation not available for a CombinedSource."
        )

    def save_current_rotation(self, fname: str) -> None:
        raise NotImplementedError("Cannot save rotation of CombinedSource.")

    def save_current_affine_transformations(self, fname: str) -> None:
        raise NotImplementedError(
            "Cannot save affine transformations of CombinedSource."
        )

    def load_affine_transformations(self, fname: str) -> None:
        raise NotImplementedError(
            "Cannot load affine transformations of CombinedSource."
        )

    # sph_kernel's _init_sm_lengths(source=self.source, ...) must work
    # martini's _prune_particles() must work
    # spectral_model's init_spectra(self.source) must work
    # must work with GlobalProfile
