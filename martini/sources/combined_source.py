"""
Provides a :class:`~martini.sources.combined_source.CombinedSource` class.

This can be used to join multiple individual sources together for use in a single mock
observation.
"""

import numpy as np
from astropy import units as U
from astropy.coordinates import (
    SkyCoord,
    SpectralCoord,
    CartesianRepresentation,
    CartesianDifferential,
)
from martini.sources.sph_source import SPHSource, _index_spectralcoord
from ..datacube import DataCube
from .._util import HIfreq
from ..L_coords import L_coords
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

from importlib.metadata import version
from packaging.version import Version

ASTROPY_VERSION = Version(version("astropy"))
if ASTROPY_VERSION < Version("7.0.0"):  # pragma: no cover
    from astropy.coordinates import concatenate
else:
    concatenate = np.concatenate

if TYPE_CHECKING:
    from martplotlib.figure import Figure


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

    _distance: U.Quantity[U.Mpc] | None
    _ra: U.Quantity[U.deg] | None
    _dec: U.Quantity[U.deg] | None
    _vsys: U.Quantity[U.km / U.s] | None
    _skycoords: SkyCoord | None
    _spectralcoords: SpectralCoord | None
    _pixcoords: U.Quantity[U.pix] | None
    _los_distance_pixcoords: U.Quantity[U.pix] | None

    def __init__(self, sources: list[SPHSource]) -> None:
        self._distance = None
        self._ra = None
        self._dec = None
        self._vsys = None
        self._skycoords = None
        self._spectralcoords = None
        self._pixcoords = None
        self._los_distance_pixcoords = None
        for source in sources:
            if not isinstance(source, SPHSource):
                raise ValueError(
                    "Pass a list of `SPHSource` (or derived class) objects as `sources`"
                    " argument."
                )
            if isinstance(source, CombinedSource):
                raise ValueError(
                    "Cannot use `CombinedSource` to combine `CombinedSource`s."
                )
        if len(sources) == 0:
            raise ValueError("List of `sources` must contain at least one item.")
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
        """
        Initialize the sky coordinates of the particles.

        This is delegated to each of the combined sources, then the coordinates are
        combined. The combined coordinates are stored once, each source holds a view of
        its part of them.

        Parameters
        ----------
        _reset : bool
            If ``True``, return particles to their original positions. Setting to
            ``False`` is only intended for testing.
        """
        for source in self.sources:
            source._init_skycoords(_reset=_reset)
        skycoords = concatenate([source.skycoords for source in self.sources])
        self.skycoords = skycoords
        self.spectralcoords = self._combine_spectralcoords(skycoords)

    def _init_pixcoords(
        self,
        datacube: DataCube,
        origin: int = 0,
        n_px_z: int | None = None,
    ) -> None:
        """
        Initialize pixel coordinates of the particles.

        This is delegated to each of the combined sources, then the coordinates are
        combined. The combined coordinates are stored once, each source holds a view of
        its part of them.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            The DataCube (including its WCS) for which to calculate coordinates.

        origin : int
            Index of the first pixel in the WCS (FITS-style is 1, python-style is 0).

        n_px_z : int, optional
            Number of pixels along the line-of-sight distance axis. If given, particle
            locations along this axis are evaluated using the same (local) pixel scale as
            for RA and Dec and stored as ``self.los_distance_pixcoords``. The coordinate
            is logarithmic in distance, such that pixels are locally cubic.
        """
        for source in self.sources:
            source._init_pixcoords(
                datacube=datacube,
                origin=origin,
                n_px_z=n_px_z,
            )
        self.pixcoords = np.hstack([source.pixcoords for source in self.sources])
        self.los_distance_pixcoords = (
            None
            if n_px_z is None
            else np.concatenate(
                [
                    np.atleast_1d(source.los_distance_pixcoords)
                    for source in self.sources
                ]
            )
        )

    def _source_slices(self) -> list[slice]:
        """
        Get the slices of the combined particle arrays belonging to each source.

        Returns
        -------
        list
            One :class:`slice` per source, in the order of ``self.sources``.
        """
        edges = np.r_[0, np.cumsum([source.npart for source in self.sources])]
        return [slice(start, end) for start, end in zip(edges[:-1], edges[1:])]

    @property
    @U.quantity_input
    def distance(self) -> U.Quantity[U.Mpc]:
        """
        The approximate distance of the source.

        A :class:`~martini.sources.combined_source.CombinedSource` has no well-defined
        distance. This property estimates a distance as the HI mass-weighted mean distance
        of the combined sources. This has no influence on the output produced by
        :mod:`martini`, but can influence values that are printed in messages.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of length. The HI
            mass-weighted mean distance of the combined sources.
        """
        if self._distance is None:
            self._distance = np.average(
                U.Quantity([source.distance for source in self.sources]),
                weights=U.Quantity([source.mHI_g.sum() for source in self.sources]),
            )
        return self._distance

    @property
    @U.quantity_input
    def vsys(self) -> U.Quantity[U.km / U.s]:
        """
        The approximate systemic velocity of the source.

        A :class:`~martini.sources.combined_source.CombinedSource` has no well-defined
        systemic velocity. This property estimates a systemic velocity as the HI
        mass-weighted systemic velocity of the combined sources. This has no influence on
        the output produced by :mod:`martini`, but can influence values that are printed
        in messages.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of speed. The HI
            mass-weighted mean systemic velocity of the combined sources.
        """
        if self._vsys is None:
            self._vsys = np.average(
                U.Quantity([source.vsys for source in self.sources]),
                weights=U.Quantity([source.mHI_g.sum() for source in self.sources]),
            )
        return self._vsys

    def preview(
        self,
        max_points: int = 5000,
        fig: "int | Figure" = 1,
        lim: U.Quantity[U.deg] | None = None,
        vlim: U.Quantity[U.km / U.s] | None = None,
        point_scaling: str = "auto",
        title: str = "",
        save: str | None = None,
    ) -> "Figure":
        """
        Produce a figure showing the source particle coordinates and velocities.

        Makes a 3-panel figure showing the projection of the combined source as it will
        appear in the mock observation. The first panel shows the particles in the y-z
        plane, coloured by the x-component of velocity (MARTINI projects the source along
        the x-axis). The second and third panels are position-velocity diagrams showing
        the x-component of velocity against the y and z coordinates, respectively.

        Parameters
        ----------
        max_points : int, optional
            Maximum number of points to draw per panel, the particles will be randomly
            subsampled if the source has more.

        fig : int or ~matplotlib.figure.Figure, optional
            Number of the figure in matplotlib, it will be created as ``plt.figure(fig)``.
            Or, an existing figure can be provided.

        lim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity`, with dimensions of length.
            The coordinate axes extent from -lim to lim. If unspecified, the maximum
            absolute coordinate of particles in the source is used.

        vlim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity`, with dimensions of speed.
            The velocity axes and colour bar extend from ``-vlim`` to ``vlim``. If
            unspecified, the maximum absolute velocity of particles in the source is used.

        point_scaling : str, optional
            By default points are scaled in size and transparency according to their HI
            mass and the smoothing length (loosely proportional to their surface
            densities, but with different scaling to achieve a visually useful plot). For
            some sources the automatic scaling may not give useful results, using
            ``point_scaling="fixed"`` will plot points of constant size without opacity.

        title : str, optional
            A title for the figure can be provided.

        save : str, optional
            If provided, the figure is saved using ``plt.savefig(save)``. A ``.png`` or
            ``.pdf`` suffix is recommended.

        Returns
        -------
        ~matplotlib.figure.Figure
            The preview figure.
        """
        import matplotlib.pyplot as plt

        # guaranteed by __init__ to have at least one source.
        figure = self.sources[0].preview(
            max_points=max_points,
            fig=fig,
            lim=lim,
            vlim=vlim,
            point_scaling=point_scaling,
            title=title,
            save=None,
        )
        for source in self.sources[1:]:
            source.preview(
                max_points=max_points,
                fig=figure,
                lim=lim,
                vlim=vlim,
                point_scaling=point_scaling,
                title=title,
                save=None,
            )
        if save is not None:
            plt.savefig(save)
        return figure

    @property
    @U.quantity_input
    def T_g(self) -> U.Quantity[U.K]:
        """
        Get the combined particle temperatures from all of the combined sources.

        The combined temperatures are evaluated from the sources each time they are
        accessed.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of temperature. Particle
            temperatures.
        """
        return np.concatenate(
            [
                np.atleast_1d(source.T_g) if source.T_g is not None else None
                for source in self.sources
            ]
        )

    @T_g.setter
    @U.quantity_input
    def T_g(self, value: U.Quantity[U.K]) -> None:
        """
        Set the particle temperatures.

        Parameters
        ----------
        value : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of temperature.
            The temperature values. The values are "unconcatenated" in the order of
            the ``self.sources`` and assigned to each.
        """
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for T_g.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.T_g = value[source_slice]

    @property
    @U.quantity_input
    def mHI_g(self) -> U.Quantity[U.Msun]:
        """
        Get the combined particle HI masses from all of the combined sources.

        The combined HI masses are evaluated from the sources each time they are
        accessed.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of mass. Particle HI masses.
        """
        return np.concatenate([np.atleast_1d(source.mHI_g) for source in self.sources])

    @mHI_g.setter
    @U.quantity_input
    def mHI_g(self, value: U.Quantity[U.Msun]) -> None:
        """
        Set the particle HI masses.

        Parameters
        ----------
        value : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of mass.
            The HI mass values. The values are "unconcatenated" in the order of
            the ``self.sources`` and assigned to each.
        """
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for mHI_g.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.mHI_g = value[source_slice]

    @property
    @U.quantity_input
    def hsm_g(self) -> U.Quantity[U.kpc]:
        """
        Get the combined particle smoothing lengths from all of the combined sources.

        The combined smoothing lengths are evaluated from the sources each time they are
        accessed.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of length. Particle
            smoothing lengths.
        """
        return np.concatenate(
            [
                np.atleast_1d(source.hsm_g) if source.hsm_g is not None else None
                for source in self.sources
            ]
        )

    @hsm_g.setter
    @U.quantity_input
    def hsm_g(self, value: U.Quantity[U.kpc]) -> None:
        """
        Set the particle smoothing lengths.

        Parameters
        ----------
        value : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of length.
            The smoothing length values. The values are "unconcatenated" in the order of
            the ``self.sources`` and assigned to each.
        """
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for hsm_g.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.hsm_g = value[source_slice]

    @property
    def coordinates_g(self) -> CartesianRepresentation:
        """
        Get the combined particle cartesian coordinates from all of the combined sources.

        The combined coordinates are evaluated from the sources each time they are
        accessed.

        Returns
        -------
        ~astropy.coordinates.CartesianRepresentation
            The combined :class:`~astropy.coordinates.CartesianRepresentation` object
            (including differentials).
        """
        if ASTROPY_VERSION < Version("7.0.0"):  # pragma: no cover
            # Neither np.concatenate nor astropy.coordinates.concatenate can handle
            # this, despite what astropy.coordinates.concatenate docs claim. Do it
            # "by hand".
            return CartesianRepresentation(
                np.hstack([source.coordinates_g.xyz for source in self.sources]),
                differentials={
                    "s": CartesianDifferential(
                        np.hstack(
                            [
                                source.coordinates_g.differentials["s"].d_xyz
                                for source in self.sources
                            ]
                        ),
                    )
                },
            )
        return np.concatenate([source.coordinates_g for source in self.sources])

    @coordinates_g.setter
    def coordinates_g(self, value: CartesianRepresentation) -> None:
        """
        Set the particle coordinates (including velocities).

        Parameters
        ----------
        value : ~astropy.coordinates.CartesianRepresentation
            The coordinates, including velocities as differentials. The values are
            "unconcatenated" in the order of the ``self.sources`` and assigned to each.
        """
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for coordinates_g.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.coordinates_g = value[source_slice]

    @property
    def skycoords(self) -> SkyCoord:
        """
        Get the combined sky coordinates from all of the combined sources.

        Returns
        -------
        ~astropy.coordinates.SkyCoord
            The combined :class:`~astropy.coordinates.SkyCoord` object.
        """
        if self._skycoords is None:
            raise RuntimeError("Call _init_skycoords before accessing skycoords.")
        return self._skycoords

    @skycoords.setter
    def skycoords(self, value: SkyCoord) -> None:
        """
        Set the particle sky coordinates.

        Parameters
        ----------
        value : ~astropy.coordinates.SkyCoord
            The sky coordinates. Each of the ``self.sources`` is assigned a view of its
            part of the values, in order, in addition to setting ``self._skycoords``.
        """
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for skycoords.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.skycoords = value[source_slice]
        self._skycoords = value

    def _combine_spectralcoords(self, skycoords: SkyCoord) -> SpectralCoord:
        """
        Combine the spectral coordinates of the combined sources.

        Parameters
        ----------
        skycoords : ~astropy.coordinates.SkyCoord
            The combined sky coordinates of the sources.

        Returns
        -------
        ~astropy.coordinates.SpectralCoord
            The combined :class:`~astropy.coordinates.SpectralCoord` object.
        """
        # Seems there's a bug in astropy: can't concatenate SkyCoords with either
        # astropy.coordinates.concatenate nor np.concatenate, so do it "manually".
        # https://github.com/astropy/astropy/issues/19560
        origin_skycoord = SkyCoord(
            x=0 * U.kpc,
            y=0 * U.kpc,
            z=0 * U.kpc,
            v_x=0 * U.km / U.s,
            v_y=0 * U.km / U.s,
            v_z=0 * U.km / U.s,
            representation_type="cartesian",
            differential_type="cartesian",
            frame=self.coordinate_frame,
        )
        return SpectralCoord(
            np.concatenate(
                [
                    getattr(source.skycoords, "radial_velocity", None)  # placate mypy
                    for source in self.sources
                ]
            ),
            doppler_convention="radio",
            doppler_rest=HIfreq,
            target=skycoords,
            observer=origin_skycoord,
        )

    @property
    def spectralcoords(self) -> SpectralCoord:
        """
        Get the combined spectral coordinates from all of the combined sources.

        Returns
        -------
        ~astropy.coordinates.SpectralCoord
            The combined :class:`~astropy.coordinates.SpectralCoord` object.
        """
        if self._spectralcoords is None:
            raise RuntimeError("Call _init_skycoords before accessing spectralcoords.")
        return self._spectralcoords

    @spectralcoords.setter
    def spectralcoords(self, value: SpectralCoord) -> None:
        """
        Set the particle spectral coordinates.

        Parameters
        ----------
        value : ~astropy.coordinates.SpectralCoord
            The spectral coordinates. Each of the ``self.sources`` is assigned a view of
            its part of the values, in order, in addition to setting
            ``self._spectralcoords``.
        """
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for spectralcoords.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.spectralcoords = _index_spectralcoord(value, source_slice)
        self._spectralcoords = value

    @property
    @U.quantity_input
    def pixcoords(self) -> U.Quantity[U.pix]:
        """
        Get the combined pixel coordinates from all of the combined sources.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels. 2D coordinates
            of particles in pixel coordinates.
        """
        if self._pixcoords is None:
            raise RuntimeError("Call _init_pixcoords before accessing pixcoords.")
        return self._pixcoords

    @pixcoords.setter
    @U.quantity_input
    def pixcoords(self, value: U.Quantity[U.pix]) -> None:
        """
        Set the particle pixel coordinates.

        Parameters
        ----------
        value : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of pixels. The pixel
            coordinates, with particles along the second axis. Each of the
            ``self.sources`` is assigned a view of its part of the values, in order, in
            addition to setting ``self._pixcoords``.
        """
        if value.shape[1] != self.npart:
            raise ValueError("Wrong number of elements for pixcoords.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.pixcoords = value[:, source_slice]
        self._pixcoords = value

    @property
    def los_distance_pixcoords(self) -> U.Quantity[U.pix] | None:
        """
        Get the combined pixel-scale distance offsets from all of the combined sources.

        Returns
        -------
        ~astropy.units.Quantity or None
            :class:`~astropy.units.Quantity`, with dimensions of pixels. LoS distance
            offsets from source distance in pixel units. ``None`` if these have not been
            initialized for the combined sources.
        """
        return self._los_distance_pixcoords

    @los_distance_pixcoords.setter
    def los_distance_pixcoords(self, value: U.Quantity[U.pix] | None) -> None:
        """
        Set the particle pixel-scale distance offsets.

        Parameters
        ----------
        value : ~astropy.units.Quantity or None
            :class:`~astropy.units.Quantity`, with dimensions of pixels. LoS distance
            offsets from source distance in pixel units. Each of the ``self.sources`` is
            assigned a view of its part of the values, in order, in addition to setting
            ``self._los_distance_pixcoords``. If ``None``, all are set to ``None``.
        """
        if value is not None and value.size != self.npart:
            raise ValueError("Wrong number of elements for los_distance_pixcoords.")
        for s, source_slice in zip(self.sources, self._source_slices()):
            s.los_distance_pixcoords = None if value is None else value[source_slice]
        self._los_distance_pixcoords = value

    def apply_mask(self, mask: np.ndarray) -> None:
        """
        Remove particles from source arrays according to a mask.

        Parameters
        ----------
        mask : ~numpy.ndarray
            Boolean mask. Remove particles with indices corresponding to ``False`` values
            from the source arrays.
        """
        self.npart = mask.sum()
        count = 0
        # check that the mask is the right size here
        for source in self.sources:
            source_npart = source.npart  # changes when mask is applied
            source.apply_mask(mask[count : count + source_npart])
            count += source_npart
        # mask the combined coordinates and give the sources views of them again:
        if self._skycoords is not None:
            self.skycoords = self._skycoords[mask]
        if self._spectralcoords is not None:
            self.spectralcoords = _index_spectralcoord(self._spectralcoords, mask)
        if self._pixcoords is not None:
            self.pixcoords = self._pixcoords[:, mask]
        if self._los_distance_pixcoords is not None:
            self.los_distance_pixcoords = self._los_distance_pixcoords[mask]

    def rotate(
        self,
        rotation: Rotation | None = None,
        *,
        L_coords: L_coords | None = None,
    ) -> Rotation:
        """
        Rotate the source.

        Not available for :class:`~martini.sources.combined_source.CombinedSource`,
        rotate individual sources instead.

        Parameters
        ----------
        rotation : ~scipy.spatial.transform.Rotation, optional
            A :class:`~scipy.spatial.transform.Rotation` specifying the rotation. This
            type of object can be initialized from many ways of specifying rotations:
            rotation matrices, Euler angles, quaternions, etc. Refer to the :mod:`scipy`
            documentation for details.
        L_coords : ~martini.L_coords.L_coords, optional
            First element containing an inclination, second element an
            azimuthal angle (both :class:`~astropy.units.Quantity` instances with
            dimensions of angle). The routine will first attempt to identify
            a preferred plane based on the angular momenta of the central 1/3
            of particles in the source. This plane will then be rotated to lie
            in the 'y-z' plane, followed by a rotation by the azimuthal angle
            about its angular momentum pole (rotation about 'x'), and then
            inclined (rotation about 'y'). By default the position angle on the
            sky is 270 degrees, but if a third element is provided it sets the
            position angle (second rotation about 'x').

        Raises
        ------
        NotImplementedError
            Always raises.
        """
        raise NotImplementedError("Rotate individual sources, not CombinedSource.")

    @U.quantity_input
    def translate(self, translation_vector: U.Quantity[U.kpc]) -> None:
        """
        Translate the source.

        Not available for :class:`~martini.sources.combined_source.CombinedSource`,
        translate individual sources instead.

        Parameters
        ----------
        translation_vector : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with shape (3, ), with dimensions of length.
            Vector by which to offset the source particle coordinates.

        Raises
        ------
        NotImplementedError
            Always raises.
        """
        raise NotImplementedError("Translate individual sources, not CombinedSource.")

    @U.quantity_input
    def boost(self, boost_vector: U.Quantity[U.km / U.s]) -> None:
        """
        Apply an offset to the source velocity.

        Not available for :class:`~martini.sources.combined_source.CombinedSource`,
        boost individual sources instead.

        Parameters
        ----------
        boost_vector : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with shape (3, ), with dimensions of
            velocity.
            Vector by which to offset the source particle velocities.

        Raises
        ------
        NotImplementedError
            Always raises.
        """
        raise NotImplementedError("Translate individual sources, not CombinedSource.")

    @property
    def current_rotation(self) -> np.ndarray:
        """
        Current rotation matrix of the source.

        Not available for :class:`~martini.sources.combined_source.CombinedSource`.

        Raises
        ------
        NotImplementedError
            Always raises.
        """
        raise NotImplementedError(
            "Current rotation not available for a CombinedSource."
        )

    def save_current_rotation(self, fname: str) -> None:
        """
        Output current rotation matrix to file.

        Parameters
        ----------
        fname : str
            File in which to save rotation matrix (as a text file).

        Raises
        ------
        NotImplementedError
            Always raises.
        """
        raise NotImplementedError("Cannot save rotation of CombinedSource.")

    def save_current_affine_transformations(self, fname: str) -> None:
        """
        Output current affine transformation matrices (position and velocity) to a file.

        This is not supported for
        :class:`~martini.sources.combined_source.CombinedSource`.

        Parameters
        ----------
        fname : str
            File in which to save affine transformation matrices (in ``*.npy`` format).

        Raises
        ------
        NotImplementedError
            Always raises.
        """
        raise NotImplementedError(
            "Cannot save affine transformations of CombinedSource."
        )

    def load_affine_transformations(self, fname: str) -> None:
        """
        Load a set of affine transformation matrices (position and velocity) from a file.

        This is not supported for
        :class:`~martini.sources.combined_source.CombinedSource`.

        Parameters
        ----------
        fname : str
            File from which to load affine transformation matrices (in ``*.npy`` format).

        Raises
        ------
        NotImplementedError
            Always raises.
        """
        raise NotImplementedError(
            "Cannot load affine transformations of CombinedSource."
        )
