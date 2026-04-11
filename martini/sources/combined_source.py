"""
Provides a :class:`~martini.sources.combined_source.CombinedSource` class.

This can be used to join multiple individual sources together for use in a single mock
observation.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as U
from astropy.coordinates import SkyCoord, SpectralCoord, CartesianRepresentation
from martini.sources.sph_source import SPHSource
from ..datacube import HIfreq, DataCube
from ..L_coords import L_coords
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

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
    _T_g: U.Quantity[U.K] | None
    _mHI_g: U.Quantity[U.Msun] | None
    _hsm_g: U.Quantity[U.kpc] | None
    _coordinates_g: CartesianRepresentation | None
    _skycoords: SkyCoord | None
    _spectralcoords: SpectralCoord | None
    _pixcoords: U.Quantity[U.pix] | None

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

        This is just delegated to each of the combined sources.

        Parameters
        ----------
        _reset : bool
            If ``True``, return particles to their original positions. Setting to
            ``False`` is only intended for testing.
        """
        for source in self.sources:
            source._init_skycoords(_reset=_reset)

    def _init_pixcoords(self, datacube: DataCube, origin: int = 0) -> None:
        """
        Initialize pixel coordinates of the particles.

        This is just delegated to each of the combined sources.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            The DataCube (including its WCS) for which to calculate coordinates.

        origin : int
            Index of the first pixel in the WCS (FITS-style is 1, python-style is 0).
        """
        for source in self.sources:
            source._init_pixcoords(datacube=datacube, origin=origin)

    @property
    def distance(self) -> U.Quantity[U.Mpc]:
        """
        The approximate distance of the source.

        A :class:`~martini.sources.combined_source.CombinedSource` has no well-defined
        distance. This property estimates a distance as the mean distance of the
        combined sources. This has no influence on the output produced by :mod:`martini`,
        but can influence values that are printed in messages.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of length. The mean distance
            of the combined sources.
        """
        if self._distance is None:
            self._distance = np.mean(  # not happy with this, but just used for messages
                U.Quantity([source.distance for source in self.sources])
            )
        return self._distance

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
    def T_g(self) -> U.Quantity[U.K]:
        """
        Get the combined particle temperatures from all of the combined sources.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of temperature. Particle
            temperatures.
        """
        if self._T_g is None:
            self._T_g = np.concatenate([source.T_g for source in self.sources])
        return self._T_g

    @T_g.setter
    def T_g(self, value: U.Quantity[U.K]) -> None:
        """
        Set the particle temperatures.

        Parameters
        ----------
        value : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of temperature.
            The temperature values. The values are "unconcatenated" in the order of
            the ``self.sources`` and assigned to each, in addition to setting
            ``self._T_g``.
        """
        nparts = [s.npart for s in self.sources]
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for T_g.")
        for s, T_g_fragment in zip(
            self.sources, np.split(value, np.cumsum(nparts)[:-1])
        ):
            s.T_g = T_g_fragment
        self._T_g = value

    @property
    def mHI_g(self) -> U.Quantity[U.Msun]:
        """
        Get the combined particle HI masses from all of the combined sources.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of mass. Particle HI masses.
        """
        if self._mHI_g is None:
            self._mHI_g = np.concatenate([source.mHI_g for source in self.sources])
        return self._mHI_g

    @mHI_g.setter
    def mHI_g(self, value: U.Quantity[U.Msun]) -> None:
        """
        Set the particle HI masses.

        Parameters
        ----------
        value : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of mass.
            The HI mass values. The values are "unconcatenated" in the order of
            the ``self.sources`` and assigned to each, in addition to setting
            ``self._mHI_g``.
        """
        nparts = [s.npart for s in self.sources]
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for mHI_g.")
        for s, mHI_g_fragment in zip(
            self.sources, np.split(value, np.cumsum(nparts)[:-1])
        ):
            s.mHI_g = mHI_g_fragment
        self._mHI_g = value

    @property
    def hsm_g(self) -> U.Quantity[U.kpc]:
        """
        Get the combined particle smoothing lengths from all of the combined sources.

        Returns
        -------
        ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of length. Particle
            smoothing lengths.
        """
        if self._hsm_g is None:
            self._hsm_g = np.concatenate([source.hsm_g for source in self.sources])
        return self._hsm_g

    @hsm_g.setter
    def hsm_g(self, value: U.Quantity[U.kpc]) -> None:
        """
        Set the particle smoothing lengths.

        Parameters
        ----------
        value : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity`, with dimensions of length.
            The smoothing length values. The values are "unconcatenated" in the order of
            the ``self.sources`` and assigned to each, in addition to setting
            ``self._hsm_g``.
        """
        nparts = [s.npart for s in self.sources]
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for hsm_g.")
        for s, hsm_g_fragment in zip(
            self.sources, np.split(value, np.cumsum(nparts)[:-1])
        ):
            s.hsm_g = hsm_g_fragment
        self._hsm_g = value

    @property
    def coordinates_g(self) -> CartesianRepresentation:
        """
        Get the combined particle cartesian coordinates from all of the combined sources.

        Returns
        -------
        ~astropy.coordinates.CartesianRepresentation
            The combined :class:`~astropy.coordinates.CartesianRepresentation` object
            (including differentials).
        """
        if self._coordinates_g is None:
            self._coordinates_g = np.concatenate(
                [source.coordinates_g for source in self.sources]
            )
        return self._coordinates_g

    @coordinates_g.setter
    def coordinates_g(self, value: CartesianRepresentation) -> None:
        """
        Set the particle coordinates (including velocities).

        Parameters
        ----------
        value : ~astropy.coordinates.CartesianRepresentation
            The coordinates, including velocities as differentials. The values are
            "unconcatenated" in the order of the ``self.sources`` and assigned to each,
            in addition to setting ``self._coordinates_g``.
        """
        nparts = [s.npart for s in self.sources]
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for coordinates_g.")
        for i, s in enumerate(self.sources):
            cumul_count = np.r_[0, np.cumsum(nparts)]
            s.coordinates_g = value[cumul_count[i] : cumul_count[i + 1]]
        self._coordinates_g = value

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
            if any([s.skycoords is None for s in self.sources]):
                raise RuntimeError("Call _init_skycoords before accessing skycoords.")
            self._skycoords = np.concatenate(
                [source.skycoords for source in self.sources]
            )
        return self._skycoords

    @skycoords.setter
    def skycoords(self, value: SkyCoord) -> None:
        """
        Set the particle sky coordinates.

        Parameters
        ----------
        value : ~astropy.coordinates.SkyCoord
            The sky coordinates. The values are "unconcatenated" in the order of the
            ``self.sources`` and assigned to each, in addition to setting
            ``self._skycoords``.
        """
        nparts = [s.npart for s in self.sources]
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for skycoords.")
        for i, s in enumerate(self.sources):
            cumul_count = np.r_[0, np.cumsum(nparts)]
            s.skycoords = value[cumul_count[i] : cumul_count[i + 1]]
        self._skycoords = value

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
            if any([source.skycoords is None for source in self.sources]):
                raise RuntimeError(
                    "Call _init_skycoords before accessing spectralcoords."
                )
            # seems there's a bug in astropy: can't just concatenate SkyCoords
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
            self._spectralcoords = SpectralCoord(
                np.concatenate(
                    [
                        getattr(
                            source.skycoords, "radial_velocity", None
                        )  # placate mypy
                        for source in self.sources
                    ]
                ),
                doppler_convention="radio",
                doppler_rest=HIfreq,
                target=np.concatenate([source.skycoords for source in self.sources]),
                observer=origin_skycoord,
            )
        return self._spectralcoords

    @spectralcoords.setter
    def spectralcoords(self, value: SpectralCoord) -> None:
        """
        Set the particle spectral coordinates.

        Parameters
        ----------
        value : ~astropy.coordinates.SpectralCoord
            The spectral coordinates. The values are "unconcatenated" in the order of the
            ``self.sources`` and assigned to each, in addition to setting
            ``self._spectralcoords``.
        """
        nparts = [s.npart for s in self.sources]
        if value.size != self.npart:
            raise ValueError("Wrong number of elements for spectralcoords.")
        for i, s in enumerate(self.sources):
            cumul_count = np.r_[0, np.cumsum(nparts)]
            s.spectralcoords = value[cumul_count[i] : cumul_count[i + 1]]
        self._spectralcoords = value

    @property
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
            self._pixcoords = np.hstack([source.pixcoords for source in self.sources])
        return self._pixcoords

    def apply_mask(self, mask: np.ndarray) -> None:
        """
        Remove particles from source arrays according to a mask.

        Parameters
        ----------
        mask : ~numpy.ndarray
            Boolean mask. Remove particles with indices corresponding to ``False`` values
            from the source arrays.
        """
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
