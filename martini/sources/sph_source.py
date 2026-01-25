"""
Provides the generic :class:`~martini.sources.sph_source.SPHSource` class.

Enables working with any SPH or similar simulation as input.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING
from astropy.coordinates import (
    CartesianRepresentation,
    CartesianDifferential,
    SphericalRepresentation,
    SkyCoord,
    SpectralCoord,
    ICRS,
)
import astropy.units as U
from ._L_align import L_align
from ._cartesian_translation import translate, translate_d
from ..datacube import HIfreq, DataCube
from ..L_coords import L_coords

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from astropy.coordinates.builtin_frames.baseradec import BaseRADecFrame

# Extend CartesianRepresentation to allow coordinate translation
setattr(CartesianRepresentation, "translate", translate)

# Extend CartesianDifferential to allow velocity (or other differential)
# translation
setattr(CartesianDifferential, "translate", translate_d)

_origin = CartesianRepresentation(
    np.zeros((3, 1)) * U.kpc,
    differentials={"s": CartesianDifferential(np.zeros((3, 1)) * U.km / U.s)},
)

# Affine transform matrices have mixed dimensionful and dimensionless elements so need to
# be processed as raw arrays. We define conventional units for length and velocity type
# affine transforms to ensure internal consistency.
_COORDINATE_TRANSFORM_UNITS = U.Mpc
_VELOCITY_TRANSFORM_UNITS = U.km / U.s


def apply_affine_transform(
    coords: U.Quantity, affine_transform: np.ndarray, transform_units: U.UnitBase
) -> U.Quantity:
    """
    Apply an affine coordinate transformation to a coordinate array.

    An arbitrary coordinate transformation mixing translations and rotations can be
    expressed as a 4x4 matrix. However, such a matrix has mixed units, so we need to
    assume a consistent unit for all transformations and work with bare arrays. We also
    always assume comoving coordinates.

    Parameters
    ----------
    coords : ~astropy.units.Quantity
        The coordinate array to be transformed.

    transform : ~numpy.ndarray
        The 4x4 affine transformation matrix.

    transform_units : ~astropy.units.UnitBase
        The units assumed in the translation portion of the transformation matrix.

    Returns
    -------
    ~astropy.units.Quantity
        The coordinate array with transformation applied.
    """
    return (
        U.Quantity(
            np.hstack(
                (
                    coords.to_value(transform_units),
                    np.ones(coords.shape[0])[:, np.newaxis],
                )
            ).dot(affine_transform)[:, :3],
            transform_units,
        )
        << coords.unit
    )


class SPHSource(object):
    r"""
    Class abstracting HI emission sources consisting of SPH simulation particles.

    This class constructs an HI emission source from arrays of SPH particle properties:
    mass, smoothing length, temperature, position, and velocity.

    Parameters
    ----------
    distance : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity along the direction to the source centre.

    rotation : ~scipy.spatial.transform.Rotation, optional
        A rotation to apply to the source particles, specified using the
        :class:`~scipy.spatial.transform.Rotation` class. That class supports many ways to
        specify a rotation (Euler angle, rotation matrices, quaternions, etc.). Refer to
        the :mod:`scipy` documentation for details. Note that the ``y-z`` plane will be
        the one eventually placed in the plane of the "sky". Cannot be used at the same
        time as ``L_coords``.

    L_coords : ~martini.L_coords.L_coords, optional
        A named tuple specifying 3 angles. Import it as ``from martini import L_coords``.
        The angles are used to orient the galaxy relative to its angular momentum vector,
        "L". The routine will first identify a preferred plane based on the angular
        momenta of the central 1/3 of HI gas. This plane will then be rotated to lie in
        the plane of the "sky" (``y-z`` plane), rotated by an angle ``az_rot`` around the
        angular momentum vector (rotation around ``x``), then inclined by ``incl`` towards
        or away from the line of sight (rotation around ``y``) and finally rotated on the
        sky to set the position angle ``pa`` (second rotation around ``x``). All rotations
        are extrinsic. The position angle refers to the receding side of the galaxy
        measured East of North. The angles should be specified using syntax like:
        ``L_coords=L_coords(incl=0 * U.deg, pa=270 * U.deg, az_rot=0 * U.deg)``. These
        example values are the defaults. Cannot be used at the same time as ``rotation``.

    ra : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Right ascension for the source centroid.

    dec : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Declination for the source centroid.

    h : float, optional
        Dimensionless hubble constant,
        :math:`H_0 = h (100\\,\\mathrm{km}\\,\\mathrm{s}^{-1}\\,\\mathrm{Mpc}^{-1})`.

    T_g : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of temperature.
        Particle temperature.

    mHI_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of mass.
        Particle HI mass.

    xyz_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity` , with dimensions of length.
        Particle position offset from source centroid. Note that the 'y-z'
        plane is that eventually placed in the plane of the "sky"; 'x' is
        the axis corresponding to the "line of sight".

    vxyz_g : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Particle velocity offset from source centroid. Note that the 'y-z'
        plane is that eventually placed in the plane of the "sky"; 'x' is
        the axis corresponding to the "line of sight".

    hsm_g : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Particle SPH smoothing lengths, defined as the FWHM of the smoothing kernel.
        Smoothing lengths are variously defined in the literature as the radius where
        the kernel amplitude reaches 0, or some rational fraction of this radius (and
        other definitions may well exist). The FWHM requested here is not a standard
        choice (with the exception of SWIFT snapshots!), but has the advantage of avoiding
        ambiguity in the definition.

    coordinate_axis : int, optional
        Rank of axis corresponding to position or velocity of a single
        particle. I.e. ``coordinate_axis=0`` if shape is (3, N), or ``1`` if (N, 3).
        Usually prefer to omit this as it can be determined automatically, but is
        ambiguous for sources with exactly 3 particles.

    coordinate_frame : ~astropy.coordinates.builtin_frames.baseradec.BaseRADecFrame
        Optional. The coordinate frame assumed in converting particle coordinates to RA
        and Dec, and for transforming coordinates and velocities to the data cube frame.
        The frame needs to have a well-defined velocity as well as spatial origin.
        Recommended frames are :class:`~astropy.coordinates.GCRS`,
        :class:`~astropy.coordinates.ICRS`, :class:`~astropy.coordinates.HCRS`,
        :class:`~astropy.coordinates.LSRK`, :class:`~astropy.coordinates.LSRD` or
        :class:`~astropy.coordinates.LSR`. The frame should be passed initialized, e.g.
        ``ICRS()`` (not just ``ICRS``).
    """

    h: float
    T_g: U.Quantity[U.K] | None
    mHI_g: U.Quantity[U.Msun]
    coordinates_g: U.Quantity[U.kpc]
    hsm_g: U.Quantity[U.kpc] | None
    npart: int
    ra: U.Quantity[U.deg]
    dec: U.Quantity[U.deg]
    distance: U.Quantity[U.Mpc]
    vpeculiar: U.Quantity[U.km / U.s]
    _coordinate_affine_transform: np.ndarray
    _velocity_affine_transform: np.ndarray
    vhubble: U.Quantity[U.km / U.s]
    vsys: U.Quantity[U.km / U.s]
    sky_coordinates: ICRS
    coordinate_frame: "BaseRADecFrame"
    pixcoords: U.Quantity[U.pix]
    input_mass: U.Quantity[U.Msun]
    skycoords: SkyCoord | None

    def __init__(
        self,
        *,
        distance: U.Quantity[U.Mpc],
        vpeculiar: U.Quantity[U.km / U.s] = 0.0 * U.km / U.s,
        rotation: Rotation | None = None,
        L_coords: L_coords | None = None,
        ra: U.Quantity[U.deg] = 0.0 * U.deg,
        dec: U.Quantity[U.deg] = 0.0 * U.deg,
        h: float = 0.7,
        T_g: U.Quantity[U.K] | None = None,
        mHI_g: U.Quantity[U.Msun],
        xyz_g: U.Quantity[U.kpc],
        vxyz_g: U.Quantity[U.km / U.s],
        hsm_g: U.Quantity[U.kpc] | None = None,
        coordinate_axis: int | None = None,
        coordinate_frame: "BaseRADecFrame" = ICRS(),
    ) -> None:
        if isinstance(rotation, dict):
            raise ValueError(
                "The method to specify rotations in martini has been updated. Replace:\n"
                "1) rotation={'rotmat': <rotation_matrix>}\n"
                "   with:\n"
                "   from scipy.spatial.transform import Rotation\n"
                "   rotation=Rotation.from_matrix(<rotation_matrix>)\n"
                "2) rotation={'axis_angle': (<axis>, <angle>)}\n"
                "   with:\n"
                "   rotation=Rotation.from_euler(<axis>, <angle>.to_value(U.rad))\n"
                "3) rotation={'L_coords': (<incl>, <az_rot>[, <pa>])}\n"
                "   with:\n"
                "   from martini import L_coords\n"
                "   L_coords=L_coords(incl=<incl>, az_rot=<az_rot>[, pa=<pa>])\n"
                "Refer to https://martini.readthedocs.io/en/stable/sources/index.html"
                "#rotation-and-translation and "
                "https://martini.readthedocs.io/en/stable/modules/"
                "martini.sources.sph_source.html#martini.sources.sph_source.SPHSource "
                "for further details."
            )

        if coordinate_axis is None:
            if (xyz_g.shape[0] == 3) and (xyz_g.shape[1] != 3):
                coordinate_axis = 0
            elif (xyz_g.shape[0] != 3) and (xyz_g.shape[1] == 3):
                coordinate_axis = 1
            elif xyz_g.shape == (3, 3):
                raise RuntimeError(
                    "martini.sources.SPHSource: cannot guess "
                    "coordinate_axis with shape (3, 3), provide"
                    " explicitly."
                )
            else:
                raise RuntimeError(
                    "martini.sources.SPHSource: incorrect "
                    "coordinate shape (not (3, N) or (N, 3))."
                )
        else:
            coordinate_axis = coordinate_axis

        if xyz_g.shape != vxyz_g.shape:
            raise ValueError(
                "martini.sources.SPHSource: xyz_g and vxyz_g must have matching shapes."
            )

        if coordinate_axis == 0:
            xyz_g = xyz_g.T
            vxyz_g = vxyz_g.T
        self.coordinate_frame = coordinate_frame
        self.h = h
        self.T_g = T_g
        self.mHI_g = mHI_g
        self.input_mass = mHI_g.sum()
        self.coordinates_g = CartesianRepresentation(
            xyz_g,
            xyz_axis=1,
            differentials={"s": CartesianDifferential(vxyz_g, xyz_axis=1)},
        )
        self.hsm_g = hsm_g

        self.npart = self.coordinates_g.size

        self.ra = ra
        self.dec = dec
        self.distance = distance
        self.vpeculiar = vpeculiar
        self.vhubble = (self.h * 100.0 * U.km * U.s**-1 * U.Mpc**-1) * self.distance
        self.vsys = self.vhubble + self.vpeculiar
        self._coordinate_affine_transform = np.eye(4)
        self._velocity_affine_transform = np.eye(4)
        self.rotate(rotation=rotation, L_coords=L_coords)  # complains if both not None
        self.skycoords = None
        self.spectralcoords = None
        self.pixcoords = None
        return

    def _init_skycoords(self, _reset: bool = True) -> None:
        """
        Initialize the sky coordinates of the particles.

        Parameters
        ----------
        _reset : bool
            If ``True``, return particles to their original positions. Setting to
            ``False`` is only intended for testing.
        """
        # _reset False only for unit testing
        distance_unit_vector = (
            SphericalRepresentation(self.ra, self.dec, 1)
            .represent_as(CartesianRepresentation)
            .xyz
        )
        distance_vector = distance_unit_vector * self.distance
        vpeculiar_vector = distance_unit_vector * self.vpeculiar

        self.rotate(Rotation.from_euler("y", -self.dec.to_value(U.rad)))
        self.rotate(Rotation.from_euler("z", self.ra.to_value(U.rad)))
        self.translate(distance_vector)
        # must be after translate:
        # \vec{v_H} = (100 h km/s/Mpc * D) * r^, but D * r^ is just \vec{r}:
        vhubble_vectors = (
            self.h * 100.0 * U.km * U.s**-1 * U.Mpc**-1
        ) * self.coordinates_g.xyz
        self.boost(vpeculiar_vector)
        # can't use boost for particle-by-particle velocities:
        self.coordinates_g.differentials["s"] = CartesianDifferential(
            self.coordinates_g.differentials["s"].d_xyz + vhubble_vectors
        )
        self.skycoords = SkyCoord(
            self.coordinates_g, frame=self.coordinate_frame, copy=True
        )
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
        self.spectralcoords = SpectralCoord(
            self.skycoords.radial_velocity,
            doppler_convention="radio",
            doppler_rest=HIfreq,
            target=self.skycoords,
            observer=origin_skycoord,
        )
        if _reset:
            self.coordinates_g.differentials["s"] = CartesianDifferential(
                self.coordinates_g.differentials["s"].d_xyz - vhubble_vectors
            )
            self.boost(-vpeculiar_vector)
            self.translate(-distance_vector)
            self.rotate(Rotation.from_euler("z", -self.ra.to_value(U.rad)))
            self.rotate(Rotation.from_euler("y", self.dec.to_value(U.rad)))
        return

    def _init_pixcoords(self, datacube: DataCube, origin: int = 0) -> None:
        """
        Initialize pixel coordinates of the particles.

        Parameters
        ----------
        datacube : ~martini.datacube.DataCube
            The DataCube (including its WCS) for which to calculate coordinates.

        origin : int
            Index of the first pixel in the WCS (FITS-style is 1, python-style is 0).
        """
        assert self.skycoords is not None, (
            "Initialize source.skycoords before calling _init_pixcoords."
        )
        assert self.spectralcoords is not None, (
            "Initialize source.spectralcoords before calling _init_pixcoords."
        )
        skycoords_df_frame = self.skycoords.transform_to(datacube.coordinate_frame)
        spectralcoords_df_specsys = (
            self.spectralcoords.with_observer_stationary_relative_to(datacube.specsys)
        )
        # pixels indexed from 0 (not like in FITS!) for better use with numpy
        self.pixcoords = (
            np.vstack(
                datacube.wcs.sub(3).wcs_world2pix(
                    skycoords_df_frame.ra.to(datacube.units[0]),
                    skycoords_df_frame.dec.to(datacube.units[1]),
                    spectralcoords_df_specsys.to(datacube.units[2]),
                    origin,
                )
            )
            * U.pix
        )
        return

    def apply_mask(self, mask: np.ndarray) -> None:
        """
        Remove particles from source arrays according to a mask.

        Parameters
        ----------
        mask : ~numpy.ndarray
            Boolean mask. Remove particles with indices corresponding to
            ``False`` values from the source arrays.
        """
        if mask.size != self.npart:
            raise ValueError("Mask must have same length as particle arrays.")
        mask_sum = np.sum(mask)
        if mask_sum == 0:
            raise RuntimeError("No non-zero mHI source particles in target region.")
        self.npart = mask_sum
        if self.T_g is not None and not self.T_g.isscalar:
            self.T_g = self.T_g[mask]
        if not self.mHI_g.isscalar:
            self.mHI_g = self.mHI_g[mask]
        self.coordinates_g = self.coordinates_g[mask]
        if self.skycoords is not None:
            self.skycoords = self.skycoords[mask]
        if self.spectralcoords is not None:
            self.spectralcoords = self.spectralcoords[mask]
        if self.pixcoords is not None:
            self.pixcoords = self.pixcoords[:, mask]
        if self.hsm_g is not None and not self.hsm_g.isscalar:
            self.hsm_g = self.hsm_g[mask]
        return

    def _append_to_coordinate_affine_transform(
        self, affine_transform: np.ndarray
    ) -> None:
        """
        Add a new transformation to the sequence for the spatial coordinates.

        The cumulative transformation is stored as a single 4x4 transformation matrix,
        so we update the current transformation using a dot product.

        Affine transform matrices contain a mix of dimensionless and dimensionful values
        so we need to process them as raw numpy arrays. By convention in :mod:`martini`
        the elements with (implicit) length dimensions are always Mpc.

        Parameters
        ----------
        affine_transform : :class:`~numpy.ndarray`
            The affine transform to add to the cumulative coordinate transformation.
        """
        self._coordinate_affine_transform = self._coordinate_affine_transform.dot(
            affine_transform
        )
        return

    def _append_to_velocity_affine_transform(
        self, affine_transform: np.ndarray
    ) -> None:
        """
        Add a new transformation to the sequence for the velocity coordinates.

        The cumulative transformation is stored as a single 4x4 transformation matrix,
        so we update the current transformation using a dot product.

        Affine transform matrices contain a mix of dimensionless and dimensionful values
        so we need to process them as raw numpy arrays. By convention in :mod:`martini`
        the elements with (implicit) velocity dimensions are always km/s.

        Parameters
        ----------
        affine_transform : :class:`~numpy.ndarray`
            The affine transform to add to the cumulative velocity transformation.
        """
        self._velocity_affine_transform = self._velocity_affine_transform.dot(
            affine_transform
        )
        return

    def rotate(
        self,
        rotation: Rotation | None = None,
        *,
        L_coords: L_coords | None = None,
    ) -> None:
        """
        Rotate the source.

        The arguments correspond to different rotation types. Multiple types cannot be
        given in a single function call.

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
        """
        args_given = (rotation is not None, L_coords is not None)
        if np.sum(args_given) == 0:
            # no-op
            return
        elif np.sum(args_given) > 1:
            raise ValueError("Multiple rotations in a single call not allowed.")

        if rotation is not None:
            do_rot = rotation.as_matrix()

        if L_coords is not None:
            do_rot = np.eye(3)
            do_rot = L_align(
                self.coordinates_g.get_xyz(),
                self.coordinates_g.differentials["s"].get_d_xyz(),
                self.mHI_g,
                frac=0.3,
                Laxis="x",
            ).dot(do_rot)
            do_rot = (
                Rotation.from_euler("x", L_coords.az_rot.to_value(U.rad))
                .as_matrix()
                .dot(do_rot)
            )
            do_rot = (
                Rotation.from_euler("y", L_coords.incl.to_value(U.rad))
                .as_matrix()
                .dot(do_rot)
            )
            do_rot = (
                Rotation.from_euler(
                    "x",
                    (
                        L_coords.pa - 90 * U.deg
                        if L_coords.incl >= 0
                        else L_coords.pa - 270 * U.deg
                    ).to_value(U.rad),
                )
                .as_matrix()
                .dot(do_rot)
            )

        affine_transform = np.eye(4)
        affine_transform[:3, :3] = do_rot
        self._append_to_coordinate_affine_transform(affine_transform)
        self._append_to_velocity_affine_transform(affine_transform)
        self.coordinates_g = self.coordinates_g.transform(do_rot)
        return

    def translate(self, translation_vector: U.Quantity[U.kpc]) -> None:
        """
        Translate the source.

        Note that the "line of sight" is along the 'x' axis.

        Parameters
        ----------
        translation_vector : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with shape (3, ), with dimensions of length.
            Vector by which to offset the source particle coordinates.
        """
        self.coordinates_g = self.coordinates_g.translate(translation_vector)
        affine_transform = np.eye(4)
        affine_transform[3, :3] = translation_vector.squeeze().to_value(
            _COORDINATE_TRANSFORM_UNITS
        )
        self._append_to_coordinate_affine_transform(affine_transform)
        return

    def boost(self, boost_vector: U.Quantity[U.km / U.s]) -> None:
        """
        Apply an offset to the source velocity.

        Note that the "line of sight" is along the 'x' axis.

        Parameters
        ----------
        boost_vector : ~astropy.units.Quantity
            :class:`~astropy.units.Quantity` with shape (3, ), with dimensions of
            velocity.
            Vector by which to offset the source particle velocities.
        """
        self.coordinates_g.differentials["s"] = self.coordinates_g.differentials[
            "s"
        ].translate(boost_vector)
        affine_transform = np.eye(4)
        affine_transform[3, :3] = boost_vector.squeeze().to_value(
            _VELOCITY_TRANSFORM_UNITS
        )
        self._append_to_velocity_affine_transform(affine_transform)
        return

    @property
    def current_rotation(self) -> np.ndarray:
        """
        Current rotation matrix of the source.

        Returns
        -------
        ~numpy.ndarray
            The rotation matrix taking the coordinate originally passed in to the source
            to the current orientation.
        """
        # The rotation part of the _coordinate_affine_transform and the
        # _velocity_affine_transform are identical, just pick one.
        return self._coordinate_affine_transform[:3, :3]

    def save_current_rotation(self, fname: str) -> None:
        """
        Output current rotation matrix to file.

        This includes the rotations applied for RA and Dec. The rotation matrix can be
        applied to astropy coordinates (e.g. a
        :class:`~astropy.coordinates.representation.cartesian.CartesianRepresentation`) as
        ``coordinates.transform(np.loadtxt(fname))``.

        Parameters
        ----------
        fname : str
            File in which to save rotation matrix. A file handle can also be passed.
        """
        np.savetxt(fname, self._coordinate_affine_transform[:3, :3])
        return

    def preview(
        self,
        max_points: int = 5000,
        fig: int = 1,
        lim: U.Quantity[U.kpc] = None,
        vlim: U.Quantity[U.km / U.s] = None,
        point_scaling: str = "auto",
        title: str = "",
        save: str | None = None,
    ) -> "Figure":
        """
        Produce a figure showing the source particle coordinates and velocities.

        Makes a 3-panel figure showing the projection of the source as it will appear in
        the mock observation. The first panel shows the particles in the y-z plane,
        coloured by the x-component of velocity (MARTINI projects the source along the
        x-axis). The second and third panels are position-velocity diagrams showing the
        x-component of velocity against the y and z coordinates, respectively.

        Parameters
        ----------
        max_points : int, optional
            Maximum number of points to draw per panel, the particles will be randomly
            subsampled if the source has more.

        fig : int, optional
            Number of the figure in matplotlib, it will be created as ``plt.figure(fig)``.

        lim : ~astropy.units.Quantity, optional
            :class:`~astropy.units.Quantity`, with dimensions of length.
            The coordinate axes extend from -lim to lim. If unspecified, the maximum
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

        # every Nth particle to plot at most max_points, or all particles
        lim = (
            max(
                np.max(np.abs(self.coordinates_g.y.to_value(U.kpc))),
                np.max(np.abs(self.coordinates_g.z.to_value(U.kpc))),
            )
            if lim is None
            else lim.to_value(U.kpc)
        )
        vlim = (
            np.max(
                np.abs(self.coordinates_g.differentials["s"].d_x.to_value(U.km / U.s))
            )
            if vlim is None
            else vlim.to_value(U.km / U.s)
        )
        cmask = np.logical_and.reduce(
            (
                np.abs(self.coordinates_g.y.to_value(U.kpc)) < lim,
                np.abs(self.coordinates_g.z.to_value(U.kpc)) < lim,
                np.abs(self.coordinates_g.differentials["s"].d_x.to_value(U.km / U.s))
                < vlim,
            )
        )
        nparts = cmask.sum()
        mask = np.arange(self.mHI_g.size)[cmask][:: max(nparts // max_points, 1)]
        hsm_factor = (
            1
            if (self.hsm_g is None or self.hsm_g.isscalar or mask.size <= 1)
            else (1 - (self.hsm_g[mask] / self.hsm_g[mask].max()) ** 0.1).to_value(
                U.dimensionless_unscaled
            )  # larger -> more transparent
            + (self.hsm_g[mask].min() == self.hsm_g[mask].max())
            * 0.5  # guard against getting all 0s
        )
        alpha = hsm_factor if point_scaling == "auto" else 1.0
        if self.hsm_g is None:
            size_scale = 1
        else:
            size_scale = (
                self.hsm_g.to_value(U.kpc) / lim
                if (self.hsm_g.isscalar or mask.size <= 1)
                else (self.hsm_g[mask].to_value(U.kpc) / lim)
            )
        size = 300 * size_scale if point_scaling == "auto" else 10
        figure = plt.figure(fig, figsize=(12, 4))
        figure.clf()
        figure.suptitle(title)

        # ----- MOMENT 1 -----
        sp1 = figure.add_subplot(1, 3, 1, aspect="equal")
        sp1.set_facecolor("#222222")
        scatter = sp1.scatter(
            self.coordinates_g.y[mask].to_value(U.kpc),
            self.coordinates_g.z[mask].to_value(U.kpc),
            c=self.coordinates_g.differentials["s"].d_x[mask].to_value(U.km / U.s),
            marker="o",
            cmap="coolwarm",
            edgecolor="None",
            s=size,
            alpha=alpha,
            vmin=-vlim,
            vmax=vlim,
            zorder=0,
        )
        sp1.plot([0], [0], marker="+", ls="None", mec="grey", ms=6, zorder=1)
        sp1.set_xlabel(r"$y\,[\mathrm{kpc}]$")
        sp1.set_ylabel(r"$z\,[\mathrm{kpc}]$")
        sp1.set_xlim((lim, -lim))
        sp1.set_ylim((-lim, lim))
        cb = figure.colorbar(mappable=scatter, ax=sp1, orientation="horizontal")
        cb.set_label(r"$v_x\,[\mathrm{km}\,\mathrm{s}^{-1}]$")

        # ----- PV Y -----
        sp2 = figure.add_subplot(1, 3, 2)
        sp2.set_facecolor("#222222")
        sp2.scatter(
            self.coordinates_g.y[mask].to_value(U.kpc),
            self.coordinates_g.differentials["s"].d_x[mask].to_value(U.km / U.s),
            c="white",
            edgecolors="None",
            marker="o",
            s=size,
            alpha=alpha,
            zorder=0,
        )
        sp2.plot([0], [0], marker="+", ls="None", mec="red", ms=6, zorder=1)
        sp2.set_xlim((lim, -lim))
        sp2.set_ylim((-vlim, vlim))
        sp2.set_xlabel(r"$y\,[\mathrm{kpc}]$")
        sp2.set_ylabel(r"$v_x\,[\mathrm{km}\,\mathrm{s}^{-1}]$")

        # ----- PV Z -----
        sp3 = figure.add_subplot(1, 3, 3)
        sp3.set_facecolor("#222222")
        sp3.scatter(
            self.coordinates_g.z[mask].to_value(U.kpc),
            self.coordinates_g.differentials["s"].d_x[mask].to_value(U.km / U.s),
            c="white",
            edgecolors="None",
            marker="o",
            s=size,
            alpha=alpha,
            zorder=0,
        )
        sp3.plot([0], [0], marker="+", ls="None", mec="red", ms=6, zorder=1)
        sp3.set_xlim((-lim, lim))
        sp3.set_ylim((-vlim, vlim))
        sp3.set_xlabel(r"$z\,[\mathrm{kpc}]$")
        sp3.set_ylabel(r"$v_x\,[\mathrm{km}\,\mathrm{s}^{-1}]$")

        figure.subplots_adjust(wspace=0.3)
        if save is not None:
            plt.savefig(save)
        return figure
