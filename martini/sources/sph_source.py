import numpy as np
from astropy.coordinates import CartesianRepresentation, CartesianDifferential, ICRS
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as U
from ._L_align import L_align
from ._cartesian_translation import translate, translate_d

# Extend CartesianRepresentation to allow coordinate translation
setattr(CartesianRepresentation, "translate", translate)

# Extend CartesianDifferential to allow velocity (or other differential)
# translation
setattr(CartesianDifferential, "translate", translate_d)


class SPHSource(object):
    """
    Class abstracting HI emission sources consisting of SPH simulation
    particles.

    This class constructs an HI emission source from arrays of SPH particle
    properties: mass, smoothing length, temperature, position, and velocity.

    Parameters
    ----------
    distance : Quantity, with dimensions of length, optional
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: 3 Mpc.)

    vpeculiar : Quantity, with dimensions of velocity, optional
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: 0 km/s.)

    rotation : dict, optional
        Must have a single key, which must be one of `axis_angle`, `rotmat` or
        `L_coords`. Note that the 'y-z' plane will be the one eventually placed in the
        plane of the "sky". The corresponding value must be:

        - `axis_angle` : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element a Quantity with \
        dimensions of angle, indicating the angle to rotate through.
        - `rotmat` : A (3, 3) numpy.array specifying a rotation.
        - `L_coords` : A 2-tuple containing an inclination and an azimuthal \
        angle (both Quantity instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about \
        'y'). A 3-tuple may be provided instead, in which case the third \
        value specifies the position angle on the sky (rotation about 'x'). \
        The default position angle is 270 degrees.

        (Default: identity rotation matrix.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension for the source centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination for the source centroid. (Default: 0 deg.)

    h : float, optional
        Dimensionless hubble constant, H0 = h * 100 km / s / Mpc.
        (Default: 0.7)

    T_g : Quantity, with dimensions of temperature
        Particle temperature.

    mHI_g : Quantity, with dimensions of mass
        Particle HI mass.

    xyz_g : Quantity, with dimensions of length
        Particle position offset from source centroid. Note that the 'y-z'
        plane is that eventually placed in the plane of the "sky"; 'x' is
        the axis corresponding to the "line of sight".

    vxyz_g : Quantity, with dimensions of velocity
        Particle velocity offset from source centroid. Note that the 'y-z'
        plane is that eventually placed in the plane of the "sky"; 'x' is
        the axis corresponding to the "line of sight".

    hsm_g : Quantity, with dimensions of length
        Particle SPH smoothing lengths, defined as the FWHM of the smoothing kernel.
        Smoothing lengths are variously defined in the literature as the radius where
        the kernel amplitude reaches 0, or some rational fraction of this radius (and
        other definitions may well exist). The FWHM requested here is not a standard
        choice (with the exception of SWIFT snapshots!), but has the advantage of avoiding
        ambiguity in the definition.

    coordinate_axis: int, optional
        Rank of axis corresponding to position or velocity of a single
        particle. I.e. coordinate_axis=0 if shape is (3, N), or 1 if (N, 3).
        Usually prefer to omit this as it can be determined automatically.
        (Default: None.)
    """

    def __init__(
        self,
        distance=3.0 * U.Mpc,
        vpeculiar=0.0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        h=0.7,
        T_g=None,
        mHI_g=None,
        xyz_g=None,
        vxyz_g=None,
        hsm_g=None,
        coordinate_axis=None,
    ):
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
                "martini.sources.SPHSource: xyz_g and vxyz_g must"
                " have matching shapes."
            )

        if coordinate_axis == 0:
            xyz_g = xyz_g.T
            vxyz_g = vxyz_g.T
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
        self.current_rotation = np.eye(3)
        self.rotate(**rotation)
        self.skycoords = None
        self.pixcoords = None
        return

    def _init_skycoords(self, _reset=True):
        # _reset False only for unit testing
        direction_vector = np.array(
            [
                np.cos(self.ra) * np.cos(self.dec),
                np.sin(self.ra) * np.cos(self.dec),
                np.sin(self.dec),
            ]
        )
        distance_vector = direction_vector * self.distance
        vsys_vector = direction_vector * self.vsys
        self.rotate(axis_angle=("y", -self.dec))
        self.rotate(axis_angle=("z", self.ra))
        self.translate(distance_vector)
        self.boost(vsys_vector)
        self.skycoords = ICRS(self.coordinates_g, copy=True)
        # pixels indexed from 0 (not like in FITS!) for better use with numpy
        if _reset:
            self.boost(-vsys_vector)
            self.translate(-distance_vector)
            self.rotate(axis_angle=("z", -self.ra))
            self.rotate(axis_angle=("y", self.dec))
        return

    def _init_pixcoords(self, datacube, origin=0):
        self.pixcoords = (
            np.vstack(
                datacube.wcs.sub(3).wcs_world2pix(
                    self.skycoords.ra.to(datacube.units[0]),
                    self.skycoords.dec.to(datacube.units[1]),
                    self.skycoords.radial_velocity.to(datacube.units[2]),
                    origin,
                )
            )
            * U.pix
        )
        return

    def apply_mask(self, mask):
        """
        Remove particles from source arrays according to a mask.

        Parameters
        ----------
        mask : array_like, with boolean-like dtype
            Remove particles with indices corresponding to False values from
            the source arrays.
        """

        if mask.size != self.npart:
            raise ValueError("Mask must have same length as particle arrays.")
        mask_sum = np.sum(mask)
        if mask_sum == 0:
            raise RuntimeError("No source particles in target region.")
        self.npart = mask_sum
        if not self.T_g.isscalar:
            self.T_g = self.T_g[mask]
        if not self.mHI_g.isscalar:
            self.mHI_g = self.mHI_g[mask]
        self.coordinates_g = self.coordinates_g[mask]
        if self.skycoords is not None:
            self.skycoords = self.skycoords[mask]
        if self.pixcoords is not None:
            self.pixcoords = self.pixcoords[:, mask]
        if not self.hsm_g.isscalar:
            self.hsm_g = self.hsm_g[mask]
        return

    def rotate(self, axis_angle=None, rotmat=None, L_coords=None):
        """
        Rotate the source.

        The arguments correspond to different rotation types. Multiple types cannot be
        given in a single function call.

        Parameters
        ----------
        axis_angle : 2-tuple
            First element one of {'x', 'y', 'z'} for the axis to rotate about,
            second element a Quantity with dimensions of angle, indicating the
            angle to rotate through (right-handed rotation).
        rotmat : array_like with shape (3, 3)
            Rotation matrix.
        L_coords : 2-tuple or 3-tuple
            First element containing an inclination, second element an
            azimuthal angle (both Quantity instances with
            dimensions of angle). The routine will first attempt to identify
            a preferred plane based on the angular momenta of the central 1/3
            of particles in the source. This plane will then be rotated to lie
            in the 'y-z' plane, followed by a rotation by the azimuthal angle
            about its angular momentum pole (rotation about 'x'), and then
            inclined (rotation about 'y'). By default the position angle on the
            sky is 270 degrees, but if a third element is provided it sets the
            position angle (rotation about 'z').
        """

        args_given = (axis_angle is not None, rotmat is not None, L_coords is not None)
        if np.sum(args_given) == 0:
            # no-op
            return
        elif np.sum(args_given) > 1:
            raise ValueError("Multiple rotations in a single call not allowed.")

        do_rot = np.eye(3)

        if axis_angle is not None:
            # rotation_matrix gives left-handed rotation, so transpose for right-handed
            do_rot = rotation_matrix(axis_angle[1], axis=axis_angle[0]).T.dot(do_rot)

        if rotmat is not None:
            do_rot = rotmat.dot(do_rot)

        if L_coords is not None:
            if len(L_coords) == 2:
                incl, az_rot = L_coords
                pa = 270 * U.deg
            elif len(L_coords) == 3:
                incl, az_rot, pa = L_coords
            do_rot = L_align(
                self.coordinates_g.get_xyz(),
                self.coordinates_g.differentials["s"].get_d_xyz(),
                self.mHI_g,
                frac=0.3,
                Laxis="x",
            ).dot(do_rot)
            # rotation_matrix gives left-handed rotation, so transpose for right-handed
            do_rot = rotation_matrix(az_rot, axis="x").T.dot(do_rot)
            do_rot = rotation_matrix(incl, axis="y").T.dot(do_rot)
            if incl >= 0:
                do_rot = rotation_matrix(pa - 90 * U.deg, axis="x").T.dot(do_rot)
            else:
                do_rot = rotation_matrix(pa - 270 * U.deg, axis="x").T.dot(do_rot)

        self.current_rotation = do_rot.dot(self.current_rotation)
        self.coordinates_g = self.coordinates_g.transform(do_rot)
        return

    def translate(self, translation_vector):
        """
        Translate the source.

        Note that the "line of sight" is along the 'x' axis.

        Parameters
        ----------
        translation_vector : Quantity with shape (3, ), with dimensions of \
        length
            Vector by which to offset the source particle coordinates.
        """

        self.coordinates_g = self.coordinates_g.translate(translation_vector)
        return

    def boost(self, boost_vector):
        """
        Apply an offset to the source velocity.

        Note that the "line of sight" is along the 'x' axis.

        Parameters
        ----------
        translation_vector : Quantity with shape (3, ), with dimensions of \
        velocity
            Vector by which to offset the source particle velocities.
        """

        self.coordinates_g.differentials["s"] = self.coordinates_g.differentials[
            "s"
        ].translate(boost_vector)
        return

    def save_current_rotation(self, fname):
        """
        Output current rotation matrix to file.

        This includes the rotations applied for RA and Dec. The rotation matrix can be
        applied to astropy coordinates (e.g. a
        :class:`~astropy.coordinates.CartesianRepresentation`) as
        `coordinates.transform(np.loadtxt(fname))`.

        Parameters
        ----------
        fname : str, or file handle
            File in which to save rotation matrix.
        """

        np.savetxt(fname, self.current_rotation)
        return
