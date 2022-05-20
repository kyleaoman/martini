import numpy as np
from astropy.coordinates import CartesianRepresentation,\
    CartesianDifferential, ICRS
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as U
from ._L_align import L_align
from ._cartesian_translation import translate, translate_d

# Extend CartesianRepresentation to allow coordinate translation
setattr(CartesianRepresentation, 'translate', translate)

# Extend CartesianDifferential to allow velocity (or other differential)
# translation
setattr(CartesianDifferential, 'translate', translate_d)


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
        Keys may be any combination of `axis_angle`, `rotmat` and/or
        `L_coords`. These will be applied in this order. Note that the 'y-z'
        plane will be the one eventually placed in the plane of the "sky". The
        corresponding values:

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
        momentum pole (rotation about 'x'), and inclined (rotation about 'y').

        (Default: rotmat with the identity rotation.)

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
        Particle SPH smoothing lengths.

    coordinate_axis: int, optional
        Rank of axis corresponding to position or velocity of a single
        particle. I.e. coordinate_axis=0 if shape is (3, N), or 1 if (N, 3).
        Usually prefer to omit this as it can be determined automatically.
        (Default: None.)
    """

    def __init__(
            self,
            distance=3. * U.Mpc,
            vpeculiar=0. * U.km / U.s,
            rotation={'rotmat': np.eye(3)},
            ra=0.*U.deg,
            dec=0.*U.deg,
            h=.7,
            T_g=None,
            mHI_g=None,
            xyz_g=None,
            vxyz_g=None,
            hsm_g=None,
            coordinate_axis=None
    ):

        if coordinate_axis is None:
            if (xyz_g.shape[0] == 3) and (xyz_g.shape[1] != 3):
                coordinate_axis = 0
            elif (xyz_g.shape[0] != 3) and (xyz_g.shape[1] == 3):
                coordinate_axis = 1
            elif xyz_g.shape == (3, 3):
                raise RuntimeError("martini.sources.SPHSource: cannot guess "
                                   "coordinate_axis with shape (3, 3), provide"
                                   " explicitly.")
            else:
                raise RuntimeError("martini.sources.SPHSource: incorrect "
                                   "coordinate shape (not (3, N) or (N, 3)).")

        if xyz_g.shape != vxyz_g.shape:
            raise ValueError("martini.sources.SPHSource: xyz_g and vxyz_g must"
                             " have matching shapes.")
        self.h = h
        self.T_g = T_g
        self.mHI_g = mHI_g
        self.coordinates_g = CartesianRepresentation(
            xyz_g,
            xyz_axis=coordinate_axis,
            differentials={'s': CartesianDifferential(
                vxyz_g,
                xyz_axis=coordinate_axis
            )}
        )
        self.hsm_g = hsm_g

        self.npart = self.mHI_g.size

        self.ra = ra
        self.dec = dec
        self.distance = distance
        self.vpeculiar = vpeculiar
        self.rotation = rotation
        self.current_rotation = np.eye(3)
        self.rotate(**self.rotation)
        self.rotate(axis_angle=('y', self.dec))
        self.rotate(axis_angle=('z', -self.ra))
        direction_vector = np.array([
            np.cos(self.ra) * np.cos(self.dec),
            np.sin(self.ra) * np.cos(self.dec),
            np.sin(self.dec)
        ])
        distance_vector = direction_vector * self.distance
        self.translate_position(distance_vector)
        self.vsys = \
            (self.h * 100.0 * U.km * U.s ** -1 * U.Mpc ** - 1) * self.distance
        hubble_flow_vector = direction_vector * self.vsys
        vpeculiar_vector = direction_vector * self.vpeculiar
        self.translate_velocity(hubble_flow_vector + vpeculiar_vector)
        self.sky_coordinates = ICRS(self.coordinates_g)
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

        self.T_g = self.T_g[mask]
        self.mHI_g = self.mHI_g[mask]
        self.coordinates_g = self.coordinates_g[mask]
        self.sky_coordinates = ICRS(self.coordinates_g)
        self.hsm_g = self.hsm_g[mask]
        self.npart = np.sum(mask)
        if self.npart == 0:
            raise RuntimeError('No source particles in target region.')
        return

    def rotate(self, axis_angle=None, rotmat=None, L_coords=None):
        """
        Rotate the source.

        The arguments correspond to different rotation types. If supplied
        together in one function call, they are applied in order: axis_angle,
        then rotmat, then L_coords.

        Parameters
        ----------
        axis_angle : 2-tuple
            First element one of {'x', 'y', 'z'} for the axis to rotate about,
            second element a Quantity with dimensions of angle, indicating the
            angle to rotate through.
        rotmat : array_like with shape (3, 3)
            Rotation matrix.
        L_coords : 2-tuple
            First element containing an inclination and second element an
            azimuthal angle (both Quantity instances with
            dimensions of angle). The routine will first attempt to identify
            a preferred plane based on the angular momenta of the central 1/3
            of particles in the source. This plane will then be rotated to lie
            in the 'y-z' plane, followed by a rotation by the azimuthal angle
            about its angular momentum pole (rotation about 'x'), and finally
            inclined (rotation about 'y').
        """

        do_rot = np.eye(3)

        if axis_angle is not None:
            do_rot = rotation_matrix(
                axis_angle[1],
                axis=axis_angle[0]
            ).dot(do_rot)

        if rotmat is not None:
            do_rot = rotmat.dot(do_rot)

        if L_coords is not None:
            incl, az_rot = L_coords
            do_rot = L_align(self.coordinates_g.get_xyz(),
                             self.coordinates_g.differentials['s'].get_d_xyz(),
                             self.mHI_g, frac=.3, Laxis='x').dot(do_rot)
            do_rot = rotation_matrix(az_rot, axis='x').dot(do_rot)
            do_rot = rotation_matrix(incl, axis='y').dot(do_rot)

        self.current_rotation = do_rot.dot(self.current_rotation)
        self.coordinates_g = self.coordinates_g.transform(do_rot)
        return

    def translate_position(self, translation_vector):
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

    def translate_velocity(self, translation_vector):
        """
        Apply an offset to the source velocity.

        Note that the "line of sight" is along the 'x' axis.

        Parameters
        ----------
        translation_vector : Quantity with shape (3, ), with dimensions of \
        velocity
            Vector by which to offset the source particle velocities.
        """

        self.coordinates_g.differentials['s'] = \
            self.coordinates_g.differentials['s'].translate(translation_vector)
        return

    def save_current_rotation(self, fname):
        """
        Output current rotation matrix to file.

        Parameters
        ----------
        fname : str, or file handle
            File in which to save rotation matrix.
        """

        np.savetxt(fname, self.current_rotation)
        return
