from abc import ABCMeta, abstractmethod
from simobj import SimObj
import time
import numpy as np
from astropy.coordinates import CartesianRepresentation, CartesianDifferential, ICRS
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as U
from kyleaoman_utilities.L_align import L_align

#extend CartesianRepresentation to allow coordinate translation
def translate(cls, translation_vector):
    return CartesianRepresentation(cls.__class__.get_xyz(cls) + translation_vector.reshape(3, 1), differentials=cls.differentials)
setattr(CartesianRepresentation, 'translate', translate)
#and velocity (or generally, differential) translation
def translate_d(cls, translation_vector):
    return CartesianDifferential(cls.__class__.get_d_xyz(cls) + translation_vector.reshape(3, 1))
setattr(CartesianDifferential, 'translate', translate_d)

class _BaseSource():

    """
    Abstract base class for HI emission sources.

    Classes inheriting from _BaseSource must do the following in their __init__, before calling
    super().__init__:
     - Set self.h to the value of the dimensionless Hubble constant ("little h").
     - Set self.T_g to an astropy.units.Quantity array of particle temperatures, with dimensions of
       temperature.
     - Set self.mHI_g to an astropy.units.Quantity array of particle HI masses, with dimensions of
       mass.
     - Set self.coordinates_g to an astropy.coordinates.CartesianRepresentation containing particle 
       coordinates with units of length, with a differentials dict containing a key 's' and a 
       corresponding value holding an astropy.coordinates.CartesianRepresentation containing particle
       velocities, with dimensions of velocity. The coorinate centroid will be placed on the sky at
       the RA and Dec provided (see below), and the velocity centroid will be the reference velocity
       which eventually lands in the central channel of the data cube; these should be set
       accordingly.
     - Set self.hsml_g to an astropy.units.Quantity array of particle smoothing lengths, with
       dimensions of length.
    Then super().__init__ must be called.

    Parameters
    ----------
    distance : astropy.units.Quantity, with dimensions of length
        Source distance, also used to set the velocity offset via Hubble's law.
    
    rotation : dict
        Keys may be any combination of 'axis_angle', 'rotmat' and/or 'L_coords'. These will be applied
        in this order. Note that the 'y-z' plane will be the one eventually placed in the plane of
        the "sky". The corresponding values:
         - 'axis_angle' : 2-tuple, first element one of 'x', 'y', 'z' for the axis to rotate about,
           second element an astropy.units.Quantity with dimensions of angle, indicating the angle to
           rotate through.
         - 'rotmat' : A (3, 3) numpy.array specifying a rotation.
         - 'L_coords' : A 2-tuple containing an inclination and an azimuthal angle (both
           astropy.units.Quantity instances with dimensions of angle). The routine will first attempt
           to identify a preferred plane based on the angular momenta of the central 1/3 of particles
           in the source. This plane will then be rotated to lie in the plane of the "sky" ('y-z'), 
           rotated by the azimuthal angle about its angular momentum pole (rotation about 'x'), and 
           inclined (rotation about 'y'). Note that this process will effectively override axis-angle
           and rotation matrix rotations.

    ra : astropy.units.Quantity, with dimensions of angle
        Right ascension for the source centroid.

    dec : astropy.units.Quantity, with dimensions of angle
        Declination for the source centroid.

    See Also
    --------
    SingleParticleSource (simplest possible implementation of a class inheriting from _BaseSource).

    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, distance=3.*U.Mpc, rotation={'L_coords': (60.*U.deg, 0.*U.deg)}, ra=0.*U.deg, dec=0.*U.deg):

        self.npart = self.mHI_g.size

        self.ra = ra
        self.dec = dec
        self.distance = distance
        self.rotation = rotation
        self.current_rotation = np.eye(3)
        self.rotate(**self.rotation)
        self.rotate(axis_angle=('y', self.dec))
        self.rotate(axis_angle=('z', -self.ra))
        distance_vector = np.array([
            np.cos(self.ra) * np.cos(self.dec),
            np.sin(self.ra) * np.cos(self.dec),
            np.sin(self.dec)
        ]) * self.distance
        self.translate_position(distance_vector)
        self.vsys = (self.h * 100.0 * U.km * U.s ** -1 * U.Mpc ** - 1) * self.distance
        hubble_flow_vector = distance_vector * self.vsys / self.distance
        self.translate_velocity(hubble_flow_vector)
        self.sky_coordinates = ICRS(self.coordinates_g)
        return

    def apply_mask(self, mask):
        """
        Remove particles from source arrays according to a mask.
        
        Parameters
        ----------
        mask : array-like, containing boolean-like
            Remove particles with indices corresponding to False values from the source arrays.
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

        The arguments correspond to different rotation types. If supplied together in one function
        call, they are applied in order: axis_angle, then rotmat, then L_coords.
        
        Parameters
        ----------
        axis_angle : 2-tuple
            First element one of 'x', 'y', 'z' for the axis to rotate about, second element an 
            astropy.units.Quantity with dimensions of angle, indicating the angle to rotate through.
        rotmat : (3, 3) array-like
            Rotation matrix.
        L_coords : 2-tuple 
            First element containing an inclination and second element an azimuthal angle (both
            astropy.units.Quantity instances with dimensions of angle). The routine will first attempt
            to identify a preferred plane based on the angular momenta of the central 1/3 of particles
            in the source. This plane will then be rotated to lie in the 'y-z' plane, followed 
            by a rotation by the azimuthal angle about its angular momentum pole (rotation about 'x'),
            and finally inclined (rotation about 'y'). Note that this process will effectively 
            override axis_angle and rotmat arguments.
           """

        do_rot = np.eye(3)

        if axis_angle is not None:
            do_rot = rotation_matrix(axis_angle[1], axis=axis_angle[0]).dot(do_rot)
            
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
        translation_vector : astropy.units.Quantity, shape (3, ), with dimensions of length
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
        translation_vector : astropy.units.Quantity, shape (3, ), with dimensions of velocity
            Vector by which to offset the source particle velocities.
        """
        
        self.coordinates_g.differentials['s'] = self.coordinates_g.differentials['s'].translate(translation_vector)
        return

class SOSource(_BaseSource):
    
    def __init__(self, distance=3.*U.Mpc, rotation={'L_coords': (60.*U.deg, 0.*U.deg)}, SO_args=dict(), ra=0.*U.deg, dec=0.*U.deg):

        self._SO_args = SO_args
        while True:
            try:
                with SimObj(**self._SO_args) as SO:
                    self.h = SO.h
                    self.T_g = SO.T_g
                    self.mHI_g = SO.mHI_g
                    self.coordinates_g = CartesianRepresentation(
                        SO.xyz_g, 
                        xyz_axis=1,
                        differentials={'s': CartesianDifferential(SO.vxyz_g, xyz_axis=1)}
                    )
                    self.hsm_g = SO.hsm_g
                break
            except RuntimeError:
                time.sleep(10)
                continue
        
        super().__init__(distance=distance, rotation=rotation, ra=ra, dec=dec)
        return

class SingleParticleSource(_BaseSource):

    def __init__(self, distance=3.*U.Mpc, rotation={'rotmat': np.eye(3)}, ra=0.*U.deg, dec=0.*U.deg):
        self.h = .7
        self.T_g = np.ones(1) * 1.E4 * U.K
        self.mHI_g = np.ones(1) * 1.E4 * U.solMass
        self.coordinates_g = CartesianRepresentation(
            np.array([[1.E-6, 1.E-6, 1.E-6]]) * U.kpc,
            xyz_axis=1,
            differentials={'s': CartesianDifferential(
                np.array([[0., 0., 0.]]) * U.km * U.s ** -1,
                xyz_axis=1
            )}
        )
        self.hsm_g = np.ones(1) * 1. * U.kpc
        super().__init__(distance=distance, rotation=rotation, ra=ra, dec=dec)
        return

class CrossSource(_BaseSource):
    
    def __init__(self, distance=3.*U.Mpc, rotation={'rotmat': np.eye(3)}, ra=0.*U.deg, dec=0.*U.deg):
        self.h = .7
        self.T_g = np.ones(4) * 1.E4 * U.K
        self.mHI_g = np.ones(4) * 1.E4 * U.solMass
        self.coordinates_g = CartesianRepresentation(
            np.array([[0, 1, 0],
                      [0, 0, 2],
                      [0, -3, 0],
                      [0, 0, -4]]) * U.kpc,
            xyz_axis=1,
            differentials={'s':CartesianDifferential(
                np.array([[0, 0, 1],
                          [0, -1, 0],
                          [0, 0, -1],
                          [0, 1, 0]]) * U.km * U.s ** -1,
                xyz_axis=1
            )}
        )
        self.hsm_g = np.ones(4) * U.kpc
        super().__init__(distance=distance, rotation=rotation, ra=ra, dec=dec)
        return
