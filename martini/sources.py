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
def translate_d(cls, translation_vector):
    return CartesianDifferential(cls.__class__.get_d_xyz(cls) + translation_vector.reshape(3, 1))
setattr(CartesianDifferential, 'translate', translate_d)

class _BaseSource():
    
    __metaclass__ = ABCMeta
    
    def __init__(self, distance=3.*U.Mpc, rotation={'L_coords': (60.*U.deg, 0.*U.deg)}):

        self.distance = distance
        self.rotation = rotation
        self.current_rotation = np.eye(3)
        self.rotate(**self.rotation)
        distance_vector = np.array([self.distance.value, 0, 0]) * self.distance.unit
        self.translate_position(distance_vector)
        self.vsys = (self.h * 100.0 * U.km * U.s ** -1 * U.Mpc ** - 1) * self.distance
        hubble_flow_vector = np.array([self.vsys.value, 0, 0]) * self.vsys.unit
        self.translate_velocity(hubble_flow_vector)
        self.sky_coordinates = ICRS(self.coordinates_g)
        return

    def apply_mask(self, mask):
        self.T_g = self.T_g[mask]
        self.mHI_g = self.mHI_g[mask]
        self.coordinates_g = self.coordinates_g[mask]
        self.sky_coordinates = ICRS(self.coordinates_g)
        self.hsm_g = self.hsm_g[mask]
        return

    def rotate(self, axis_angle=None, rotmat=None, L_coords=None):
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
        self.coordinates_g = self.coordinates_g.translate(translation_vector)
        return
        
    def translate_velocity(self, translation_vector):
        self.coordinates_g.differentials['s'] = self.coordinates_g.differentials['s'].translate(translation_vector)
        return

class SOSource(_BaseSource):
    
    def __init__(self, distance=3.*U.Mpc, rotation={'L_coords': (60.*U.deg, 0.*U.deg)}, SO_args=dict()):

        self._SO_args = SO_args
        while True:
            try:
                with SimObj(**self._SO_args) as SO:
                    self.h = SO.h
                    self.T_g = SO.T_g
                    self.mHI_g = SO.mHI_g
                    print(SO.xyz_g)
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

        super(SOSource, self).__init__(distance=distance, rotation=rotation)
        return

class SingleParticleSource(_BaseSource):

    def __init__(self, distance=3.*U.Mpc, rotation={'L_coords': (60.*U.deg, 0.*U.deg)}):
        self.h = 1. * U.kpc
        self.T_g = 1.E4 * U.K
        self.mHI_g = 1.E4 * U.solMass
        self.coordinates_g = CartesianRepresenation(
            np.array([[0., 0., 0.]]) * U.kpc
        )
        return
