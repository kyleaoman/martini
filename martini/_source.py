from simobj import SimObj
import time
import numpy as np
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.matrix_utilities import rotation_matrix
from kyleaoman_utilities.L_align import L_align

#extend CartesianRepresentation to allow coordinate translation
def translate(cls, translation_vector):
    return CartesianRepresentation(cls.__class__.get_xyz(cls) + translation_vector.reshape(3, 1), differentials=cls.differentials)
setattr(CartesianRepresentation, 'translate', translate)
def translate_d(cls, translation_vector):
    return CartesianDifferential(cls.__class__.get_d_xyz(cls) + translation_vector.reshape(3, 1))
setattr(CartesianDifferential, 'translate', translate_d)

import numpy as np
import astropy.units as U

class Source():
    
    def __init__(self, distance=3. * U.Mpc, SO_args=dict()):

        self.distance = distance
        self._SO_args = SO_args
        while True:
            try:
                with SimObj(**self._SO_args) as SO:
                    self.mHI_g = SO.mHI_g
                    self.coordinates_g = CartesianRepresentation(
                        SO.xyz_g, 
                        xyz_axis=1,
                        differentials={'s': CartesianDifferential(SO.vxyz_g, xyz_axis=1)}
                    )
                    self.hsm_g = SO.hsm_g
                break
            except RuntimeError:
                print('Waiting on lock release...')
                time.sleep(10)
                continue

        self.current_rotation = np.eye(3)


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
                             self.mHI_g, frac=.3, Laxis='y').dot(do_rot)
            do_rot = rotation_matrix(az_rot, axis='y').dot(do_rot)
            do_rot = rotation_matrix(incl, axis='x').dot(do_rot)
            
        self.current_rotation = do_rot.dot(self.current_rotation)
        self.coordinates_g = self.coordinates_g.transform(do_rot)
        return
                
    def translate_position(self, translation_vector):
        self.coordinates_g = self.coordinates_g.translate(translation_vector)
        return
        
    def translate_velocity(self, translation_vector):
        self.coordinates_g.differentials['s'] = self.coordinates_g.differentials['s'].translate(translation_vector)
        return
