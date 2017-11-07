from scipy.ndimage import convolve
import numpy as np
import astropy.units as U

class Martini():

    def __init__(self, source=None, datacube=None, beam=None, baselines=None, noise=None, sph_kernel_integral=None, spectral_model=None):
        self.source = source
        self.datacube = datacube
        self.beam = beam
        self.baselines = baselines
        self.noise = noise
        self.sph_kernel_integral = sph_kernel_integral
        self.spectral_model = spectral_model

        self.beam.init_kernel(self.datacube)
        if self.beam is not None:
            self.datacube.add_pad(self.beam.needs_pad())
        
        return

    def convolve_beam(self):
        unit = self.datacube._array.unit
        self.datacube._array = convolve(
            self.datacube._array, 
            self.beam.kernel,
            mode='constant',
            cval=0.0
        ) * unit
        self.datacube.drop_pad()
        return

    def add_noise(self):
        self.datacube._array = self.datacube._array + self.noise.f_noise()(self.datacube)
        return
    
    def insert_source_in_cube(self):
        origin = 0 #pixels indexed from 0 (not like in FITS!) for better use with numpy
        particle_coords = np.vstack(self.datacube.wcs.wcs_world2pix(
            self.source.sky_coordinates.ra, 
            self.source.sky_coordinates.dec,
            self.source.sky_coordinates.radial_velocity,
            origin)) * U.pix
        sm_length = np.arctan(self.source.hsm_g / self.source.sky_coordinates.distance).to(U.arcsec) \
                    / self.datacube.px_size * U.pix
        sm_range = np.ceil(sm_length).astype(int)
        
        #pixel iteration
        px_iter = np.nditer(self.datacube._array[...,0], flags=['multi_index', 'refs_ok'])
        while not px_iter.finished: #parallelize?
            ij = np.array(px_iter.multi_index).astype(np.int)[..., np.newaxis]
            particle_mask = (ij * U.pix - particle_coords[:2] <= sm_range).all(axis=0)
            
            weights = self.sph_kernel_integral(
                np.power(particle_coords[:2, particle_mask] - ij * U.pix, 2).sum(axis=0), 
                sm_length[particle_mask]
            )
            self.datacube._array[ij] = self.spectral_model(
                self.datacube, 
                self.source, 
                particle_mask, 
                weights
            )
            px_iter.iternext()

        return
