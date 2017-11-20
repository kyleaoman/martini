from scipy.ndimage import convolve
import numpy as np
import astropy.units as U
from astropy.io import fits
from astropy import __version__ as astropy_version
from datetime import datetime
from itertools import product
from multiprocessing import Pool

class Martini():

    def __init__(self, source=None, datacube=None, beam=None, baselines=None, noise=None, sph_kernel=None, spectral_model=None):
        self.source = source
        self.datacube = datacube
        self.beam = beam
        self.baselines = baselines
        self.noise = noise
        self.sph_kernel = sph_kernel
        self.spectral_model = spectral_model

        if self.beam is not None:
            self.beam.init_kernel(self.datacube)
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
        self.datacube._array = self.datacube._array.to(U.Jy * U.beam ** -1, equivalencies=self.beam.px_to_beam)
        return

    def add_noise(self):
        self.datacube._array = self.datacube._array + self.noise.generate(self.datacube)
        return
    
    def insert_source_in_cube(self):
        origin = 0 #pixels indexed from 0 (not like in FITS!) for better use with numpy
        particle_coords = np.vstack(self.datacube.wcs.sub(3).wcs_world2pix(
            self.source.sky_coordinates.ra, 
            self.source.sky_coordinates.dec,
            self.source.sky_coordinates.radial_velocity,
            origin)) * U.pix
        sm_length = np.arctan(
            self.source.hsm_g / self.source.sky_coordinates.distance
        ).to(U.pix, U.pixel_scale(self.datacube.px_size / U.pix))
        sm_range = np.ceil(sm_length).astype(int)
        
        
        #pixel iteration   
        ij_pxs = list(product(
            np.arange(self.datacube._array.shape[0]), 
            np.arange(self.datacube._array.shape[1])
        ))
        for ij_px in ij_pxs:
            print('start ', time.clock())
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            print('masking', time.clock())
            particle_mask = (ij - particle_coords[:2] <= sm_range).all(axis=0)
            print('weights', time.clock())
            weights = self.sph_kernel.line_integral(
                np.power(particle_coords[:2, particle_mask] - ij, 2).sum(axis=0), 
                sm_length[particle_mask]
            )
            print('spectrum', time.clock())
            spectrum = self.spectral_model.spectrum(
                self.source,
                self.datacube.channel_edges,
                particle_mask,
                weights
            )
            print('insert', time.clock())
            self.datacube._array[ij_px[0], ij_px[1], :, 0] = spectrum
            print('end', time.clock())
            exit()
        
        return

    def write_fits(self, filename, channels='frequency'):
        
        self.datacube.drop_pad()
        if channels == 'frequency':
            self.datacube.freq_channels()
        elif channels == 'velocity':
            pass
        else:
            raise ValueError("Martini.write_fits: Unknown 'channels' value "
                             "(use 'frequency' or 'velocity'.")

        filename = filename if filename[-5:] == '.fits' else filename + '.fits'

        wcs_header = self.datacube.wcs.to_header()
        wcs_header.rename_keyword('WCSAXES', 'NAXIS')

        header = fits.Header()
        header.append(('SIMPLE', 'T'))
        header.append(('BITPIX', 16))
        header.append(('NAXIS', wcs_header['NAXIS']))
        header.append(('NAXIS1', self.datacube.n_px_x))
        header.append(('NAXIS2', self.datacube.n_px_y))
        header.append(('NAXIS3', self.datacube.n_channels))
        header.append(('NAXIS4', 1))
        header.append(('BLOCKED', 'T'))
        header.append(('CDELT1', wcs_header['CDELT1']))
        header.append(('CRPIX1', wcs_header['CRPIX1']))
        header.append(('CRVAL1', wcs_header['CRVAL1']))
        header.append(('CTYPE1', wcs_header['CTYPE1']))
        header.append(('CUNIT1', wcs_header['CUNIT1']))
        header.append(('CDELT2', wcs_header['CDELT2']))
        header.append(('CRPIX2', wcs_header['CRPIX2']))
        header.append(('CRVAL2', wcs_header['CRVAL2']))
        header.append(('CTYPE2', wcs_header['CTYPE2']))
        header.append(('CUNIT2', wcs_header['CUNIT2']))
        header.append(('CDELT3', wcs_header['CDELT3']))
        header.append(('CRPIX3', wcs_header['CRPIX3']))
        header.append(('CRVAL3', wcs_header['CRVAL3']))
        header.append(('CTYPE3', wcs_header['CTYPE3']))
        header.append(('CUNIT3', wcs_header['CUNIT3']))
        header.append(('CDELT4', wcs_header['CDELT4']))
        header.append(('CRPIX4', wcs_header['CRPIX4']))
        header.append(('CRVAL4', wcs_header['CRVAL4']))
        header.append(('CTYPE4', wcs_header['CTYPE4']))
        header.append(('CUNIT4', 'PAR'))
        header.append(('EPOCH', 2000))
        header.append(('INSTRUME', 'MARTINI'))
        #header.append(('BLANK', -32768)) #only for integer data
        header.append(('BSCALE', 1.0))
        header.append(('BZERO', 0.0))
        header.append(('DATAMAX', np.max(self.datacube._array.value)))
        header.append(('DATAMIN', np.min(self.datacube._array.value)))
        header.append(('ORIGIN', 'astropy v'+astropy_version))
        header.append(('OBJECT', 'MOCK'))
        header.append(('BPA', self.beam.bpa.to(U.deg).value))
        header.append(('OBSERVER', 'K. Oman'))
        #header.append(('NITERS', ???))
        #header.append(('RMS', ???))
        #header.append(('LWIDTH', ???))
        #header.append(('LSTEP', ???))
        header.append(('BUNIT', str(self.datacube._array.unit).replace(' ', '')))
        #header.append(('PCDEC', ???))
        #header.append(('LSTART', ???))
        header.append(('DATE-OBS', datetime.utcnow().isoformat()[:-5]))
        #header.append(('LTYPE', ???))
        #header.append(('PCRA', ???))
        #header.append(('CELLSCAL', ???))
        header.append(('BMAJ', self.beam.bmaj.to(U.deg).value))
        header.append(('BMIN', self.beam.bmin.to(U.deg).value))
        header.append(('BTYPE', 'Intensity'))
        header.append(('SPECSYS', wcs_header['SPECSYS']))

        hdu = fits.PrimaryHDU(header=header, data=self.datacube._array.value.T) #flip axes to write
        hdu.writeto(filename, overwrite=True)

