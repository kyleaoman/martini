from scipy.signal import fftconvolve
import numpy as np
import astropy.units as U
from astropy.io import fits
from astropy import __version__ as astropy_version
from datetime import datetime
from itertools import product
from multiprocessing import Pool

class Martini():

    def __init__(self, source=None, datacube=None, beam=None, baselines=None, noise=None, sph_kernel=None, spectral_model=None, logtag=''):
        self.source = source
        self.datacube = datacube
        self.beam = beam
        self.baselines = baselines
        self.noise = noise
        self.sph_kernel = sph_kernel
        self.spectral_model = spectral_model
        self.logtag = logtag

        if self.beam is not None:
            self.beam.init_kernel(self.datacube)
            self.datacube.add_pad(self.beam.needs_pad())

        self.prune_source()

        self.spectral_model.init_spectra(self.source, self.datacube)
        
        return

    def convolve_beam(self):
        unit = self.datacube._array.unit
        self.datacube._array = fftconvolve(
            self.datacube._array, 
            self.beam.kernel,
            mode='same'
        ) * unit
        self.datacube.drop_pad()
        self.datacube._array = self.datacube._array.to(U.Jy * U.beam ** -1, equivalencies=[self.beam.arcsec_to_beam])
        return        

    def add_noise(self):
        self.datacube._array = self.datacube._array + self.noise.generate(self.datacube)
        return

    def prune_source(self):
        origin = 0 #pixels indexed from 0 (not like in FITS!) for better use with numpy
        particle_coords = np.vstack(self.datacube.wcs.sub(3).wcs_world2pix(
            self.source.sky_coordinates.ra.to(self.datacube.units[0]), 
            self.source.sky_coordinates.dec.to(self.datacube.units[1]),
            self.source.sky_coordinates.radial_velocity.to(self.datacube.units[2]),
            origin)) * U.pix
        sm_length = np.arctan(
            self.source.hsm_g / self.source.sky_coordinates.distance
        ).to(U.pix, U.pixel_scale(self.datacube.px_size / U.pix))
        sm_range = np.ceil(sm_length).astype(int)
        spectrum_half_width = self.spectral_model.half_width(self.source) / self.datacube.channel_width
        reject_conditions = (
            (particle_coords[:2] + sm_range[np.newaxis] < 0 * U.pix).any(axis=0),
            particle_coords[0] - sm_range > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            particle_coords[1] - sm_range > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            particle_coords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            particle_coords[2] - 4 * spectrum_half_width * U.pix > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(particle_coords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        self.source.apply_mask(np.logical_not(reject_mask))
        return
    
    def insert_source_in_cube(self, skip_validation=False):
        origin = 0 #pixels indexed from 0 (not like in FITS!) for better use with numpy
        particle_coords = np.vstack(self.datacube.wcs.sub(3).wcs_world2pix(
            self.source.sky_coordinates.ra.to(self.datacube.units[0]), 
            self.source.sky_coordinates.dec.to(self.datacube.units[1]),
            self.source.sky_coordinates.radial_velocity.to(self.datacube.units[2]),
            origin)) * U.pix
        sm_length = np.arctan(
            self.source.hsm_g / self.source.sky_coordinates.distance
        ).to(U.pix, U.pixel_scale(self.datacube.px_size / U.pix))
        if skip_validation != True:
            self.sph_kernel.validate(sm_length)
        sm_range = np.ceil(sm_length).astype(int)
        
        #pixel iteration   
        ij_pxs = list(product(
            np.arange(self.datacube._array.shape[0]), 
            np.arange(self.datacube._array.shape[1])
        ))
        print('  ' + self.logtag + '  [rows: {0:.0f}, columns: {1:.0f}]'.format(self.datacube._array.shape[0], self.datacube._array.shape[1]))
        for ij_px in ij_pxs:
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            if (ij[1, 0].value == 0) and (ij[0, 0].value % 100 == 0):
                print('  ' + self.logtag + '  [row {:.0f}]'.format(ij[0, 0].value))
            mask = (np.abs(ij - particle_coords[:2]) <= sm_range).all(axis=0)
            weights = self.sph_kernel.px_weight(
                particle_coords[:2, mask] - ij,
                sm_length[mask]
            )
            self.datacube._array[ij_px[0], ij_px[1], :, 0] = (self.spectral_model.spectra[mask] * weights[..., np.newaxis]).sum(axis=-2)
            
        self.datacube._array = self.datacube._array / np.power(self.datacube.px_size / U.pix, 2)
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
        if wcs_header['CUNIT3'] == 'm s-1':
            header.append(('CUNIT3', 'm/s'))
        else:
            header.append(('CUNIT3', wcs_header['CUNIT3']))
        header.append(('CDELT4', wcs_header['CDELT4']))
        header.append(('CRPIX4', wcs_header['CRPIX4']))
        header.append(('CRVAL4', wcs_header['CRVAL4']))
        header.append(('CTYPE4', wcs_header['CTYPE4']))
        header.append(('CUNIT4', 'PAR'))
        header.append(('EPOCH', 2000))
        header.append(('INSTRUME', 'WSRT', 'MARTINI Synthetic'))
        #header.append(('BLANK', -32768)) #only for integer data
        header.append(('BSCALE', 1.0))
        header.append(('BZERO', 0.0))
        header.append(('DATAMAX', np.max(self.datacube._array.value)))
        header.append(('DATAMIN', np.min(self.datacube._array.value)))
        header.append(('ORIGIN', 'astropy v'+astropy_version))
        header.append(('OBJECT', 'MOCK'))
        if self.beam is not None:
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
        if self.beam is not None:
            header.append(('BMAJ', self.beam.bmaj.to(U.deg).value))
            header.append(('BMIN', self.beam.bmin.to(U.deg).value))
        header.append(('BTYPE', 'Intensity'))
        header.append(('SPECSYS', wcs_header['SPECSYS']))

        hdu = fits.PrimaryHDU(header=header, data=self.datacube._array.value.T) #flip axes to write
        hdu.writeto(filename, overwrite=True)

