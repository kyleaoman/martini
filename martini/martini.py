import subprocess
import os
from scipy.signal import fftconvolve
import numpy as np
import astropy.units as U
from astropy.io import fits
from astropy import __version__ as astropy_version
from datetime import datetime
from itertools import product
from ._version import __version__ as martini_version
from warnings import warn

try:
    gc = subprocess.check_output(
        ['git', 'describe', '--always'],
        stderr=open(os.devnull, 'w'),
        cwd=os.path.dirname(os.path.realpath(__file__))
    )
except (subprocess.CalledProcessError, FileNotFoundError):
    pass
else:
    martini_version = martini_version + '_commit_' + gc.strip().decode()


def _gen_particle_coords(source=None, datacube=None):
    # pixels indexed from 0 (not like in FITS!) for better use with numpy
    origin = 0
    skycoords = source.sky_coordinates
    return np.vstack(
        datacube.wcs.sub(3).wcs_world2pix(
            skycoords.ra.to(datacube.units[0]),
            skycoords.dec.to(datacube.units[1]),
            skycoords.radial_velocity.to(datacube.units[2]),
            origin)) * U.pix


class Martini():
    """
    Creates synthetic HI data cubes from simulation data.

    Usual use of martini involves first creating instances of classes from each
    of the required and optional sub-modules, then creating a Martini with
    these instances as arguments. The object can then be used to create
    synthetic observations, usually by calling `insert_source_in_cube`,
    (optionally) `add_noise`, (optionally) `convolve_beam` and `write_fits` in
    order.

    Parameters
    ----------
    source : an instance of a class derived from martini.source._BaseSource
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). Sources leveraging the simobj package for reading
        simulation data (github.com/kyleaoman/simobj) and a few test sources
        (e.g. single particle) are provided, creation of customized sources,
        for instance to leverage other interfaces to simulation data, is
        straightforward. See sub-module documentation.

    datacube : martini.DataCube instance
        A description of the datacube to create, including pixels, channels,
        sky position. See sub-module documentation.

    beam : an instance of a class derived from beams._BaseBeam, optional
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See sub-module documentation.

    noise : an instance of a class derived from noise._BaseNoise, optional
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        sub-module documentation.

    sph_kernel : an instance of a class derived from sph_kernels._BaseSPHKernel
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        SPH kernel submodule documentation for guidance.

    spectral_model : an instance of a class derived from \
    spectral_models._BaseSpectrum
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See sub-module documentation.

    logtag : string
        String to prepend to standard output messages.

    See Also
    --------
    martini.sources
    martini.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models

    Examples
    --------
    More detailed examples can be found in the examples directory in the github
    distribution of the package.

    The following example illustrates basic use of martini, using a (very!)
    crude model of a gas disk. This example can be run by doing
    'from martini import demo; demo()'::

        from martini import Martini, DataCube
        from martini.beams import GaussianBeam
        from martini.noise import GaussianNoise
        from martini.spectral_models import GaussianSpectrum
        from martini.sph_kernels import DiracDeltaKernel
        from martini.sources import SPHSource
        import astropy.units as U
        import numpy as np

        # ------make a toy galaxy----------
        N = 1000
        phi = np.random.rand(N) * 2 * np.pi
        r = []
        for L in np.random.rand(N):
            def f(r):
                return L - .5 * (2 - np.exp(-r) * (np.power(r, 2) + 2 * r + 2))
            r.append(fsolve(f, 1.)[0])
        r = np.array(r)
        # exponential disk
        r *= 3 / np.sort(r)[N // 2]
        z = -np.log(np.random.rand(N))
        # exponential scale height
        z *= .5 / np.sort(z)[N // 2] * np.sign(np.random.rand(N) - .5)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz_g = np.vstack((x, y, z)) * U.kpc
        # linear rotation curve
        vphi = 100 * r / 6.
        vx = -vphi * np.sin(phi)
        vy = vphi * np.cos(phi)
        # small pure random z velocities
        vz = (np.random.rand(N) * 2. - 1.) * 5
        vxyz_g = np.vstack((vx, vy, vz)) * U.km * U.s ** -1
        T_g = np.ones(N) * 8E3 * U.K
        mHI_g = np.ones(N) / N * 5.E9 * U.Msun
        # ~mean interparticle spacing smoothing
        hsm_g = np.ones(N) * 2 / np.sqrt(N) * U.kpc
        # ---------------------------------

        source = SPHSource(
            distance=5. * U.Mpc,
            rotation={'L_coords': (60. * U.deg, 0. * U.deg)},
            ra=0. * U.deg,
            dec=0. * U.deg,
            h=.7,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g
        )

        datacube = DataCube(
            n_px_x=128,
            n_px_y=128,
            n_channels=32,
            px_size=10. * U.arcsec,
            channel_width=10. * U.km * U.s ** -1,
            velocity_centre=source.vsys
        )

        beam = GaussianBeam(
            bmaj=30. * U.arcsec,
            bmin=30. * U.arcsec,
            bpa=0. * U.deg,
            truncate = 4.
        )

        noise = GaussianNoise(
            rms=3.E-4 * U.Jy * U.arcsec ** -2
        )

        spectral_model = GaussianSpectrum(
            sigma=7 * U.km * U.s ** -1
        )

        sph_kernel = DiracDeltaKernel()

        M = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel
        )

        M.insert_source_in_cube()
        M.add_noise()
        M.convolve_beam()
        M.write_beam_fits('testbeam.fits', channels='velocity')
        M.write_fits('testcube.fits', channels='velocity')

    """

    def __init__(self,
                 source=None,
                 datacube=None,
                 beam=None,
                 noise=None,
                 sph_kernel=None,
                 spectral_model=None,
                 logtag=''):
        self.source = source
        self.datacube = datacube
        self.beam = beam
        self.noise = noise
        self.sph_kernel = sph_kernel
        self.spectral_model = spectral_model
        self.logtag = logtag

        if self.beam is not None:
            self.beam.init_kernel(self.datacube)
            self.datacube.add_pad(self.beam.needs_pad())

        self.sph_kernel._init_sm_lengths(source=source, datacube=datacube)
        self.sph_kernel._init_sm_ranges()
        self._prune_particles()  # prunes both source, and kernel if applicable

        self.spectral_model.init_spectra(self.source, self.datacube)

        return

    def convolve_beam(self):
        """
        Convolve the beam and DataCube.
        """

        if self.beam is None:
            warn('Skipping beam convolution, no beam object provided to '
                 'Martini.')

        unit = self.datacube._array.unit
        for spatial_slice in self.datacube.spatial_slices():
            # use a view [...] to force in-place modification
            spatial_slice[...] = fftconvolve(
                spatial_slice, self.beam.kernel, mode='same') * unit
        self.datacube.drop_pad()
        self.datacube._array = self.datacube._array.to(
            U.Jy * U.beam**-1, equivalencies=[self.beam.arcsec_to_beam])
        return

    def add_noise(self):
        """
        Insert noise into the DataCube.
        """

        if self.noise is None:
            warn('Skipping noise, no noise object provided to Martini.')
            return

        self.datacube._array = \
            self.datacube._array + self.noise.generate(self.datacube)
        return

    def _prune_particles(self):
        """
        Determines which particles cannot contribute to the DataCube and
        removes them to speed up calculation. Assumes the kernel is 0 at
        distances greater than the kernel size (which may differ from the
        SPH smoothing length).
        """

        particle_coords = _gen_particle_coords(
            source=self.source,
            datacube=self.datacube
        )
        spectrum_half_width = self.spectral_model.half_width(self.source) / \
            self.datacube.channel_width
        reject_conditions = (
            (particle_coords[:2] + self.sph_kernel.sm_ranges[np.newaxis] <
             0 * U.pix).any(axis=0),
            particle_coords[0] - self.sph_kernel.sm_ranges >
            (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            particle_coords[1] - self.sph_kernel.sm_ranges >
            (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            particle_coords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            particle_coords[2] - 4 * spectrum_half_width * U.pix >
            self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(particle_coords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        self.source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        self.sph_kernel.apply_mask(np.logical_not(reject_mask))
        return

    def insert_source_in_cube(self, skip_validation=False, printfreq=100):
        """
        Populates the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the DataCube is approximated for
            increased speed. For some combinations of pixel size, distance
            and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            RuntimeError if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter True. (Default: False.)

        printfreq : int or None, optional
            Every printfreq rows a message will be printed to track progress.
            Messages completely suppressed with printfreq=None. (Default: 100.)
        """

        particle_coords = _gen_particle_coords(
            source=self.source,
            datacube=self.datacube
        )
        self.sph_kernel.confirm_validation(noraise=skip_validation)

        # pixel iteration
        ij_pxs = list(
            product(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1])))
        if printfreq is not None:
            print(
                '  ' + self.logtag + '  [columns: {0:.0f}, rows: {1:.0f}]'
                .format(
                    self.datacube._array.shape[0],
                    self.datacube._array.shape[1]
                )
            )
        for ij_px in ij_pxs:
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            if printfreq is not None:
                if (ij[1, 0].value == 0) and (ij[0, 0].value % printfreq == 0):
                    print('  ' + self.logtag +
                          '  [row {:.0f}]'.format(ij[0, 0].value))
            mask = (np.abs(ij - particle_coords[:2])
                    <= self.sph_kernel.sm_ranges).all(axis=0)
            weights = self.sph_kernel.px_weight(
                particle_coords[:2, mask] - ij,
                mask=mask
            )
            self.datacube._array[ij_px[0], ij_px[1], :, 0] = \
                (self.spectral_model.spectra[mask] *
                 weights[..., np.newaxis]).sum(axis=-2)

        self.datacube._array = \
            self.datacube._array / np.power(self.datacube.px_size / U.pix, 2)
        return

    def write_fits(self, filename, channels='frequency', overwrite=True):
        """
        Output the DataCube to a FITS-format file.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)
        """

        self.datacube.drop_pad()
        if channels == 'frequency':
            self.datacube.freq_channels()
        elif channels == 'velocity':
            self.datacube.velocity_channels()
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
        header.append(('EXTEND', 'T'))
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
        header.append(('INSTRUME', 'MARTINI', martini_version))
        # header.append(('BLANK', -32768)) #only for integer data
        header.append(('BSCALE', 1.0))
        header.append(('BZERO', 0.0))
        header.append(('DATAMAX', np.max(self.datacube._array.value)))
        header.append(('DATAMIN', np.min(self.datacube._array.value)))
        header.append(('ORIGIN', 'astropy v' + astropy_version))
        # long names break fits format, don't let the user set this
        header.append(('OBJECT', 'MOCK'))
        if self.beam is not None:
            header.append(('BPA', self.beam.bpa.to(U.deg).value))
        header.append(('OBSERVER', 'K. Oman'))
        # header.append(('NITERS', ???))
        # header.append(('RMS', ???))
        # header.append(('LWIDTH', ???))
        # header.append(('LSTEP', ???))
        header.append(('BUNIT', self.datacube._array.unit.to_string('fits')))
        # header.append(('PCDEC', ???))
        # header.append(('LSTART', ???))
        header.append(('DATE-OBS', datetime.utcnow().isoformat()[:-5]))
        # header.append(('LTYPE', ???))
        # header.append(('PCRA', ???))
        # header.append(('CELLSCAL', ???))
        if self.beam is not None:
            header.append(('BMAJ', self.beam.bmaj.to(U.deg).value))
            header.append(('BMIN', self.beam.bmin.to(U.deg).value))
        header.append(('BTYPE', 'Intensity'))
        header.append(('SPECSYS', wcs_header['SPECSYS']))

        # flip axes to write
        hdu = fits.PrimaryHDU(header=header, data=self.datacube._array.value.T)
        hdu.writeto(filename, overwrite=overwrite)

        if channels == 'frequency':
            self.datacube.velocity_channels()
        return

    def write_beam_fits(self, filename, channels='frequency', overwrite=True):
        """
        Output the beam to a FITS-format file.

        The beam is written to file, with pixel sizes, coordinate system, etc.
        similar to those used for the DataCube.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        Raises
        ------
        ValueError
            If Martini was initialized without a beam.
        """

        if self.beam is None:
            raise ValueError("Martini.write_beam_fits: Called with beam set "
                             "to 'None'.")

        if channels == 'frequency':
            self.datacube.freq_channels()
        elif channels == 'velocity':
            self.datacube.velocity_channels()
        else:
            raise ValueError("Martini.write_beam_fits: Unknown 'channels' "
                             "value (use 'frequency' or 'velocity'.")

        filename = filename if filename[-5:] == '.fits' else filename + '.fits'

        wcs_header = self.datacube.wcs.to_header()

        header = fits.Header()
        header.append(('SIMPLE', 'T'))
        header.append(('BITPIX', 16))
        # header.append(('NAXIS', self.beam.kernel.ndim))
        header.append(('NAXIS', 3))
        header.append(('NAXIS1', self.beam.kernel.shape[0]))
        header.append(('NAXIS2', self.beam.kernel.shape[1]))
        header.append(('NAXIS3', 1))
        header.append(('EXTEND', 'T'))
        header.append(('BSCALE', 1.0))
        header.append(('BZERO', 0.0))
        # this is 1/arcsec^2, is this right?
        header.append(('BUNIT', self.beam.kernel.unit.to_string('fits')))
        header.append(('CRPIX1', self.beam.kernel.shape[0] // 2 + 1))
        header.append(('CDELT1', wcs_header['CDELT1']))
        header.append(('CRVAL1', wcs_header['CRVAL1']))
        header.append(('CTYPE1', wcs_header['CTYPE1']))
        header.append(('CUNIT1', wcs_header['CUNIT1']))
        header.append(('CRPIX2', self.beam.kernel.shape[1] // 2 + 1))
        header.append(('CDELT2', wcs_header['CDELT2']))
        header.append(('CRVAL2', wcs_header['CRVAL2']))
        header.append(('CTYPE2', wcs_header['CTYPE2']))
        header.append(('CUNIT2', wcs_header['CUNIT2']))
        header.append(('CRPIX3', 1))
        header.append(('CDELT3', wcs_header['CDELT3']))
        header.append(('CRVAL3', wcs_header['CRVAL3']))
        header.append(('CTYPE3', wcs_header['CTYPE3']))
        header.append(('CUNIT3', wcs_header['CUNIT3']))
        header.append(('SPECSYS', wcs_header['SPECSYS']))
        header.append(('BMAJ', self.beam.bmaj.to(U.deg).value))
        header.append(('BMIN', self.beam.bmin.to(U.deg).value))
        header.append(('BPA', self.beam.bpa.to(U.deg).value))
        header.append(('BTYPE', 'beam    '))
        header.append(('EPOCH', 2000))
        header.append(('OBSERVER', 'K. Oman'))
        # long names break fits format
        header.append(('OBJECT', 'MOCKBEAM'))
        header.append(('INSTRUME', 'MARTINI', martini_version))
        header.append(('DATAMAX', np.max(self.beam.kernel.value)))
        header.append(('DATAMIN', np.min(self.beam.kernel.value)))
        header.append(('ORIGIN', 'astropy v' + astropy_version))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header, data=self.beam.kernel.value[..., np.newaxis].T)
        hdu.writeto(filename, overwrite=True)

        if channels == 'frequency':
            self.datacube.velocity_channels()
        return

    def write_hdf5(self, filename, channels='frequency', overwrite=True,
                   memmap=False, compact=False):
        """
        Output the DataCube and Beam to a HDF5-format file. Requires the h5py
        package.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.hdf5' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        memmap: bool, optional
            If True, create a file-like object in memory and return it instead
            of writing file to disk. (Default: False.)

        compact: bool, optional
            If True, omit pixel coordinate arrays to save disk space. In this
            case pixel coordinates can still be reconstructed from FITS-style
            keywords stored in the FluxCube attributes. (Default: False.)
        """

        import h5py

        self.datacube.drop_pad()
        if channels == 'frequency':
            self.datacube.freq_channels()
        elif channels == 'velocity':
            pass
        else:
            raise ValueError("Martini.write_fits: Unknown 'channels' value "
                             "(use 'frequency' or 'velocity'.")

        filename = filename if filename[-5:] == '.hdf5' else filename + '.hdf5'

        wcs_header = self.datacube.wcs.to_header()

        mode = 'w' if overwrite else 'x'
        driver = 'core' if memmap else None
        h5_kwargs = {'backing_store': False} if memmap else dict()
        f = h5py.File(filename, mode, driver=driver, **h5_kwargs)
        f['FluxCube'] = self.datacube._array.value[..., 0]
        c = f['FluxCube']
        origin = 0  # index from 0 like numpy, not from 1
        if not compact:
            xgrid, ygrid, vgrid = np.meshgrid(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
                np.arange(self.datacube._array.shape[2])
            )
            cgrid = np.vstack((
                xgrid.flatten(),
                ygrid.flatten(),
                vgrid.flatten(),
                np.zeros(vgrid.shape).flatten()
            )).T
            wgrid = self.datacube.wcs.all_pix2world(cgrid, origin)
            ragrid = wgrid[:, 0].reshape(self.datacube._array.shape)[..., 0]
            decgrid = wgrid[:, 1].reshape(self.datacube._array.shape)[..., 0]
            chgrid = wgrid[:, 2].reshape(self.datacube._array.shape)[..., 0]
            f['RA'] = ragrid
            f['RA'].attrs['Unit'] = wcs_header['CUNIT1']
            f['Dec'] = decgrid
            f['Dec'].attrs['Unit'] = wcs_header['CUNIT2']
            f['channel_mids'] = chgrid
            f['channel_mids'].attrs['Unit'] = wcs_header['CUNIT3']
        c.attrs['AxisOrder'] = '(RA,Dec,Channels)'
        c.attrs['FluxCubeUnit'] = str(self.datacube._array.unit)
        c.attrs['deltaRA_in_RAUnit'] = wcs_header['CDELT1']
        c.attrs['RA0_in_px'] = wcs_header['CRPIX1'] - 1
        c.attrs['RA0_in_RAUnit'] = wcs_header['CRVAL1']
        c.attrs['RAUnit'] = wcs_header['CUNIT1']
        c.attrs['RAProjType'] = wcs_header['CTYPE1']
        c.attrs['deltaDec_in_DecUnit'] = wcs_header['CDELT2']
        c.attrs['Dec0_in_px'] = wcs_header['CRPIX2'] - 1
        c.attrs['Dec0_in_DecUnit'] = wcs_header['CRVAL2']
        c.attrs['DecUnit'] = wcs_header['CUNIT2']
        c.attrs['DecProjType'] = wcs_header['CTYPE2']
        c.attrs['deltaV_in_VUnit'] = wcs_header['CDELT3']
        c.attrs['V0_in_px'] = wcs_header['CRPIX3'] - 1
        c.attrs['V0_in_VUnit'] = wcs_header['CRVAL3']
        c.attrs['VUnit'] = wcs_header['CUNIT3']
        c.attrs['VProjType'] = wcs_header['CTYPE3']
        if self.beam is not None:
            c.attrs['BeamPA'] = self.beam.bpa.to(U.deg).value
            c.attrs['BeamMajor_in_deg'] = self.beam.bmaj.to(U.deg).value
            c.attrs['BeamMinor_in_deg'] = self.beam.bmin.to(U.deg).value
        c.attrs['DateCreated'] = datetime.utcnow().isoformat()[:-5]
        c.attrs['MartiniVersion'] = martini_version
        c.attrs['AstropyVersion'] = astropy_version
        if self.beam is not None:
            f['Beam'] = self.beam.kernel.value[..., np.newaxis]
            b = f['Beam']
            b.attrs['BeamUnit'] = self.beam.kernel.unit.to_string('fits')
            b.attrs['deltaRA_in_RAUnit'] = wcs_header['CDELT1']
            b.attrs['RA0_in_px'] = self.beam.kernel.shape[0] // 2
            b.attrs['RA0_in_RAUnit'] = wcs_header['CRVAL1']
            b.attrs['RAUnit'] = wcs_header['CUNIT1']
            b.attrs['RAProjType'] = wcs_header['CTYPE1']
            b.attrs['deltaDec_in_DecUnit'] = wcs_header['CDELT2']
            b.attrs['Dec0_in_px'] = self.beam.kernel.shape[1] // 2
            b.attrs['Dec0_in_DecUnit'] = wcs_header['CRVAL2']
            b.attrs['DecUnit'] = wcs_header['CUNIT2']
            b.attrs['DecProjType'] = wcs_header['CTYPE2']
            b.attrs['deltaV_in_VUnit'] = wcs_header['CDELT3']
            b.attrs['V0_in_px'] = 0
            b.attrs['V0_in_VUnit'] = wcs_header['CRVAL3']
            b.attrs['VUnit'] = wcs_header['CUNIT3']
            b.attrs['VProjType'] = wcs_header['CTYPE3']
            b.attrs['BeamPA'] = self.beam.bpa.to(U.deg).value
            b.attrs['BeamMajor_in_deg'] = self.beam.bmaj.to(U.deg).value
            b.attrs['BeamMinor_in_deg'] = self.beam.bmin.to(U.deg).value
            b.attrs['DateCreated'] = datetime.utcnow().isoformat()[:-5]
            b.attrs['MartiniVersion'] = martini_version
            b.attrs['AstropyVersion'] = astropy_version

        if channels == 'frequency':
            self.datacube.velocity_channels()
        if memmap:
            return f
        else:
            f.close()
            return

    def reset(self):
        """
        Re-initializes the DataCube with zero-values.
        """
        init_kwargs = dict(
            n_px_x=self.datacube.n_px_x,
            n_px_y=self.datacube.n_px_y,
            n_channels=self.datacube.n_channels,
            px_size=self.datacube.px_size,
            channel_width=self.datacube.channel_width,
            velocity_centre=self.datacube.velocity_centre,
            ra=self.datacube.ra,
            dec=self.datacube.dec
        )
        self.datacube.__init__(**init_kwargs)
        return
