from martini.sources import TNGSource
from martini import DataCube, Martini
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import GaussianKernel, CubicSplineKernel
import astropy.units as U


def tngdemo(cubefile='tngdemo.fits', beamfile='tngdemo_beam.fits',
            hdf5file='tngdemo.hdf5'):
    # This example runs in about 1 minute in the IllustrisTNG JupyterLab
    # environment.

    # Parameters myBasePath, mySnap and myId follow the usual TNG
    # conventions as in the illustris_python package.
    myBasePath = 'sims.TNG/TNG100-1/output/'
    mySnap = 99
    myId = 385350  # first central with 218 < Vmax < 222, and SFR > 1

    # The different martini sub-modules need to be initialized, see
    # https://kyleaoman.github.io/martini/build/html/martini.html for a
    # high-level overview. See the documentation for the individual sub-
    # modules for details of all configuration options. Comments below
    # highlight a few suggested best-practices specific to TNG.

    # SOURCE
    # The rotation configuration takes an inclination (here 60deg) and
    # rotation about the pole (here 0deg). The code attempts to
    # automatically align the galactic disk in the y-z plane by aligning
    # the angular momentum along the x-axis. The polar rotation is then
    # applied, and finally the disc inclined by a rotation around the
    # y-axis (the line of sight is along the x-axis). The automatic
    # alignment will work for typical reasonably isolated discs, but will
    # struggle when companions are present, when the angular momentum axis
    # is a poor tracer of the disc plane, and especially for satellites. If
    # finer control of the orientation is needed, derive the transformation
    # from the simulation box coordinates to the desired coordinates for
    # the 'observation', keeping in mind that the line of sight is along
    # the x-axis. This rotation matrix can then be passed to rotation as
    # {'rotmat': np.eye(3)} (here the identity rotation matrix used as an
    # example). A common problem in this case is deriving the inverse
    # transform instead of the forward transform, if unexpected results are
    # obtained, first try passing the transpose of the rotation matrix.
    # Note that initializing the source takes some time as the particle
    # data must be read from disk.
    source = TNGSource(
        myBasePath,
        mySnap,
        myId,
        distance=30 * U.Mpc,
        rotation={'L_coords': (60 * U.deg, 0. * U.deg)},
        ra=0. * U.deg,
        dec=0. * U.deg
    )

    # DATACUBE
    # It is usually advisable to set the centre of the cube to track the
    # centre of the source, as illustrated below. Note that the source
    # systemic velocity is set according to the distance and Hubble's law.
    # These values can instead be set explicitly, if desired. A datacube
    # with 128x128 pixels usually takes a few minutes, 1024x1024 could take
    # several hours. The number of channels has less influence on the
    # runtime. Most of the runtime is spent when M.insert_source_in_cube is
    # called below.
    datacube = DataCube(
        n_px_x=128,
        n_px_y=128,
        n_channels=64,
        px_size=10. * U.arcsec,
        channel_width=10. * U.km * U.s ** -1,
        velocity_centre=source.vsys,
        ra=source.ra,
        dec=source.dec
    )

    # BEAM
    # It is usually advisable to set the beam size to be ~3x the pixel
    # size. Note that the data cube is padded according to the size of the
    # beam, this usually results in the number of pixel rows printed in the
    # progress messages to differ from the requested dimensions. The
    # padding is required for accurate convolution with the beam, but
    # contains incorrect values after convolution and is discarded to
    # produce the final data cube of the requested size.
    beam = GaussianBeam(
        bmaj=30. * U.arcsec,
        bmin=30. * U.arcsec,
        bpa=0. * U.deg,
        truncate=3.
    )

    # NOISE
    # The noise is normally added before convolution with the beam (as
    # below in this example). The rms value passed is for the noise before
    # convolution, the rms noise in the output data cube will therefore
    # typically differ from this value.
    noise = GaussianNoise(
        rms=5.E-6 * U.Jy * U.arcsec ** -2
    )

    # SPECTRAL MODEL
    # The 'subgrid' velocity dispersion can also be fixed to a constant
    # value, e.g. sigma=7 * U.km * U.s**-1.
    spectral_model = GaussianSpectrum(
        sigma='thermal'
    )

    # SPH KERNEL
    # Since IllustrisTNG uses a moving mesh hydrodynamics solver (Arepo),
    # there are no formal SPH smoothing lengths and no specified kernel.
    # However, approximate smoothing lengths can be estimated from the
    # Voronoi cell volumes, so a reasonable approximation is to use these
    # for imaging. The lengths correspond roughly to a cubic spline kernel.
    # The implementation of the cubic spline kernel included in Martini
    # uses an approximation which breaks down when the particle smoothing
    # lengths are small compared to the size of the pixels (at the distance
    # of the source).
    # The Gaussian kernel implementation does not suffer from this
    # limitation and can be scaled to mimic the cubic spline kernel as
    # illustrated below. This solution should generally work; for
    # specialized applications, consider making a histogram of the
    # smoothing lengths of the particles in the source (in units of the
    # pixel scale) and consulting the documentation of the sph_kernels sub-
    # module to select an appropriate kernel.
    sph_kernel = GaussianKernel.mimic(
        CubicSplineKernel(rescale_sph_h=.5),
        truncate=3
    )

    M = Martini(
        source=source,
        datacube=datacube,
        beam=beam,
        noise=noise,
        spectral_model=spectral_model,
        sph_kernel=sph_kernel
    )

    # Progress messages will be printed every printfreq rows; suppress by
    # setting to None.
    M.insert_source_in_cube(printfreq=1)
    M.add_noise()
    M.convolve_beam()

    # Two output formats are available, depending on preference. Both
    # formats are self-documenting, via FITS header keywords and HDF5
    # attributes, respectively. For HDF5 output, the beam image is included
    # in the same file.
    M.write_fits(cubefile, channels='velocity')
    M.write_beam_fits(beamfile, channels='velocity')
    try:
        M.write_hdf5(hdf5file, channels='velocity')
    except ModuleNotFoundError:
        print('h5py package not present, skipping hdf5 output demo')
