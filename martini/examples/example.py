from martini import Martini, DataCube
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
from martini.spectral_models import GaussianSpectrum, DiracDeltaSpectrum
from martini.sph_kernels import WendlandC2Kernel, DiracDeltaKernel
from martini.sources import SOSource, SingleParticleSource
import astropy.units as U
from collections import namedtuple
import numpy as np

snap_id = namedtuple('snap_id', ['res', 'phys', 'vol', 'snap'])
mysnap = snap_id(res=1, phys='hydro', vol=1, snap=127)
obj_id = namedtuple('obj_id', ['fof', 'sub'])
myobj = obj_id(fof=8, sub=0)

SO_args = {
    'obj_id': myobj,
    'snap_id': mysnap,
    'mask_type': 'fof',
    'mask_args': (myobj, ),
    'mask_kwargs': dict(),
    'configfile': '~/code/simobj/simobj/configs/example.py',
    'simfiles_configfile': '~/code/simfiles/simfiles/configs/example.py',
    'cache_prefix': './cachedir/',
    'disable_cache': False,
    'ncpu': 0
}

source = SOSource(
    distance = 3.657 * U.Mpc,
    rotation = {'L_coords': (60. * U.deg, 0. * U.deg)},
    SO_args = SO_args
)

#SingleParticleSource(
#    distance = 3. * U.Mpc,
#    rotation = {'rotmat': np.eye(3)}
#)

datacube = DataCube(
    n_px_x = 1024, #64
    n_px_y = 1024, #64
    n_channels = 100, #32
    px_size = 3. * U.arcsec, #30
    channel_width = 4. * U.km * U.s ** -1, #16
    velocity_centre = source.vsys
)

beam = GaussianBeam(
    bmaj = 6. * U.arcsec, #60
    bmin = 6. * U.arcsec, #60
    bpa = 0. * U.deg,
    truncate = 4.
)

baselines = None

noise = GaussianNoise(
    rms = 1.E-5 * U.Jy * U.arcsec ** -2
)

spectral_model = GaussianSpectrum(
    sigma = 'thermal'
)

#spectral_model = DiracDeltaSpectrum()

sph_kernel = WendlandC2Kernel()

M = Martini(
    source=source, 
    datacube=datacube, 
    beam=beam,
    baselines=baselines,
    noise=noise,
    spectral_model=spectral_model,
    sph_kernel=sph_kernel
)

M.insert_source_in_cube()
#M.add_noise()
M.convolve_beam()
M.write_fits('test{:.0f}'.format(datacube.n_px_x), channels='velocity')
