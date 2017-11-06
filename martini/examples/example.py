from martini import Martini, DataCube, Source
from martini.beams import GaussianBeam
from martini.noise import GaussianNoise
import astropy.units as U
from collections import namedtuple
import os

snap_id = namedtuple('snap_id', ['res', 'phys', 'vol', 'snap'])
mysnap = snap_id(res=3, phys='hydro', vol=1, snap=127)
obj_id = namedtuple('obj_id', ['fof', 'sub'])
myobj = obj_id(fof=1, sub=0)

SO_args = {
    'obj_id': myobj,
    'snap_id': mysnap,
    'mask_type': 'fofsub',
    'mask_args': (myobj, ),
    'mask_kwargs': dict(),
    'configfile': '~/code/simobj/simobj/configs/example.py',
    'simfiles_configfile': '~/code/simfiles/simfiles/configs/example.py',
    'cache_prefix': './',
    'disable_cache': False
}

source = Source(
    distance = 3. * U.Mpc,
    SO_args = SO_args,
    rotation = {'L_coords': (60. * U.deg, 0. * U.deg)}
)

datacube = DataCube(
    n_px_x = 64,
    n_px_y = 64,
    n_channels = 16,
    px_size = 6. * U.arcsec,
    channel_width = 4. * U.km * U.s ** -1
)

beam = GaussianBeam(
    bmaj = 60. * U.arcsec,
    bmin = 60. * U.arcsec,
    bpa = 0. * U.deg,
    truncate = 4.
)

baselines = None

noise = GaussianNoise(
    rms = 1. * U.Jy
)

M = Martini(source=source, datacube=datacube, beam=beam, noise=noise)
#M.convolve_beam()
#M.add_noise()
