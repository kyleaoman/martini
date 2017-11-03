from martini import Martini, DataCube, Source
from martini.beams import GaussianBeam
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
    SO_args = SO_args
)

datacube = DataCube(
    n_px_x = 256,
    n_px_y = 256,
    n_channels = 5,
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

noise = None

M = Martini(source=source, datacube=datacube, beam=beam)

print(M.beam.kernel.shape)
M.datacube._array[64,64,0] = 1. * U.Jy
M.convolve_beam()
import matplotlib.pyplot as pp
sp = pp.subplot(111, aspect='equal')
pp.imshow(M.datacube._array[...,0].value)
pp.show()
