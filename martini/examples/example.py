from martini import Martini, DataCube, Source
import astropy.units as U
from collections import namedtuple
import os

home = os.environ['HOME']

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
    'configfile': home + '/code/simobj/simobj/configs/example.py',
    'simfiles_configfile': home + '/code/simfiles/simfiles/configs/example.py',
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
    n_channels = 64,
    px_size = 15. * U.arcsec,
    channel_width = 4. * U.km * U.s ** -1
)

M = Martini(source=source, datacube=datacube)
