"""Scratch it."""

import numpy as np
from scipy.spatial.transform import Rotation
from astropy import units as U
from martini.sources import SPHSource, CombinedSource
from martini.datacube import DataCube
from martini.beams import GaussianBeam
from martini.sph_kernels import GaussianKernel
from martini.spectral_models import GaussianSpectrum
from martini.martini import Martini

n = 1000
r = np.random.rand(n) * 5 * U.kpc
v = 100 * U.km / U.s
t = np.random.rand(n) * 2 * np.pi * U.rad

source1 = SPHSource(
    distance=10 * U.Mpc,
    vpeculiar=30 * U.km / U.s,
    rotation=Rotation.from_euler("y", (30 * U.deg).to_value(U.rad)),
    ra=49.95 * U.deg,
    dec=29.95 * U.deg,
    T_g=np.ones(n) * U.K,
    mHI_g=np.ones(n) * 1e7 * U.Msun,
    xyz_g=U.Quantity(
        [
            np.zeros(n) * U.kpc,
            r * np.cos(t),
            r * np.sin(t),
        ]
    ),
    vxyz_g=U.Quantity(
        [
            np.zeros(n) * U.km / U.s,
            -v * np.sin(t),
            v * np.cos(t),
        ]
    ),
    hsm_g=np.ones(n) * 0.3 * U.kpc,
)
source2 = SPHSource(
    distance=12 * U.Mpc,
    vpeculiar=-50 * U.km / U.s,
    rotation=Rotation.from_euler("z", (60 * U.deg).to_value(U.rad)),
    ra=50.05 * U.deg,
    dec=30.05 * U.deg,
    T_g=np.ones(n) * U.K,
    mHI_g=np.ones(n) * 1e7 * U.Msun,
    xyz_g=U.Quantity(
        [
            np.zeros(n) * U.kpc,
            r * np.cos(t),
            r * np.sin(t),
        ]
    ),
    vxyz_g=U.Quantity(
        [
            np.zeros(n) * U.km / U.s,
            -v * np.sin(t) / 5,
            v * np.cos(t) / 5,
        ]
    ),
    hsm_g=np.ones(n) * 0.3 * U.kpc,
)
source = CombinedSource([source1, source2])
datacube = DataCube(
    n_px_x=128,
    n_px_y=128,
    n_channels=128,
    px_size=5 * U.arcsec,
    channel_width=3 * U.km / U.s,
    spectral_centre=760 * U.km / U.s,
    ra=50 * U.deg,
    dec=30 * U.deg,
)
beam = GaussianBeam(
    bmaj=15 * U.arcsec,
    bmin=15 * U.arcsec,
)
sph_kernel = GaussianKernel()
spectral_model = GaussianSpectrum(sigma="thermal")
m = Martini(
    source=source,
    datacube=datacube,
    beam=beam,
    noise=None,
    sph_kernel=sph_kernel,
    spectral_model=spectral_model,
)
m.preview(save="scratch.pdf", point_scaling="fixed")
# m.insert_source_in_cube()
# m.convolve_beam()
