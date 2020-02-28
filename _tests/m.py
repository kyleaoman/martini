import numpy as np
from martini import DataCube, Martini
from martini.beams import GaussianBeam
from martini.sources import _SingleParticleSource as SingleParticleSource
from martini.sph_kernels import DiracDeltaKernel, CubicSplineKernel
from martini.spectral_models import DiracDeltaSpectrum
from astropy import units as U

# This is a prototype for a test which checks that the input mass in the
# particles gives the correct total flux density in the datacube, by
# checking the conversion back to total mass.

# Should probably actually just inline this into martini to monitor how
# much of the input mass ends up in the datacube?

# SingleParticleSource has a mass of 1E4Msun,
# smoothing length of 1kpc, temperature of 1E4K

source = SingleParticleSource(
    distance=3 * U.Mpc
)

datacube = DataCube(
    n_px_x=64,
    n_px_y=64,
    n_channels=16,
    px_size=10 * U.arcsec,
    channel_width=10 * U.km / U.s,
    velocity_centre=source.vsys,
    ra=source.ra,
    dec=source.dec
)

beam = GaussianBeam(
    bmaj=30 * U.arcsec,
    bmin=30 * U.arcsec
)

noise = None

spectral_model = DiracDeltaSpectrum()

sph_kernel = DiracDeltaKernel()
sph_kernel = CubicSplineKernel()

M = Martini(
    source=source,
    datacube=datacube,
    beam=beam,
    noise=noise,
    spectral_model=spectral_model,
    sph_kernel=sph_kernel
)

M.insert_source_in_cube(printfreq=None)
M.convolve_beam()

# radiant intensity
Irad = M.datacube._array.sum()  # Jy / beam

# beam area, for a Gaussian beam
A = np.pi * beam.bmaj * beam.bmin / 4 / np.log(2) / U.beam

# distance
D = source.distance

# channel width
dv = datacube.channel_width

# flux
F = (Irad / A).to(U.Jy / U.arcsec ** 2) * datacube.px_size ** 2

# HI mass
MHI = 2.36E5 * U.Msun \
    * D.to(U.Mpc).value ** 2 \
    * F.to(U.Jy).value \
    * dv.to(U.km / U.s).value

# demand accuracy within 1%
try:
    assert np.isclose(
        MHI.to(U.Msun).value,
        source.mHI_g.sum().to(U.Msun).value,
        rtol=1E-10
    )
except AssertionError:
    print('Mass in cube is {:.10f} of input value.'.format(
        MHI.to(U.Msun).value / source.mHI_g.sum().to(U.Msun).value
    ))
else:
    print('Mass is OK within 1%.')

# This should probably be a separate test:
# M.write_fits('m.fits')
# with fits.open('m.fits') as f:
#     assert np.isclose(f[0].data.sum(), Irad.to(U.Jy / U.beam).value)
