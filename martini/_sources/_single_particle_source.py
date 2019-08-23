import numpy as np
import astropy.units as U
from ._sph_source import SPHSource


class SingleParticleSource(SPHSource):
    """
    Class illustrating inheritance from martini.sources.SPHSource, creates a
    single particle test source.

    A simple test source consisting of a single particle will be created. The
    particle has a mass of 10^4 Msun, a SPH smoothing length of 1 kpc, a
    temperature of 10^4 K, a position offset by (x, y, z) = (1 pc, 1 pc, 1 pc)
    from the source centroid, a peculiar velocity of 0 km/s, and will be placed
    in the Hubble flow assuming h = 0.7 and the distance provided.

    Parameters
    ----------
    distance : astropy.units.Quantity, with units of length
        Source distance, also used to place the source in the Hubble flow
        assuming h = 0.7.

    vpeculiar : astropy.units.Quantity, with dimensions of velocity
        Source peculiar velocity, added to the velocity from Hubble's law.

    ra : astropy.units.Quantity, with dimensions of angle
        Right ascension for the source centroid.

    dec : astropy.units.Quantity, with dimensions of angle
        Declination for the source centroid.

    Returns
    -------
    out : SingleParticleSource
        An appropriately initialized SingleParticleSource object.
    """

    def __init__(
            self,
            distance=3 * U.Mpc,
            vpeculiar=0 * U.km / U.s,
            ra=0. * U.deg,
            dec=0. * U.deg
    ):

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation={'rotmat': np.eye(3)},
            ra=ra,
            dec=dec,
            h=.7,
            T_g=np.ones(1) * 1.E4 * U.K,
            mHI_g=np.ones(1) * 1.E4 * U.solMass,
            xyz_g=np.ones((1, 3)) * 1.E-3 * U.kpc,
            vxyz_g=np.zeros((1, 3)) * U.km * U.s ** -1,
            hsm_g=np.ones(1) * U.kpc
            )
        return
