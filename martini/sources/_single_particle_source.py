import numpy as np
import astropy.units as U
from .sph_source import SPHSource


class _SingleParticleSource(SPHSource):
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
    distance : Quantity, with dimensions of length, optional
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: 3 Mpc.)

    vpeculiar : Quantity, with dimensions of velocity, optional
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: 0 km/s.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension for the source centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination for the source centroid. (Default: 0 deg.)

    hsm_g: Quantity, with dimensions of length, optional
        Smoothing length for the particle. (Default: 1 kpc.)
    """

    def __init__(
        self,
        distance=3 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        hsm_g=1 * U.kpc,
    ):

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation={"rotmat": np.eye(3)},
            ra=ra,
            dec=dec,
            h=0.7,
            T_g=np.ones(1) * 1.0e4 * U.K,
            mHI_g=np.ones(1) * 1.0e4 * U.Msun,
            xyz_g=np.ones((1, 3)) * 1.0e-3 * U.kpc,
            vxyz_g=np.zeros((1, 3)) * U.km * U.s**-1,
            hsm_g=np.ones(1) * hsm_g,
        )
        return
