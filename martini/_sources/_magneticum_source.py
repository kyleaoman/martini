import numpy as np
import astropy.units as U
from ._sph_source import SPHSource
from g3t.stable.g3read import GadgetFile, read_particles_in_box


class MagneticumSource(SPHSource):

    def __init__(
            self,
            snapBase=None,
            haloPosition=None,
            haloVelocity=None,
            haloRadius=None,
            xH=0.76,  # not in header
            Lbox=100*U.Mpc,  # what is it, actually?
            internal_units=dict(
                L=U.kpc,
                M=1E10 * U.Msun,
                V=U.km/U.s,
                T=U.K
            ),
            distance=3*U.Mpc,
            vpeculiar=0*U.km/U.s,
            rotation={'L_coords': (60.*U.deg, 0.*U.deg)},
            ra=0*U.deg,
            dec=0*U.deg
    ):

        particles = {}

        # Here all is still in code units
        header = GadgetFile(snapBase + '.0').header

        a = header.time
        h = header.HubbleParam

        l_unit = internal_units['L'] * a / h
        m_unit = internal_units['M'] / h
        v_unit = internal_units['V'] * np.sqrt(a)
        T_unit = internal_units['T']

        f_gas = read_particles_in_box(
            snapBase,
            haloPosition,
            haloRadius,
            ['POS ', 'VEL ', 'MASS', 'TEMP', 'NH  ', 'HSML'],
            [0]
        )

        particles['xyz_g'] = f_gas['POS '] * l_unit
        particles['vxyz_g'] = f_gas['VEL '] * v_unit
        # full kernel width - CHECK IF NEED .5?
        particles['hsm_g'] = f_gas['HSML'] * l_unit
        particles['T_g'] = f_gas['TEMP'] * T_unit
        particles['mHI_g'] = f_gas['NH  '] * xH * f_gas['MASS'] * m_unit

        particles['xyz_g'] -= haloPosition * l_unit
        particles['xyz_g'][particles['xyz_g'] > Lbox / 2.] -= Lbox.to(U.kpc)
        particles['xyz_g'][particles['xyz_g'] < -Lbox / 2.] += Lbox.to(U.kpc)
        particles['vxyz_g'] -= haloVelocity * v_unit

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            **particles
        )
        return
