import astropy.units as U
from ._sph_source import SPHSource


class SOSource(SPHSource):
    """
    Class abstracting HI sources using the SimObj package for interface to
    simulation data.

    This class accesses simulation data via the SimObj package
    (https://github.com/kyleaoman/simobj); see the documentation of that
    package for further configuration instructions.

    Parameters
    ----------
    distance : Quantity, with dimensions of length, optional
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: 3 Mpc.)

    vpeculiar : Quantity, with dimensions of velocity, optional
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: 0 km/s.)

    rotation : dict, optional
        Keys may be any combination of `axis_angle`, `rotmat` and/or
        `L_coords`. These will be applied in this order. Note that the 'y-z'
        plane will be the one eventually placed in the plane of the "sky". The
        corresponding values:

        - `axis_angle` : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element a Quantity with \
        dimensions of angle, indicating the angle to rotate through.
        - `rotmat` : A (3, 3) numpy.array specifying a rotation.
        - `L_coords` : A 2-tuple containing an inclination and an azimuthal \
        angle (both Quantity instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about 'y').

        (Default: rotmat with the identity rotation.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension for the source centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination for the source centroid. (Default: 0 deg.)

    SO_args : dict
        Dictionary of keyword arguments to pass to a call to simobj.SimObj.
        Arguments are: `obj_id`, `snap_id`, `mask_type`, `mask_args`,
        `mask_kwargs`, `configfile`, `simfiles_configfile`, `ncpu`. See simobj
        package documentation for details. Provide SO_args or SO_instance, not
        both.

    SO_instance : SimObj instance
        An initialized SimObj object. Provide SO_instance or SO_args, not both.
    """

    def __init__(
            self,
            distance=3.*U.Mpc,
            vpeculiar=0*U.km/U.s,
            rotation={'L_coords': (60.*U.deg, 0.*U.deg)},
            ra=0.*U.deg,
            dec=0.*U.deg,
            SO_args=None,
            SO_instance=None,
            rescale_hsm_g=1
    ):

        from simobj import SimObj  # optional dependency for this source class

        self._SO_args = SO_args
        self.rescale_hsm_g = rescale_hsm_g
        if (SO_args is not None) and (SO_instance is not None):
            raise ValueError('martini.source.SOSource: Provide SO_args or '
                             'SO_instance, not both.')
        if SO_args is not None:
            with SimObj(**self._SO_args) as SO:
                super().__init__(
                    distance=distance,
                    rotation=rotation,
                    ra=ra,
                    dec=dec,
                    h=SO.h,
                    T_g=SO.T_g,
                    mHI_g=SO.mHI_g,
                    xyz_g=SO.xyz_g,
                    vxyz_g=SO.vxyz_g,
                    hsm_g=SO.hsm_g * self.rescale_hsm_g
                )
        elif SO_instance is not None:
            super().__init__(
                distance=distance,
                vpeculiar=vpeculiar,
                rotation=rotation,
                ra=ra,
                dec=dec,
                h=SO_instance.h,
                T_g=SO_instance.T_g,
                mHI_g=SO_instance.mHI_g,
                xyz_g=SO_instance.xyz_g,
                vxyz_g=SO_instance.vxyz_g,
                hsm_g=SO_instance.hsm_g
            )
        else:
            raise ValueError('martini.sources.SOSource: Provide one of SO_args'
                             ' or SO_instance.')
        return
