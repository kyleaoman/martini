import numpy as np
import astropy.units as U
from astropy.coordinates import ICRS
from .sph_source import SPHSource


class SOSource(SPHSource):
    """
    Class abstracting HI sources using the :mod:`simobj` package for interface to
    simulation data.

    This class accesses simulation data via the :mod:`simobj` package
    (https://github.com/kyleaoman/simobj); see the documentation of that
    package for further configuration instructions.

    Parameters
    ----------
    distance : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: ``3 * U.Mpc``)

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: ``0 * U.km * U.s**-1``)

    rotation : dict, optional
        Must have a single key, which must be one of ``axis_angle``, ``rotmat`` or
        ``L_coords``. Note that the 'y-z' plane will be the one eventually placed in the
        plane of the "sky". The corresponding value must be:

        - ``axis_angle`` : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element a :class:`~astropy.units.Quantity` with \
        dimensions of angle, indicating the angle to rotate through.
        - ``rotmat`` : A (3, 3) :class:`~numpy.ndarray` specifying a rotation.
        - ``L_coords`` : A 2-tuple containing an inclination and an azimuthal \
        angle (both :class:`~astropy.units.Quantity` instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about \
        'y'). A 3-tuple may be provided instead, in which case the third \
        value specifies the position angle on the sky (second rotation about 'x'). \
        The default position angle is 270 degrees.

        (Default: ``np.eye(3)``)

    ra : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Right ascension for the source centroid. (Default: ``0 * U.deg``)

    dec : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Declination for the source centroid. (Default: ``0 * U.deg``)

    coordinate_frame : ~astropy.coordinates.builtin_frames.baseradec.BaseRADecFrame, \
    optional
        The coordinate frame assumed in converting particle coordinates to RA and Dec, and
        for transforming coordinates and velocities to the data cube frame. The frame
        needs to have a well-defined velocity as well as spatial origin. Recommended
        frames are :class:`~astropy.coordinates.GCRS`, :class:`~astropy.coordinates.ICRS`,
        :class:`~astropy.coordinates.HCRS`, :class:`~astropy.coordinates.LSRK`,
        :class:`~astropy.coordinates.LSRD` or :class:`~astropy.coordinates.LSR`. The frame
        should be passed initialized, e.g. ``ICRS()`` (not just ``ICRS``).
        (Default: ``astropy.coordinates.ICRS()``)

    SO_args : dict, optional
        Dictionary of keyword arguments to pass to a call to
        :class:`simobj.simobj.SimObj`.
        Arguments are: ``obj_id``, ``snap_id``, ``mask_type``, ``mask_args``,
        ``mask_kwargs``, ``configfile``, ``simfiles_configfile``, ``ncpu``.
        See :mod:`simobj` package documentation for details. Provide ``SO_args`` or
        ``SO_instance``, not both.

    SO_instance : simobj.simobj.SimObj, optional
        An initialized :class:`simobj.simobj.SimObj` object. Provide ``SO_instance``
        or ``SO_args``, not both.
    """

    def __init__(
        self,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        coordinate_frame=ICRS(),
        SO_args=None,
        SO_instance=None,
        rescale_hsm_g=1,
    ):
        from simobj import SimObj  # optional dependency for this source class

        self._SO_args = SO_args
        self.rescale_hsm_g = rescale_hsm_g
        if (SO_args is not None) and (SO_instance is not None):
            raise ValueError(
                "martini.source.SOSource: Provide SO_args or " "SO_instance, not both."
            )
        elif SO_args is not None:
            with SimObj(**self._SO_args) as SO:
                particles = dict(
                    T_g=SO.T_g,
                    mHI_g=SO.mHI_g,
                    xyz_g=SO.xyz_g,
                    vxyz_g=SO.vxyz_g,
                    hsm_g=SO.hsm_g * self.rescale_hsm_g,
                )
                super().__init__(
                    distance=distance,
                    rotation=rotation,
                    ra=ra,
                    dec=dec,
                    h=SO.h,
                    **particles,
                )
        elif SO_instance is not None:
            particles = dict(
                T_g=SO_instance.T_g,
                mHI_g=SO_instance.mHI_g,
                xyz_g=SO_instance.xyz_g,
                vxyz_g=SO_instance.vxyz_g,
                hsm_g=SO_instance.hsm_g,
            )
            super().__init__(
                distance=distance,
                vpeculiar=vpeculiar,
                rotation=rotation,
                ra=ra,
                dec=dec,
                h=SO_instance.h,
                coordinate_frame=coordinate_frame,
                **particles,
            )
        else:
            raise ValueError(
                "martini.sources.SOSource: Provide one of SO_args" " or SO_instance."
            )
        return
