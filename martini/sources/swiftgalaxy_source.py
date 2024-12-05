"""
Provides the :class:`~martini.sources.swiftgalaxy_source.SWIFTGalaxySource` class for
working with SWIFT simulations as input.
"""

import numpy as np
from .sph_source import SPHSource
from astropy import units as U
from astropy.coordinates import ICRS


class SWIFTGalaxySource(SPHSource):
    """
    Class abstracting HI sources designed to work with the :mod:`swiftsimio` and
    :mod:`swiftgalaxy` modules.

    Parameters
    ----------
    galaxy : ~swiftgalaxy.reader.SWIFTGalaxy
        Instance of a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    distance : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: ``3 * U.Mpc``)

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity along the direction to the source centre.
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
    """

    def __init__(
        self,
        galaxy,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        coordinate_frame=ICRS(),
        _mHI_g=None,
    ):
        h = galaxy.metadata.cosmology.h
        mHI_g = (
            galaxy.gas.atomic_hydrogen_masses.to_astropy() if _mHI_g is None else _mHI_g
        )
        # SWIFT guarantees smoothing lengths are 2x kernel std
        # We should convert the 2x std of the kernel intrinsically used in the simulation
        # to the FWHM of the same kernel. For this we need to detect which kernel was
        # used.
        kernel_function = galaxy.metadata.hydro_scheme["Kernel function"].decode()
        compact_support_per_h = {
            "Quartic spline (M5)": 2.018932,
            "Quintic spline (M6)": 2.195775,
            "Cubic spline (M4)": 1.825742,
            "Wendland C2": 1.936492,
            "Wendland C4": 2.207940,
            "Wendland C6": 2.449490,
        }[kernel_function]
        fwhm_per_compact_support = {
            "Quartic spline (M5)": 0.637756,
            "Quintic spline (M6)": 0.577395,
            "Cubic spline (M4)": 0.722352,
            "Wendland C2": 0.627620,
            "Wendland C4": 0.560649,
            "Wendland C6": 0.504964,
        }[kernel_function]
        hsm_g = (
            galaxy.gas.smoothing_lengths.to_astropy()
            * compact_support_per_h
            * fwhm_per_compact_support
        )
        particles = dict(
            xyz_g=galaxy.gas.coordinates.to_astropy(),
            vxyz_g=galaxy.gas.velocities.to_astropy(),
            T_g=galaxy.gas.temperatures.to_astropy(),
            hsm_g=hsm_g,
            mHI_g=mHI_g,
        )
        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            coordinate_frame=coordinate_frame,
            coordinate_axis=1,
            **particles,
        )
        return
