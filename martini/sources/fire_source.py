import numpy as np
from ..sph_kernels import _CubicSplineKernel, find_fwhm
from astropy import units as U, constants as C
from astropy.coordinates import ICRS
from .sph_source import SPHSource


class FIRESource(SPHSource):
    """
    Class abstracting HI sources designed to work with publicly available FIRE
    snapshot and group data.

    For file access, see https://flathub.flatironinstitute.org/fire

    The authors of the :mod:`gizmo_analysis` package request the folowing: "If you use
    this package in work that you publish, please cite it, along the lines of: 'This work
    used GizmoAnalysis (http://ascl.net/2002.015), which first was used in Wetzel et al.
    2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W).'"

    Parameters
    ----------
    simulation_directory : str, optional
        Base directory containing FIRE simulation output. (Default: ``"."``)

    snapshot_directory : str, optional
        Directory within ``simulation_directory`` containing snapshots. If ``None``,
        the default defined by :mod:`gizmo_analysis` is assumed (Default: ``None``)

    snapshot : tuple, optional
        A 2-tuple specifying the snapshot. The first element is the type of identifier,
        a string chosen from ``"index"``, ``"redshift"`` or ``"scalefactor"``. The
        second element is the desired value of the identifier. For example setting
        ``snapshot=("scalefactor", 0.2)`` will select the closest snapshot to
        :math:`a=0.2`. (Default: ``("redshift", 0)``)

    host_number : int, optional
        Galaxy ("host") position in the catalogue, indexed from 1. (Default: ``1``)

    assign_hosts : str, optional
        Method to compute galaxy centres. Iterative zoom-in based methods are
        recommended, options include ``"mass"`` (mass-weighted), ``"potential"``
        (potential-weighted) and ``"massfraction.metals"`` (metallicity-weighted).
        Other options are ``"track"`` (time-interpolated position) and ``"halo"``
        (from Rockstar halo catalogue). Setting ``True`` will attempt to find a
        sensible default by first trying ``"track"`` then ``"mass"``.
        (Default: ``"mass"``)

    convert_float32 : bool, optional
        If ``True``, convert floating point values to 32 bit precision to reduce
        memory usage. (Default: ``False``)

    gizmo_io_verbose : bool, optional
        If ``True``, allow the gizmo.io module to print progress and diagnostic messages.
        (Default: ``False``)

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
    """

    def __init__(
        self,
        simulation_directory=".",
        snapshot_directory=None,
        snapshot=("redshift", 0),
        host_number=1,
        assign_hosts="mass",
        convert_float32=False,
        gizmo_io_verbose=False,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        coordinate_frame=ICRS(),
    ):
        import gizmo_analysis as gizmo

        gizmo_read_kwargs = dict(
            species=["gas", "star"],
            properties=[
                "position",
                "velocity",
                "temperature",
                "size",
                "density",
                "mass",
                "potential",
                "massfraction.metals.hydrogen",
                "hydrogen.neutral.fraction",
            ],
            simulation_directory=simulation_directory,
            snapshot_value_kind=snapshot[0],
            snapshot_values=snapshot[1],
            assign_hosts=assign_hosts,
            host_number=host_number,
            convert_float32=convert_float32,
            verbose=gizmo_io_verbose,
        )
        if snapshot_directory is not None:
            gizmo_read_kwargs["snapshot_directory"] = snapshot_directory
        gizmo_snap = gizmo.io.Read.read_snapshots(
            **gizmo_read_kwargs,
        )
        particles = dict(
            xyz_g=(
                gizmo_snap["gas"]["position"]
                - gizmo_snap.host["position"][host_number - 1]
            )
            * gizmo_snap.snapshot["scalefactor"]
            * U.kpc,
            vxyz_g=(
                gizmo_snap["gas"]["velocity"]
                - gizmo_snap.host["velocity"][host_number - 1]
            )
            * U.km
            * U.s**-1,
            T_g=gizmo_snap["gas"]["temperature"] * U.K,
            # see doi:10.1093/mnras/sty1241 Appendix B for molecular partition:
            mHI_g=np.where(
                np.logical_and(
                    gizmo_snap["gas"]["temperature"] * U.K < 300 * U.K,
                    gizmo_snap["gas"]["density"] * U.Msun * U.kpc**-3
                    > C.m_p * 10 * U.cm**-3,
                ),
                0,
                gizmo_snap["gas"].prop("mass.hydrogen.neutral"),
            )
            * U.Msun,
            # per A. Wetzel, size / 0.5077 is radius of cpmpact support
            hsm_g=gizmo_snap["gas"]["size"]
            / 0.5077
            * U.kpc
            * find_fwhm(_CubicSplineKernel().kernel),
        )
        super().__init__(
            **particles,
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=gizmo_snap.info["hubble"],
            coordinate_frame=coordinate_frame,
        )
        return
