import numpy as np
from .sph_source import SPHSource
from ..sph_kernels import _WendlandC2Kernel, find_fwhm
from os.path import join, normpath, sep
import astropy.units as U
from astropy.coordinates import ICRS


class EAGLESource(SPHSource):
    """
    Class abstracting HI sources designed to work with publicly available
    EAGLE snapshot + group data.

    For file access, see http://icc.dur.ac.uk/Eagle/database.php.

    Parameters
    ----------
    snapPath : str
        Directory containing snapshot files. The directory structure unpacked
        from the publicly available tarballs is expected; removing/renaming
        files or directories below this will cause errors.

    snapBase : str
        Filename of snapshot files, omitting portion ``'.X.hdf5'``.

    fof : int
        FOF group number of the target object. Note that all particles in the
        FOF group to which the subhalo belongs are used to construct the data
        cube. This avoids strange "holes" at the locations of other subhaloes
        in the same group, and gives a more realistic treatment of foreground
        and background emission local to the source. In the EAGLE database,
        this is the 'GroupNumber'.

    sub : int
        Subfind subhalo number of the target object. For centrals the subhalo
        number is 0, for satellites >0. In the EAGLE database, this is then
        'SubGroupNumber'.

    db_user : str
        Database username.

    db_key : str, optional
        Database password, or omit for a prompt at runtime. (Default: ``None``)

    subBoxSize : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Box half-side length of a region to load around the object of interest,
        in physical (not comoving, no little h) units. Using larger values
        will include more foreground/background, which may be desirable, but
        will also slow down execution and impair the automatic routine used
        to find a disc plane.

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

    print_query : bool, optional
        If True, the SQL query submitted to the EAGLE database is printed.
        (Default: ``False``)
    """

    def __init__(
        self,
        snapPath=None,
        snapBase=None,
        fof=None,
        sub=None,
        db_user=None,
        db_key=None,
        subBoxSize=50.0 * U.kpc,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        coordinate_frame=ICRS(),
        print_query=False,
    ):
        if snapPath is None:
            raise ValueError("Provide snapPath argument to EAGLESource.")
        if snapBase is None:
            raise ValueError("Provide snapBase argument to EAGLESource.")
        if fof is None:
            raise ValueError("Provide fof argument to EAGLESource.")
        if sub is None:
            raise ValueError("Provide sub argument to EAGLESource.")
        if db_user is None:
            raise ValueError("Provide EAGLE database username.")

        # optional dependencies for this source class
        from eagleSqlTools import connect, execute_query
        from pyread_eagle import EagleSnapshot
        from Hdecompose.atomic_frac import atomic_frac
        import h5py

        snapNum = int(snapBase.split("_")[1])
        volCode = normpath(snapPath).split(sep)[-2]
        query = (
            "SELECT "
            "  sh.redshift as redshift, "
            "  sh.CentreOfPotential_x as x, "
            "  sh.CentreOfPotential_y as y, "
            "  sh.CentreOfPotential_z as z, "
            "  sh.Velocity_x as vx, "
            "  sh.Velocity_y as vy, "
            "  sh.Velocity_z as vz "
            "FROM "
            "  {:s}_SubHalo as sh ".format(volCode) + "WHERE "
            "  sh.Snapnum = {:d} ".format(snapNum)
            + "  and sh.GroupNumber = {:d} ".format(fof)
            + "  and sh.SubGroupNumber = {:d}".format(sub)
        )
        if print_query:
            print("-----EAGLE-DB-QUERY-----")
            print(query)
            print("-------QUERY-ENDS-------")
        if db_key is None:
            print("EAGLE database")
        q = execute_query(connect(db_user, db_key), query)
        redshift = q["redshift"]
        a = np.power(1 + redshift, -1)
        cop = np.array([q[coord] for coord in "xyz"]) * a * U.Mpc
        vcent = np.array([q["v" + coord] for coord in "xyz"]) * U.km / U.s

        snapFile = join(snapPath, snapBase + ".0.hdf5")

        with h5py.File(snapFile, "r") as f:
            h = f["RuntimePars"].attrs["HubbleParam"]
            subBoxSize = (subBoxSize * h / a).to(U.Mpc).value
            centre = (cop * h / a).to(U.Mpc).value
            eagle_data = EagleSnapshot(snapFile)
            region = np.vstack((centre - subBoxSize, centre + subBoxSize)).T.flatten()
            eagle_data.select_region(*region)
            lbox = f["/Header"].attrs["BoxSize"] * U.Mpc / h
            fH = f["/RuntimePars"].attrs["InitAbundance_Hydrogen"]
            fHe = f["/RuntimePars"].attrs["InitAbundance_Helium"]
            proton_mass = f["/Constants"].attrs["PROTONMASS"] * U.g
            mu = 1 / (fH + 0.25 * fHe)
            gamma = f["/RuntimePars"].attrs["EOS_Jeans_GammaEffective"]
            T0 = f["/RuntimePars"].attrs["EOS_Jeans_TempNorm_K"] * U.K

            def fetch(att, ptype=0):
                # gas is type 0, only need gas properties
                tmp = eagle_data.read_dataset(ptype, att)
                dset = f["/PartType{:d}/{:s}".format(ptype, att)]
                aexp = dset.attrs.get("aexp-scale-exponent")
                hexp = dset.attrs.get("h-scale-exponent")
                return tmp[()] * np.power(a, aexp) * np.power(h, hexp)

            code_to_g = f["/Units"].attrs["UnitMass_in_g"] * U.g
            code_to_cm = f["/Units"].attrs["UnitLength_in_cm"] * U.cm
            code_to_cm_s = f["/Units"].attrs["UnitVelocity_in_cm_per_s"] * U.cm / U.s
            ng_g = fetch("GroupNumber")
            particles = dict(
                xyz_g=(fetch("Coordinates") * code_to_cm).to(U.kpc),
                vxyz_g=(fetch("Velocity") * code_to_cm_s).to(U.km / U.s),
                T_g=fetch("Temperature") * U.K,
                hsm_g=(fetch("SmoothingLength") * code_to_cm).to(U.kpc)
                * find_fwhm(_WendlandC2Kernel().kernel),
            )
            rho_g = fetch("Density") * U.g * U.cm**-3
            SFR_g = fetch("StarFormationRate")
            Habundance_g = fetch("ElementAbundance/Hydrogen")

        particles["mHI_g"] = (
            atomic_frac(
                redshift,
                rho_g * Habundance_g / (mu * proton_mass),
                particles["T_g"],
                rho_g,
                Habundance_g,
                onlyA1=True,
                EAGLE_corrections=True,
                SFR=SFR_g,
                mu=mu,
                gamma=gamma,
                fH=fH,
                T0=T0,
            )
            * code_to_g
        ).to(U.Msun)

        mask = ng_g == fof
        for k, v in particles.items():
            particles[k] = v[mask]

        particles["xyz_g"] -= cop
        particles["xyz_g"][particles["xyz_g"] > lbox / 2.0] -= lbox.to(U.kpc)
        particles["xyz_g"][particles["xyz_g"] < -lbox / 2.0] += lbox.to(U.kpc)
        particles["vxyz_g"] -= vcent

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            coordinate_frame=coordinate_frame,
            **particles,
        )
        return
