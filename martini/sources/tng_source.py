import io
import os
import numpy as np
from astropy import units as U, constants as C
from astropy.coordinates import ICRS
from .sph_source import SPHSource
from ..sph_kernels import _CubicSplineKernel, find_fwhm


def api_get(path, params=None, api_key=None):
    """
    Make a request to the TNG web API service.

    Parameters
    ----------
    path : str
        The request to submit to the API.

    params : dict, optional
        Additional options for the API request. (Default: ``None``)

    api_key : str
        API key to authenticate to the TNG web API service. (Default: ``None``)

    Returns
    -------
    out : str
        Response from the API, a JSON-encoded string.
    """
    import requests

    baseUrl = "http://www.tng-project.org/api/"
    r = requests.get(f"{baseUrl}/{path}", params=params, headers={"api-key": api_key})
    r.raise_for_status()
    if r.headers["content-type"] == "application/json":
        return r.json()
    return r


def cutout_file(simulation, snapNum, haloID):
    """
    Helper to generate a string identifying a cutout file.

    Parameters
    ----------
    simulation : str
        Identifier of the simulation.

    snapNum : int
        Snapshot identifier.

    haloID : int
        Halo identifier.

    Returns
    -------
    out : str
        A string to use for a cutout file.
    """
    return f"martini-cutout-{simulation}-{snapNum}-{haloID}.hdf5"


class TNGSource(SPHSource):
    """
    Class abstracting HI sources for use with IllustrisTNG simulations.

    If used in the IllustrisTNG JupyterLab environment
    (https://www.tng-project.org/data/lab/), files on disk are accessed directly.
    Otherwise, particles for the galaxy of interest are automatically retrieved using
    the TNG web API (https://www.tng-project.org/data/docs/api/).

    Use of the TNG web API requires an API key: login at
    https://www.tng-project.org/users/login/ or register at
    https://www.tng-project.org/users/register/ then obtain your API
    key from https://www.tng-project.org/users/profile/ and pass to TNGSource as the
    ``api_key`` parameter. An API key is not required if logged into the TNG JupyterLab.

    Parameters
    ----------
    simulation : str
        Simulation identifier string, for example ``"TNG100-1"``, see
        https://www.tng-project.org/data/docs/background/

    snapNum : int
        Snapshot number. In TNG, snapshot 99 is the final output. Note that
        if a 'mini' snapshot (see
        http://www.tng-project.org/data/docs/specifications/#sec1a) is selected then
        some additional approximations are used.

    subID : int
        Subhalo ID of the target object. Note that all particles in the FOF
        group to which the subhalo belongs are used to construct the data cube.
        This avoids strange 'holes' at the locations of other subhaloes in the
        same group, and gives a more realistic treatment of foreground and
        background emission local to the source. An object of interest could be
        found using https://www.tng-project.org/data/search/, for instance. The
        "ID" column in the search results on that page is the subID.

    api_key: str, optional
        Use of the TNG web API requires an API key: login at
        https://www.tng-project.org/users/login/ or register at
        https://www.tng-project.org/users/register/ then obtain your API
        key from https://www.tng-project.org/users/profile/ and provide as a string. An
        API key is not required if logged into the TNG JupyterLab. (Default: ``None``)

    cutout_dir: str, optional
        Ignored if running on the TNG JupyterLab. Directory in which to search for and
        save cutout files (see documentation at
        https://www.tng-project.org/data/docs/api/ for a description of cutouts). Cutout
        filenames are enforced by MARTINI. If `cutout_dir==None` (the default), then the
        data will always be downloaded. If a `cutout_dir` is provided, it will first be
        searched for the required data. If the data are found, the local copy is used,
        otherwise the data are downloaded and a local copy is saved in `cutout_dir` for
        future use. (Default: ``None``)

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
        simulation,
        snapNum,
        subID,
        api_key=None,
        cutout_dir=None,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        coordinate_frame=ICRS(),
    ):
        # optional dependencies for this source class
        import h5py
        from Hdecompose.atomic_frac import atomic_frac

        X_H = 0.76

        full_fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "CenterOfMass",
            "GFM_Metals",
        )
        mdi_full = [None, None, None, None, None, None, 0]
        mini_fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "Coordinates",
        )

        # are we running on the TNG jupyterlab?
        jupyterlab = os.path.exists("/home/tnguser/sims.TNG")
        if not jupyterlab:
            data_header = dict()
            data_sub = dict()
            data_g = dict()
            from requests import HTTPError

            if api_key is None:
                raise ValueError(
                    "A TNG API key is required: login at "
                    "https://www.tng-project.org/users/login/ or register at "
                    "https://www.tng-project.org/users/register/ then obtain your API "
                    "key from https://www.tng-project.org/users/profile/"
                )
            if cutout_dir is not None:
                grnr_file = os.path.join(
                    cutout_dir,
                    f"martini-cutout-grnr-{simulation}-{snapNum}-{subID}.npy",
                )
                if os.path.exists(grnr_file):
                    haloID = np.load(grnr_file)
                    have_cutout = True
                else:
                    have_cutout = False
            else:
                print("No cutout_dir provided, cutout will be downloaded.")
                have_cutout = False
            if have_cutout:
                # check for an existing local cutout file
                if not os.path.exists(
                    os.path.join(cutout_dir, cutout_file(simulation, snapNum, haloID))
                ):
                    have_cutout = False
                cfname = os.path.join(
                    cutout_dir, cutout_file(simulation, snapNum, haloID)
                )
                print(f"Using local cutout file {os.path.basename(cfname)}")
            if not have_cutout:  # not else because previous if can modify have_cutout
                print("No local cutout found, cutout will be downloaded.")
                sub_api_path = f"{simulation}/snapshots/{snapNum}/subhalos/{subID}"
                sub = api_get(sub_api_path, api_key=api_key)
                haloID = sub["grnr"]
                np.save(grnr_file, haloID)
                data_sub["SubhaloPos"] = np.array([sub[f"pos_{ax}"] for ax in "xyz"])
                data_sub["SubhaloVel"] = np.array([sub[f"vel_{ax}"] for ax in "xyz"])
                cutout_api_path = (
                    f"{simulation}/snapshots/{snapNum}/halos/{haloID}/cutout.hdf5"
                )
                cutout_request = dict(gas=",".join(full_fields_g))
                try:
                    cutout = api_get(
                        cutout_api_path, params=cutout_request, api_key=api_key
                    )
                except HTTPError:
                    cutout_request = dict(gas=",".join(mini_fields_g))
                    cutout = api_get(
                        cutout_api_path, params=cutout_request, api_key=api_key
                    )
                # hold file in memory
                cfname = io.BytesIO(cutout.content)
                if cutout_dir is not None:
                    # write a copy to disk for later use
                    ofile = os.path.join(
                        cutout_dir, cutout_file(simulation, snapNum, haloID)
                    )
                    print(f"Writing downloaded cutout to {ofile}")
                    with open(ofile, "wb") as of:
                        of.write(cutout.content)
                    with h5py.File(ofile, "r+") as of:
                        of.create_group(f"{subID}")
                        of[f"{subID}"].attrs["pos"] = data_sub["SubhaloPos"]
                        of[f"{subID}"].attrs["vel"] = data_sub["SubhaloVel"]
            with h5py.File(cfname, "r") as cf:
                minisnap = "CenterOfMass" not in cf["PartType0"].keys()
                fields_g = mini_fields_g if minisnap else full_fields_g
                data_g = {field: cf["PartType0"][field][()] for field in fields_g}
                if len(data_header) == 0:
                    data_header["HubbleParam"] = cf["Header"].attrs["HubbleParam"]
                    data_header["Redshift"] = cf["Header"].attrs["Redshift"]
                    data_header["Time"] = cf["Header"].attrs["Time"]
                if len(data_sub) == 0:
                    data_sub["SubhaloPos"] = cf[f"{subID}"].attrs["pos"]
                    data_sub["SubhaloVel"] = cf[f"{subID}"].attrs["vel"]
            X_H_g = X_H if minisnap else data_g["GFM_Metals"][:, 0]

        else:
            from ._illustris_tools import (
                loadHeader,
                loadSingle,
                loadSubset,
                getSnapOffsets,
            )

            if cutout_dir is not None:
                print("Running on TNG JupyterLab, cutout_dir ignored.")
            basePath = f"/home/tnguser/sims.TNG/{simulation}/output/"
            data_header = loadHeader(basePath, snapNum)
            data_sub = loadSingle(basePath, snapNum, subhaloID=subID)
            haloID = data_sub["SubhaloGrNr"]
            subset_g = getSnapOffsets(basePath, snapNum, haloID, "Group")
            try:
                data_g = loadSubset(
                    basePath,
                    snapNum,
                    "gas",
                    fields=full_fields_g,
                    subset=subset_g,
                    mdi=mdi_full,
                )
                minisnap = False
            except Exception as exc:
                if ("Particle type" in exc.args[0]) and (
                    "does not have field" in exc.args[0]
                ):
                    data_g.update(
                        loadSubset(
                            basePath,
                            snapNum,
                            "gas",
                            fields=("CenterOfMass",),
                            subset=subset_g,
                            sq=False,
                        )
                    )
                    minisnap = True
                    X_H_g = X_H
                else:
                    raise
            X_H_g = (
                X_H if minisnap else data_g["GFM_Metals"]
            )  # only loaded column 0: Hydrogen

        a = data_header["Time"]
        z = data_header["Redshift"]
        h = data_header["HubbleParam"]
        xe_g = data_g["ElectronAbundance"]
        rho_g = data_g["Density"] * 1e10 / h * U.Msun * np.power(a / h * U.kpc, -3)
        u_g = data_g["InternalEnergy"]  # unit conversion handled in T_g
        mu_g = 4 * C.m_p.to(U.g).value / (1 + 3 * X_H_g + 4 * X_H_g * xe_g)
        gamma = 5.0 / 3.0  # see http://www.tng-project.org/data/docs/faq/#gen4
        T_g = (gamma - 1) * u_g / C.k_B.to(U.erg / U.K).value * 1e10 * mu_g * U.K
        m_g = data_g["Masses"] * 1e10 / h * U.Msun
        # cast to float64 to avoid underflow error
        nH_g = U.Quantity(rho_g * X_H_g / mu_g, dtype=np.float64) / C.m_p
        # In TNG_corrections I set f_neutral = 1 for particles with density
        # > .1cm^-3. Might be possible to do a bit better here, but HI & H2
        # tables for TNG will be available soon anyway.
        fatomic_g = atomic_frac(
            z, nH_g, T_g, rho_g, X_H_g, onlyA1=True, TNG_corrections=True
        )
        mHI_g = m_g * X_H_g * fatomic_g
        try:
            xyz_g = data_g["CenterOfMass"] * a / h * U.kpc
        except KeyError:
            xyz_g = data_g["Coordinates"] * a / h * U.kpc
        vxyz_g = data_g["Velocities"] * np.sqrt(a) * U.km / U.s
        V_cell = (
            data_g["Masses"] / data_g["Density"] * np.power(a / h * U.kpc, 3)
        )  # Voronoi cell volume
        r_cell = np.power(3.0 * V_cell / 4.0 / np.pi, 1.0 / 3.0).to(U.kpc)
        # hsm_g has in mind a cubic spline that =0 at r=h, I think
        hsm_g = 2.5 * r_cell * find_fwhm(_CubicSplineKernel().kernel)
        xyz_centre = data_sub["SubhaloPos"] * a / h * U.kpc
        xyz_g -= xyz_centre
        vxyz_centre = data_sub["SubhaloVel"] * np.sqrt(a) * U.km / U.s
        vxyz_g -= vxyz_centre

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )
        return
