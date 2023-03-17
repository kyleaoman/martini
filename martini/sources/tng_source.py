import os
import h5py
import numpy as np
import astropy.units as U
import astropy.constants as C
from .sph_source import SPHSource
from ..sph_kernels import CubicSplineKernel, find_fwhm


def api_get(path, params=None, api_key=None):
    import requests

    baseUrl = "http://www.tng-project.org/api/"
    r = requests.get(f"{baseUrl}/{path}", params=params, headers={"api-key": api_key})
    r.raise_for_status()
    if r.headers["content-type"] == "application/json":
        return r.json()
    return r


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
    api_key parameter. An API key is not required if logged into the TNG JupyterLab.

    Parameters
    ----------
    simulation : str
        Simulation identifier string, for example "TNG100-1", see
        https://www.tng-project.org/data/docs/background/

    snapNum : int
        Snapshot number. In TNG, snapshot 99 is the final output. Note that
        a full snapshot (not a 'mini' snapshot, see
        http://www.tng-project.org/data/docs/specifications/#sec1a) must be
        used.

    subID : int
        Subhalo ID of the target object. Note that all particles in the FOF
        group to which the subhalo belongs are used to construct the data cube.
        This avoids strange 'holes' at the locations of other subhaloes in the
        same group, and gives a more realistic treatment of foreground and
        background emission local to the source. An object of interest could be
        found using https://www.tng-project.org/data/search/, for instance. The
        "ID" column in the search results on that page is the subID.

    distance : Quantity, with dimensions of length, optional
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: 3 Mpc.)

    vpeculiar : Quantity, with dimensions of velocity, optional
        Source peculiar velocity, added to the velocity from Hubble's law.
        (Default: 0 km/s.)

    rotation : dict, optional
        Must have a single key, which must be one of `axis_angle`, `rotmat` or
        `L_coords`. Note that the 'y-z' plane will be the one eventually placed in the
        plane of the "sky". The corresponding value must be:

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
        momentum pole (rotation about 'x'), and inclined (rotation about \
        'y'). A 3-tuple may be provided instead, in which case the third \
        value specifies the position angle on the sky (rotation about 'x'). \
        The default position angle is 270 degrees.

        (Default: identity rotation matrix.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension for the source centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination for the source centroid. (Default: 0 deg.)
    """

    def __init__(
        self,
        simulation,
        snapNum,
        subID,
        api_key=None,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    ):
        # optional dependencies for this source class
        import io
        from Hdecompose.atomic_frac import atomic_frac

        X_H = 0.76

        # are we running on the TNG jupyterlab?
        jupyterlab = os.path.exists("/home/tnguser/sims.TNG")
        if not jupyterlab:
            from requests import HTTPError

            if api_key is None:
                raise ValueError(
                    "A TNG API key is required: login at "
                    "https://www.tng-project.org/users/login/ or register at "
                    "https://www.tng-project.org/users/register/ then obtain your API "
                    "key from https://www.tng-project.org/users/profile/"
                )
            data_header = dict()
            data_sub = dict()
            sim_header_api_path = f"{simulation}"
            snap_header_api_path = f"{simulation}/snapshots/{snapNum}"
            sub_api_path = f"{simulation}/snapshots/{snapNum}/subhalos/{subID}"
            sim_header = api_get(sim_header_api_path, api_key=api_key)
            snap_header = api_get(snap_header_api_path, api_key=api_key)
            sub = api_get(sub_api_path, api_key=api_key)
            haloID = sub["grnr"]
            data_sub["SubhaloPos"] = np.array([sub[f"pos_{ax}"] for ax in "xyz"])
            data_sub["SubhaloVel"] = np.array([sub[f"vel_{ax}"] for ax in "xyz"])
            data_header["HubbleParam"] = sim_header["hubble"]
            data_header["Redshift"] = snap_header["redshift"]
            data_header["Time"] = 1 / (1 + data_header["Redshift"])
            data_g = dict()
            cutout_api_path = (
                f"{simulation}/snapshots/{snapNum}/halos/{haloID}/cutout.hdf5"
            )
            fields_g = (
                "Masses",
                "Velocities",
                "InternalEnergy",
                "ElectronAbundance",
                "Density",
            )
            cutout_request = dict(gas=",".join(fields_g))
            cutout = api_get(cutout_api_path, params=cutout_request, api_key=api_key)
            with h5py.File(io.BytesIO(cutout.content), "r") as cutout_file:
                for field in fields_g:
                    data_g[field] = cutout_file["PartType0"][field][()]
            try:
                cutout_request = dict(gas="CenterOfMass")
                cutout = api_get(
                    cutout_api_path, params=cutout_request, api_key=api_key
                )
            except HTTPError:
                cutout_request = dict(gas="Coordinates")
                cutout = api_get(
                    cutout_api_path, params=cutout_request, api_key=api_key
                )
                coordinate_type = "Coordinates"
            else:
                coordinate_type = "CenterOfMass"
            with h5py.File(io.BytesIO(cutout.content), "r") as cutout_file:
                data_g[coordinate_type] = cutout_file["PartType0"][coordinate_type][()]
            try:
                cutout_request = dict(gas="GFM_Metals")
                cutout = api_get(
                    cutout_api_path, params=cutout_request, api_key=api_key
                )
            except HTTPError:
                X_H_g = X_H
            else:
                with h5py.File(io.BytesIO(cutout.content), "r") as cutout_file:
                    data_g["GFM_Metals"] = cutout_file["PartType0"]["GFM_Metals"][:, 0]
                X_H_g = data_g["GFM_Metals"]

        else:
            from ._illustris_tools import (
                loadHeader,
                loadSingle,
                loadSubset,
                getSnapOffsets,
            )

            basePath = f"/home/tnguser/sims.TNG/{simulation}/output/"
            data_header = loadHeader(basePath, snapNum)
            data_sub = loadSingle(basePath, snapNum, subhaloID=subID)
            haloID = data_sub["SubhaloGrNr"]
            fields_g = (
                "Masses",
                "Velocities",
                "InternalEnergy",
                "ElectronAbundance",
                "Density",
            )
            subset_g = getSnapOffsets(basePath, snapNum, haloID, "Group")
            data_g = loadSubset(
                basePath, snapNum, "gas", fields=fields_g, subset=subset_g
            )
            try:
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
            except Exception as exc:
                if ("Particle type" in exc.args[0]) and (
                    "does not have field" in exc.args[0]
                ):
                    data_g.update(
                        loadSubset(
                            basePath,
                            snapNum,
                            "gas",
                            fields=("Coordinates",),
                            subset=subset_g,
                            sq=False,
                        )
                    )
            else:
                raise
            try:
                data_g.update(
                    loadSubset(
                        basePath,
                        snapNum,
                        "gas",
                        fields=("GFM_Metals",),
                        subset=subset_g,
                        mdi=(0,),
                        sq=False,
                    )
                )
            except Exception as exc:
                if ("Particle type" in exc.args[0]) and (
                    "does not have field" in exc.args[0]
                ):
                    X_H_g = X_H
                else:
                    raise
            else:
                X_H_g = data_g["GFM_Metals"]  # only loaded column 0: Hydrogen

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
        hsm_g = 2.5 * r_cell * find_fwhm(CubicSplineKernel().kernel)
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
