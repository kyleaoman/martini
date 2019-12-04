import numpy as np
import astropy.units as U
import astropy.constants as C
from ._sph_source import SPHSource
from ..sph_kernels import CubicSplineKernel, find_fwhm


class TNGSource(SPHSource):
    """
    Class abstracting HI sources designed to run in the IllustrisTNG JupyterLab
    environment for access to simulation data.

    Can also be used in other environments, but requires that the
    `illustris_python` module be importable, and further that the data are laid
    out on disk in the fiducial way
    (see: http://www.tng-project.org/data/docs/scripts/).

    Parameters
    ----------
    basePath : str
        Directory containing simulation data, for instance 'TNG100-1/output/',
        see also http://www.tng-project.org/data/docs/scripts/

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
        background emission local to the source.

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
    """

    def __init__(
            self,
            basePath,
            snapNum,
            subID,
            distance=3.*U.Mpc,
            vpeculiar=0*U.km/U.s,
            rotation={'L_coords': (60.*U.deg, 0.*U.deg)},
            ra=0.*U.deg,
            dec=0.*U.deg
    ):

        # optional dependencies for this source class
        from illustris_python.groupcat import loadSingle, loadHeader
        from illustris_python.snapshot import loadSubset, getSnapOffsets
        from Hdecompose.atomic_frac import atomic_frac

        X_H = 0.76
        data_header = loadHeader(basePath, snapNum)
        data_sub = loadSingle(basePath, snapNum, subhaloID=subID)
        haloID = data_sub['SubhaloGrNr']
        fields_g = ('Masses', 'Velocities', 'InternalEnergy',
                    'ElectronAbundance', 'Density')
        subset_g = getSnapOffsets(basePath, snapNum, haloID, "Group")
        data_g = loadSubset(basePath, snapNum, 'gas', fields=fields_g,
                            subset=subset_g)
        try:
            data_g.update(loadSubset(basePath, snapNum, 'gas',
                                     fields=('CenterOfMass', ),
                                     subset=subset_g, sq=False))
        except Exception as exc:
            if ('Particle type' in exc.args[0]) and \
               ('does not have field' in exc.args[0]):
                data_g.update(loadSubset(basePath, snapNum, 'gas',
                                         fields=('Coordinates', ),
                                         subset=subset_g, sq=False))
            else:
                raise
        try:
            data_g.update(loadSubset(basePath, snapNum, 'gas',
                                     fields=('GFM_Metals', ), subset=subset_g,
                                     mdi=(0, ), sq=False))
        except Exception as exc:
            if ('Particle type' in exc.args[0]) and \
               ('does not have field' in exc.args[0]):
                X_H_g = X_H
            else:
                raise
        else:
            X_H_g = data_g['GFM_Metals']  # only loaded column 0: Hydrogen

        a = data_header['Time']
        z = data_header['Redshift']
        h = data_header['HubbleParam']
        xe_g = data_g['ElectronAbundance']
        rho_g = data_g['Density'] * 1E10 / h * U.Msun \
            * np.power(a / h * U.kpc, -3)
        u_g = data_g['InternalEnergy']  # unit conversion handled in T_g
        mu_g = 4 * C.m_p.to(U.g).value / (1 + 3 * X_H_g + 4 * X_H_g * xe_g)
        gamma = 5. / 3.  # see http://www.tng-project.org/data/docs/faq/#gen4
        T_g = (gamma - 1) * u_g / C.k_B.to(U.erg / U.K).value * 1E10 * mu_g \
            * U.K
        m_g = data_g['Masses'] * 1E10 / h * U.Msun
        # cast to float64 to avoid underflow error
        nH_g = U.Quantity(rho_g * X_H_g / mu_g, dtype=np.float64) / C.m_p
        # In TNG_corrections I set f_neutral = 1 for particles with density
        # > .1cm^-3. Might be possible to do a bit better here, but HI & H2
        # tables for TNG will be available soon anyway.
        fatomic_g = atomic_frac(
            z,
            nH_g,
            T_g,
            rho_g,
            X_H_g,
            onlyA1=True,
            TNG_corrections=True
        )
        mHI_g = m_g * X_H_g * fatomic_g
        try:
            xyz_g = data_g['CenterOfMass'] * a / h * U.kpc
        except KeyError:
            xyz_g = data_g['Coordinates'] * a / h * U.kpc
        vxyz_g = data_g['Velocities'] * np.sqrt(a) * U.km / U.s
        V_cell = data_g['Masses'] / data_g['Density'] \
            * np.power(a / h * U.kpc, 3)  # Voronoi cell volume
        r_cell = np.power(3. * V_cell / 4. / np.pi, 1. / 3.).to(U.kpc)
        # hsm_g has in mind a cubic spline that =0 at h, I think
        hsm_g = 2.5 * r_cell * find_fwhm(CubicSplineKernel().kernel)
        xyz_centre = data_sub['SubhaloPos'] * a / h * U.kpc
        xyz_g -= xyz_centre
        vxyz_centre = data_sub['SubhaloVel'] * np.sqrt(a) * U.km / U.s
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
            hsm_g=hsm_g
        )
        return
