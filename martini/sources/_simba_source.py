import numpy as np
from ._sph_source import SPHSource
from ..sph_kernels import CubicSplineKernel, find_fwhm
from os.path import join
from astropy import units as U, constants as C


class SimbaSource(SPHSource):
    """
    Class abstracting HI sources designed to work with SIMBA snapshot and
    group data.

    For file access, enquire with R. Davé (rad@roe.ac.uk).

    Parameters
    ----------
    snapPath : str
        Directory containing snapshot files.

    snapName : str
        Filename of snapshot file.

    groupPath : str
        Directory containing group catalogue files.

    groupName : str
        Group catalogue filename.

    groupID : int
        Identifier in the GroupID column of group catalogue.

    aperture : Quantity, with dimensions of length
        Radial extent of a region to load around the object of interest,
        in physical (not comoving, no little h) units. Using larger values
        will include more foreground/background, which may be desirable, but
        will also slow down execution and impair the automatic routine used
        to find a disc plane.

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
            snapPath=None,
            snapName=None,
            groupPath=None,
            groupName=None,
            groupID=None,
            aperture=50.*U.kpc,
            distance=3.*U.Mpc,
            vpeculiar=0*U.km/U.s,
            rotation={'L_coords': (60.*U.deg, 0.*U.deg)},
            ra=0.*U.deg,
            dec=0.*U.deg
    ):

        if snapPath is None:
            raise ValueError('Provide snapPath argument to SimbaSource.')
        if snapName is None:
            raise ValueError('Provide snapName argument to SimbaSource.')
        if groupPath is None:
            raise ValueError('Provide groupPath argument to SimbaSource.')
        if groupName is None:
            raise ValueError('Provide groupName argument to SimbaSource.')
        if groupID is None:
            raise ValueError('Provide groupID argument to SimbaSource.')

        # optional dependencies for this source class
        import h5py

        snapFile = join(snapPath, snapName)
        groupFile = join(groupPath, groupName)

        gamma = 5 / 3

        with h5py.File(snapFile, 'r') as f:
            a = f['Header'].attrs['Time']
            h = f['Header'].attrs['HubbleParam']
            lbox = f['Header'].attrs['BoxSize'] / h * U.kpc
            gas = f['PartType0']
            fH = gas['Metallicity'][()][:, 0]
            fHe = gas['Metallicity'][()][:, 1]
            xe = gas['ElectronAbundance'][()]
            particles = dict(
                xyz_g=gas['Coordinates'][()] * a / h * U.kpc,
                vxyz_g=gas['Velocities'][()] * np.sqrt(a) * U.km / U.s,
                T_g=(
                    (1 + 4 * fHe / (1 - fHe))
                    / (1 + fHe / 4 / (1 - fHe) + xe) * C.m_p
                    * (gamma - 1)
                    * gas['InternalEnergy'][()] * (U.km / U.s) ** 2 / C.k_B
                ).to(U.K),
                hsm_g=gas['SmoothingLength'][()] * a / h * U.kpc
                * find_fwhm(CubicSplineKernel().kernel),
                mHI_g=gas['Masses'][()] * fH * gas['GrackleHI'][()]
                * 1E10 / h * U.Msun
            )
            del fH, fHe, xe

        with h5py.File(groupFile, 'r') as f:
            groupIDs = f['galaxy_data/GroupID'][()]
            gmask = groupID == groupIDs
            # no h^-1 on minpotpos, not sure about comoving yet
            cop = f['galaxy_data/minpotpos'][()][gmask][0] * a * U.kpc
            vcent = f['galaxy_data/vel'][()][gmask][0] * np.sqrt(a) \
                * U.km / U.s

        particles['xyz_g'] -= cop
        particles['xyz_g'][particles['xyz_g'] > lbox / 2.] -= lbox
        particles['xyz_g'][particles['xyz_g'] < -lbox / 2.] += lbox
        particles['vxyz_g'] -= vcent

        mask = np.zeros(particles['xyz_g'].shape[0], dtype=np.bool)
        outer_cube = (np.abs(particles['xyz_g']) < aperture).all(axis=1)
        inner_cube = np.zeros(particles['xyz_g'].shape[0], dtype=np.bool)
        inner_cube[outer_cube] = (
            np.abs(particles['xyz_g'][outer_cube]) < aperture / np.sqrt(3)
        ).all(axis=1)
        need_distance = np.logical_and(outer_cube, np.logical_not(inner_cube))
        mask[inner_cube] = True
        mask[need_distance] = np.sum(
            np.power(particles['xyz_g'][need_distance], 2), axis=1
        ) < np.power(aperture, 2)

        for k, v in particles.items():
            particles[k] = v[mask]

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
