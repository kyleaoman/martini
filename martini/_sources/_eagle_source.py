import numpy as np
from os.path import join
import astropy.units as U
from ._sph_source import SPHSource


class EAGLESource(SPHSource):
    """
    Class abstracting HI sources designed to work with publicly available
    EAGLE snapshot + group files. For file access, see
    http://icc.dur.ac.uk/Eagle/database.php.

    Parameters
    ----------
    snapPath : string
        Directory containing snapshot files.

    snapBase : string
        Filename of snapshot files, omitting portion '.X.hdf5'. Note these must
        be the subfind-processed files, usually named 
        'eagle_subfind_particles[...]'.

    groupPath : string
        Directory containing group catalog files.

    groupBase : string
        Filename of group catalog files, omitting portion '.X.hdf5'. Note
        these must be the subfind tables, usually named 
        'eagle_subfind_tab[...]'.

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

    subBoxSize : astropy.units.Quantity, with dimensions of length
        Box half-side length of a region to load around the object of interest,
        in physical (not comoving, no little h) units. This is only to avoid
        needing to load the entire particle arrays. By default set to 1 Mpc,
        which should be adequate for most galaxies.

    distance : astropy.units.Quantity, with dimensions of length
        Source distance, also used to set the velocity offset via Hubble's law.

    vpeculiar : astropy.units.Quantity, with dimensions of velocity
        Source peculiar velocity, added to the velocity from Hubble's law.

    rotation : dict
        Keys may be any combination of 'axis_angle', 'rotmat' and/or
        'L_coords'. These will be applied in this order. Note that the 'y-z'
        plane will be the one eventually placed in the plane of the "sky". The
        corresponding values:
        - 'axis_angle' : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element an astropy.units.Quantity with \
        dimensions of angle, indicating the angle to rotate through.
        - 'rotmat' : A (3, 3) numpy.array specifying a rotation.
        - 'L_coords' : A 2-tuple containing an inclination and an azimuthal \
        angle (both astropy.units.Quantity instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about 'y').

    ra : astropy.units.Quantity, with dimensions of angle
        Right ascension for the source centroid.

    dec : astropy.units.Quantity, with dimensions of angle
        Declination for the source centroid.

    Returns
    -------
    out : EAGLESource
        An appropriately initialized EAGLESource object.
    """

    def __init__(
            self,
            snapPath=None,
            snapBase=None,
            groupPath=None,
            groupBase=None,
            fof=None,
            sub=None,
            subBoxSize=1*U.Mpc,
            distance=3.*U.Mpc,
            vpeculiar=0*U.km/U.s,
            rotation={'L_coords': (60.*U.deg, 0.*U.deg)},
            ra=0.*U.deg,
            dec=0.*U.deg
    ):

        if snapPath is None:
            raise ValueError('Provide snapPath argument to EAGLESource.')
        if snapBase is None:
            raise ValueError('Provide snapBase argument to EAGLESource.')
        if groupPath is None:
            raise ValueError('Provide groupPath argument to EAGLESource.')
        if groupBase is None:
            raise ValueError('Provide groupBase argument to EAGLESource.')
        if fof is None:
            raise ValueError('Provide fof argument to EAGLESource.')
        if sub is None:
            raise ValueError('Provide sub argument to EAGLESource.')

        # optional dependencies for this source class
        from read_eagle import EagleSnapshot
        from Hdecompose.atomic_frac import atomic_frac
        import h5py

        fileN = 0
        while True:  # will break when group found or raise when out of files
            groupFile = join(groupPath, groupBase+'.{:d}.hdf5'.format(fileN))
            with h5py.File(groupFile, 'r') as g:
                mask = np.logical_and(
                    np.array(g['/Subhalo/GroupNumber']) == fof,
                    np.array(g['/Subhalo/SubGroupNumber']) == sub
                )
                if not np.sum(mask):
                    fileN += 1
                    continue  # group not in this file, try next one
                a = g['Header'].attrs['Time']
                redshift = 1 / a - 1
                h = g['Header'].attrs['HubbleParam']
                lbox = g['Header'].attrs['BoxSize'] * U.Mpc / h
                code_to_g = g['/Units'].attrs['UnitMass_in_g'] * U.g
                code_to_cm = g['/Units'].attrs['UnitLength_in_cm'] * U.cm
                code_to_cm_s = g['/Units'].attrs['UnitVelocity_in_cm_per_s'] \
                    * U.cm / U.s
                dset = g['/Subhalo/CentreOfPotential']
                aexp = dset.attrs.get('aexp-scale-exponent')
                hexp = dset.attrs.get('h-scale-exponent')
                cop = (dset[mask, :] * np.power(a, aexp) * np.power(h, hexp)
                       * code_to_cm).to(U.kpc)
                dset = g['/Subhalo/Velocity']
                aexp = dset.attrs.get('aexp-scale-exponent')
                hexp = dset.attrs.get('h-scale-exponent')
                vcent = (dset[mask, :] * np.power(a, aexp) * np.power(h, hexp)
                         * code_to_cm_s).to(U.km / U.s)
                break

        snapFile = join(snapPath, snapBase+'.0.hdf5')
        subBoxSize = (subBoxSize * h / a).to(U.Mpc).value
        centre = (cop * h / a).to(U.Mpc).value
        eagle_data = EagleSnapshot(snapFile)
        region = np.vstack((
            centre - subBoxSize,
            centre + subBoxSize
        )).T.flatten()
        eagle_data.select_region(*region)

        with h5py.File(snapFile, 'r') as f:

            fH = f['/RuntimePars'].attrs['InitAbundance_Hydrogen']
            fHe = f['/RuntimePars'].attrs['InitAbundance_Helium']
            proton_mass = f['/Constants'].attrs['PROTONMASS'] * U.g
            mu = 1 / (fH + .25 * fHe)
            gamma = f['/RuntimePars'].attrs['EOS_Jeans_GammaEffective']
            T0 = f['/RuntimePars'].attrs['EOS_Jeans_TempNorm_K'] * U.K

            def fetch(att, ptype=0):
                # gas is type 0, only need gas properties
                tmp = eagle_data.read_dataset(ptype, att)
                dset = f['/PartType{:d}/{:s}'.format(ptype, att)]
                aexp = dset.attrs.get('aexp-scale-exponent')
                hexp = dset.attrs.get('h-scale-exponent')
                return np.array(tmp, dtype='f8') * np.power(a, aexp) \
                    * np.power(h, hexp)

            ng_g = fetch('GroupNumber')
            particles = dict(
                xyz_g=(fetch('Coordinates') * code_to_cm).to(U.kpc),
                vxyz_g=(fetch('Velocity') * code_to_cm_s).to(U.km / U.s),
                T_g=fetch('Temperature') * U.K,
                hsm_g=(fetch('SmoothingLength') * code_to_cm).to(U.kpc)
            )
            rho_g = fetch('Density') * U.g * U.cm ** -3
            SFR_g = fetch('StarFormationRate')
            Habundance_g = fetch('ElementAbundance/Hydrogen')

        particles['mHI_g'] = (atomic_frac(
            redshift,
            rho_g * Habundance_g / (mu * proton_mass),
            particles['T_g'],
            rho_g,
            Habundance_g,
            onlyA1=True,
            EAGLE_corrections=True,
            SFR=SFR_g,
            mu=mu,
            gamma=gamma,
            fH=fH,
            T0=T0
        ) * code_to_g).to(U.solMass)

        mask = ng_g == fof
        for k, v in particles.items():
            particles[k] = v[mask]

        particles['xyz_g'] -= cop
        particles['xyz_g'][particles['xyz_g'] > lbox / 2.] -= lbox.to(U.kpc)
        particles['xyz_g'][particles['xyz_g'] < -lbox / 2.] += lbox.to(U.kpc)
        particles['vxyz_g'] -= vcent

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
