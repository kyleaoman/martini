import numpy as np
import astropy.units as U

# copied from github.com/kyleaoman/kyleaoman_utilities/kyleaoman_utilities/
# commit-id 63dc535c5de1a0dc3caf09f98a33c9d09aa0eddc
# Note: No git-based solution (e.g. via submodules) seems practical to include
# selected files from external repositories; a direct copy is included here
# to produce a self-contained package.


def L_align(xyz, vxyz, m, frac=0.3, saverot=None, Laxis="z"):
    """
    Determine the rotation matrix to align with the angular momentum vector of particles.

    Parameters
    ----------
    xyz : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Particle coordinates.

    vxyz : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Particle velocities.

    m : ~astropy.units.Quantity
        :class:`~astropy.units.Quantity`, with dimensions of mass.
        Particle masses.

    frac : float, optional
        Fraction of particles with smallest radii to use in calculation.
        (Default: ``0.3``)

    saverot : str, optional
        If not ``None``, name of file in which to save the rotation matrix. Uses
        :func:`numpy.save`, so a ``.npy`` extension is recommended.

    Laxis : str, optional
        One of ``"x"``, ``"y"``, ``"z"``, specifying the axis to which the angular
        momentum should be aligned. (Default: ``"z"``)
    """
    transposed = False
    if xyz.ndim != 2:
        raise ValueError(
            "L_align: cannot guess coordinate axis for input with" " ndim != 2."
        )
    elif (xyz.shape[0] == 3) and (xyz.shape[1] == 3):
        raise ValueError(
            "L_align: cannot guess coordinate axis for input with" " shape (3, 3)."
        )
    elif xyz.shape[1] == 3:
        xyz = xyz.T
        vxyz = vxyz.T
        transposed = True
    elif (xyz.shape[0] != 3) and (xyz.shape[1] != 3):
        raise ValueError(
            "L_align: coordinate array shape "
            + str(xyz.shape)
            + " invalid (one dim must be 3)."
        )

    rsort = np.argsort(np.sum(np.power(xyz, 2), axis=0), kind="quicksort")
    p = m[np.newaxis] * vxyz
    L = np.cross(xyz, p, axis=0)
    p = p[:, rsort]
    L = L[:, rsort]
    m = m[rsort]
    mcumul = np.cumsum(m) / np.sum(m)
    Nfrac = np.argmin(np.abs(mcumul - frac))
    Nfrac = np.max([Nfrac, 100])  # use a minimum of 100 particles
    Nfrac = np.min([Nfrac, len(m)])  # unless this exceeds particle count
    p = p[:, :Nfrac]
    L = L[:, :Nfrac]
    Ltot = np.sqrt(np.sum(np.power(np.sum(L, axis=1), 2)))
    Lhat = np.sum(L, axis=1) / Ltot
    zhat = Lhat / np.sqrt(np.sum(np.power(Lhat, 2)))  # normalized
    xaxis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)  # default unlikely Laxis
    if (zhat == xaxis).all():
        raise RuntimeError(
            "Angular momentum exactly aligned with arbitrarily chosen vector, L_align"
            " failed."
        )
    xhat = xaxis - xaxis.dot(zhat) * zhat
    xhat = xhat / np.sqrt(np.sum(np.power(xhat, 2)))  # normalized
    yhat = np.cross(zhat, xhat)  # guarantees right-handedness

    rotmat = np.vstack((xhat, yhat, zhat)).to(U.dimensionless_unscaled).value
    if Laxis == "z":
        pass
    elif Laxis == "y":
        rotmat = np.roll(rotmat, 2, axis=0)
    elif Laxis == "x":
        rotmat = np.roll(rotmat, 1, axis=0)
    else:
        raise ValueError("L_align: Laxis must be one of 'x', 'y' or 'z'.")

    if transposed:
        rotmat = rotmat.T
    if saverot is not None:
        np.save(saverot, rotmat)

    return rotmat
