"""Interpolant functions to render particle properties onto a grid."""

from scipy.spatial import distance, KDTree
from sklearn.neighbors import KDTree as lKDTree
from astropy import units as U
import numpy as np
from functools import partial
from typing import Callable


def is_iterable(obj: object) -> bool:
    """
    Check if ``obj`` is iterable.

    Parameters
    ----------
    obj : object
        Object to check for iteration support.

    Returns
    -------
    bool
        ``True`` if ``obj`` is iterable, ``False`` otherwise.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def eval_kernel(
    x_eval: U.Quantity[U.pix],
    x_particle: U.Quantity[U.pix],
    h: U.Quantity[U.pix],
    kernel: Callable[
        [U.Quantity[U.dimensionless_unscaled]],
        np.ndarray,
    ],
) -> U.Quantity[U.pix**-3]:
    """
    Evaluate the kernel.

    Parameters
    ----------
    x_eval:
            positions at which to evaluate kernel
    x_particle:
            positions of particles for which to evaluate kernel
    h:
            particle smoothing lengths
    kernel:
            kernel function

    Returns
    -------
    ~astropy.units.Quantity
        Evaluated kernel at ``x_eval`` for particles at positions ``x_particle`` with
        smoothing lengths ``h``. Has units of ``pixels**-3``.
    """
    q = distance.cdist(x_eval / h, x_particle / h)
    return kernel(q) / (h**3)


def _evalCacheKernel(q, kernelCache, kernelCacheResolution):
    return kernelCache[(np.clip(q, 0, 1) * kernelCacheResolution).astype(np.uint8)]


def sphLoop(
    M,
    MHI,
    P,
    T,
    H,
    dist,
    slices,
    cellVolumes,
    kernelCache,
    kernelCacheResolution,
    nPos,
    N,
    velocityUnit,
    massUnit,
    volumeUnit,
    maskOutOfBound,
):
    fieldMHI = np.zeros(nPos)
    fieldM = np.zeros(nPos)
    fieldV = np.zeros(nPos)
    fieldT = np.zeros(nPos)
    H3 = H**3
    for i in range(N):
        if len(slices[i]) == 0:
            continue
        particleKernel = (
            _evalCacheKernel(dist[i] / H[i], kernelCache, kernelCacheResolution) / H3[i]
        )
        if not maskOutOfBound[
            i
        ]:  # Since the particle is not out bound, we know the kernel should sum to 1. The kernel not summing to 1 is due to resolution effects.
            particleKernel /= np.sum(particleKernel * cellVolumes[slices[i]])
        fieldM[slices[i]] += particleKernel * M[i]
        fieldMHI[slices[i]] += particleKernel * MHI[i]
        fieldV[slices[i]] += particleKernel * P[i]
        fieldT[slices[i]] += particleKernel * T[i]
    kernelSlice = fieldM != 0
    finalV = np.zeros(nPos) * velocityUnit
    finalT = np.zeros(nPos) * velocityUnit**2
    finalMHI = fieldMHI * massUnit / volumeUnit
    finalV[kernelSlice] = fieldV[kernelSlice] * velocityUnit / fieldM[kernelSlice]
    finalT[kernelSlice] = fieldT[kernelSlice] * velocityUnit**2 / fieldM[kernelSlice]
    return finalV, finalMHI, finalT


def mfmLoop(
    M,
    MHI,
    P,
    T,
    H,
    dist,
    slices,
    cellVolumes,
    kernelCache,
    kernelCacheResolution,
    nPos,
    N,
    velocityUnit,
    massUnit,
    volumeUnit,
    maskOutOfBound,
):
    fieldMHI = np.zeros(nPos)
    fieldM = np.zeros(nPos)
    fieldV = np.zeros(nPos)
    fieldT = np.zeros(nPos)
    H3 = H**3
    totalKernel = np.zeros(nPos)
    for i in range(N):
        if len(slices[i]) == 0:
            continue
        particleKernel = (
            _evalCacheKernel(dist[i] / H[i], kernelCache, kernelCacheResolution) / H3[i]
        )
        totalKernel[slices[i]] += particleKernel
        slices[i] = slices[i][particleKernel != 0]
        dist[i] = dist[i][particleKernel != 0]
    fieldMHI = np.zeros(nPos)
    fieldM = np.zeros(nPos)
    fieldV = np.zeros(nPos)
    fieldT = np.zeros(nPos)
    for i in range(N):
        if len(slices[i]) == 0:
            continue
        particleKernel = (
            _evalCacheKernel(dist[i] / H[i], kernelCache, kernelCacheResolution) / H3[i]
        )
        volume = np.sum(
            particleKernel * (cellVolumes[slices[i]] / totalKernel[slices[i]])
        )
        if maskOutOfBound[i]:
            volume *= (
                np.pi * 4 / 3 * H[i] ** 3 / np.sum(cellVolumes[slices[i]])
            )  # for out of bounds particles, the volume is scaled up
        fieldMHI[slices[i]] += particleKernel * MHI[i] / volume
        fieldM[slices[i]] += particleKernel * M[i] / volume
        fieldV[slices[i]] += particleKernel * P[i] / volume
        fieldT[slices[i]] += particleKernel * T[i] / volume
    kernelSlice = totalKernel != 0
    finalV = np.zeros(nPos) * velocityUnit
    finalT = np.zeros(nPos) * velocityUnit**2
    finalMHI = np.zeros(nPos) * massUnit / volumeUnit
    finalM = np.zeros(nPos)
    finalMHI[kernelSlice] = (
        fieldMHI[kernelSlice] * massUnit / volumeUnit / totalKernel[kernelSlice]
    )
    finalM[kernelSlice] = fieldM[kernelSlice] / totalKernel[kernelSlice]
    finalV[kernelSlice] = (
        fieldV[kernelSlice]
        * velocityUnit
        / totalKernel[kernelSlice]
        / finalM[kernelSlice]
    )
    finalT[kernelSlice] = (
        fieldT[kernelSlice]
        * velocityUnit**2
        / totalKernel[kernelSlice]
        / finalM[kernelSlice]
    )
    return finalV, finalMHI, finalT


def _getOutOfBoundParticles(particlePos, particleRadius, fieldPos):
    lowBound = np.min(fieldPos, axis=0)
    topBound = np.max(fieldPos, axis=0)
    maskOutOfBound = ((particlePos + particleRadius[:, np.newaxis]) > topBound) | (
        (particlePos - particleRadius[:, np.newaxis]) < lowBound
    )
    maskOutOfBound = np.any(maskOutOfBound, axis=1)
    return maskOutOfBound


def particleScatter(
    mainLoop,
    X,
    V,
    H,
    MHI,
    T,
    M,
    kernel,
    fieldPos,
    dVolume,
    *,
    kernelCacheResolution=256,
    **kwargs,
):
    kernelCache = kernel(np.linspace(0, 1, kernelCacheResolution))
    maskOutOfBound = _getOutOfBoundParticles(X, H, fieldPos)
    M *= U.dimensionless_unscaled
    N, nDim = X.shape
    if V.ndim != 1:
        V = V[:, 0]  # more than one dimension of velocity is given, use radial velocity
    nPos = len(fieldPos)
    if not is_iterable(dVolume):
        dVolume = np.ones(nPos) * dVolume
    slices, dist = lKDTree(fieldPos.value).query_radius(
        X.value, H.value, return_distance=True
    )
    P = V.value * M.value
    thermal = T.to_value(V.unit**2) * M.value
    return mainLoop(
        M.value,
        MHI.value,
        P,
        thermal,
        H.value,
        dist,
        slices,
        dVolume.value,
        kernelCache,
        kernelCacheResolution,
        nPos,
        N,
        V.unit,
        MHI.unit,
        H.unit**3,
        maskOutOfBound,
    )


SPH = partial(particleScatter, sphLoop)
MFM = partial(particleScatter, mfmLoop)


def _evalVoronoiField(
    particleQuantities,
    nearestParticleIndices,
    missedParticleCellIndices,
    missedParticleMask,
    fieldNParticle,
):
    fieldQuantity = particleQuantities[nearestParticleIndices]
    fieldQuantity[missedParticleCellIndices] += particleQuantities[missedParticleMask]
    fieldQuantity /= fieldNParticle
    return fieldQuantity


def voronoiMesh(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, **kwargs):
    """
    Compute the interpolated radial velocity, density and temperature fields using voronoi mesh.
    Assumes that fieldPos creates a box.

    Parameters
    ----------
    X :
            particle positions
    V :
            particle radial velocities
    H :
            Unuseed.
            (particle volume)**(-3)
            Only used to convert MHI into a particle density.
            Exact volume is not required if you know the density.
    MHI :
            particle HI mass
    T :
            particle temperature in V**2 units
    M :
            particle mass (unused)
    kernel :
            unused
    fieldPos :
            positions at which to interpolate fields.
    dVolume :
            volume element size for fieldPos

    Returns
    -------
    fieldV : array astropy quantity
            interpolated velocity
    fieldMHI : array astropy quantity
            interpolated HI mass
    fieldT : array atropy quantity
            interpolated thermal velocity dispersion
    """

    M *= U.dimensionless_unscaled
    N, nDim = X.shape
    if V.ndim != 1:
        V = V[:, 0]  # more than one dimension of velocity is given, use radial velocity
    particleIndices = np.arange(len(X))
    _, nearestParticleIndices = KDTree(X).query(
        fieldPos
    )  # nearest neighbor assignment of particles to field pos

    # construct a mask for inbound particles but not assigned to a cell
    inboundParticleMask = np.all(X > fieldPos.min(axis=0), axis=1) & np.all(
        X < fieldPos.max(axis=0), axis=1
    )  # assume box shape for field pos
    usedParticleMask = np.isin(particleIndices, nearestParticleIndices)
    missedParticleMask = inboundParticleMask & ~usedParticleMask
    missedParticleIndices = particleIndices[missedParticleMask]
    _, missedParticleCellIndices = KDTree(fieldPos).query(X[missedParticleMask])

    particleMasks = nearestParticleIndices == particleIndices[:, np.newaxis]
    particleMasks[missedParticleIndices, missedParticleCellIndices] = True

    fieldNParticle = np.ones(len(fieldPos), dtype=np.uint64)
    fieldNParticle[missedParticleCellIndices] += 1

    particleVolumes = np.einsum(
        "ij,j->i", particleMasks, dVolume / fieldNParticle
    )  # for shared cells, the volume is divided between the particles
    density = np.zeros(MHI.shape) * MHI.unit / particleVolumes.unit
    volumeMask = ~(particleVolumes == 0)
    density[volumeMask] = MHI[volumeMask] / particleVolumes[volumeMask]
    fieldV = _evalVoronoiField(
        V,
        nearestParticleIndices,
        missedParticleCellIndices,
        missedParticleMask,
        fieldNParticle,
    )
    fieldMHI = _evalVoronoiField(
        density,
        nearestParticleIndices,
        missedParticleCellIndices,
        missedParticleMask,
        fieldNParticle,
    )
    fieldT = _evalVoronoiField(
        T,
        nearestParticleIndices,
        missedParticleCellIndices,
        missedParticleMask,
        fieldNParticle,
    )
    return fieldV, fieldMHI, fieldT


def manualSPH(X, V, H, MHI, T, M, kernel, fieldPos, dVolume, **kwargs):
    """
    Compute the interpolated radial velocity, density and temperature fields using SPH interpolation evaluated at fieldPos positions
    Note that different SPH schemes have different definitions for velocity interpolation.
    This interpolant assumes that the conserved quantities are interpolated.
    This SPH interpolant serves for testing purposes and writes the equations out explicitely.
    Consequently, it is slow but safe.

    Parameters
    ----------
    X :
            particle positions
    V :
            particle radial velocities
    H :
            particle smoothing lengths
    MHI :
            particle HI mass
    T :
            particle temperature in V**2 units
    M :
            particle mass
    kernel :
            kernel used in simulation
    fieldPos :
            positions at which to interpolate fields.
    dVolume :
            volume element size.

    Returns
    -------
    finalV : array astropy quantity
            interpolated velocity
    fieldMHI : array astropy quantity
            interpolated HI mass
    final T : array atropy quantity
            interpolated thermal velocity dispersion
    """
    M *= U.dimensionless_unscaled
    N, nDim = X.shape
    if V.ndim != 1:
        V = V[:, 0]  # more than one dimension of velocity is given, use radial velocity
    nPos = len(fieldPos)
    if not is_iterable(dVolume):
        dVolume = np.ones(nPos) * dVolume
    slices = KDTree(fieldPos).query_ball_point(X, H)
    fieldMHI = np.zeros(nPos) * MHI.unit / dVolume.unit
    fieldM = np.zeros(nPos) * M.unit / dVolume.unit
    fieldV = np.zeros(nPos) * V.unit * M.unit / dVolume.unit
    fieldT = np.zeros(nPos) * V.unit**2 * M.unit / dVolume.unit
    for i in range(N):
        particleKernel = eval_kernel(
            fieldPos[slices[i]], X[i].reshape((1, nDim)), H[i], kernel
        )[:, 0]
        fieldM[slices[i]] += particleKernel * M[i]
        fieldMHI[slices[i]] += particleKernel * MHI[i]
        fieldV[slices[i]] += (
            particleKernel * V[i] * M[i]
        )  # quantity of movement is conserved
        fieldT[slices[i]] += particleKernel * T[i] * M[i]  # thermal energy is conserved
    del slices
    kernelSlice = fieldM != 0
    finalV = np.zeros(nPos) * V.unit
    finalT = np.zeros(nPos) * V.unit**2
    finalV[kernelSlice] = fieldV[kernelSlice] / fieldM[kernelSlice]
    finalT[kernelSlice] = fieldT[kernelSlice] / fieldM[kernelSlice]
    return finalV, fieldMHI, finalT
