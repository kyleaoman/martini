"""Provide miscellaneous global utilities."""

from warnings import warn
import contextlib
from typing import Generator
import numpy as np
import astropy.units as U

try:
    import numba
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True

if not NUMBA_AVAILABLE:
    warn(  # pragma: no cover
        "'numba' is unavailable, 'martini' will run slowly. Installing 'numba' is "
        "recommended.",
        RuntimeWarning,
    )

# HI line transition frequency:
HIfreq: U.Quantity[U.Hz] = 1.420405751e9 * U.Hz

# dtypes where we can use streaming to speed up FITS writes:
_FITS_BITPIX = {
    "uint8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
    "float32": -32,
    "float64": -64,
}


def resolve_ncpu(ncpu: int) -> int:
    """
    Convert a requested thread count to an actual number of threads.

    Parameters
    ----------
    ncpu : int
        The requested number of threads. A value of ``-1`` means as many threads as
        available cores (as many as :mod:`numba` allows, or 1 without :mod:`numba`).

    Returns
    -------
    int
        The number of threads to use.

    Raises
    ------
    ValueError
        If ``ncpu`` is not a positive integer or ``-1``.
    """
    if ncpu == -1:
        if not NUMBA_AVAILABLE:
            return 1
        return int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
    if ncpu < 1:
        raise ValueError(f"ncpu must be a positive integer or -1, got {ncpu}.")
    return ncpu


@contextlib.contextmanager
def numba_threads(thread_count: int) -> Generator[None, None, None]:
    """
    Temporarily set a desired thread count.

    Parameters
    ----------
    thread_count : int
        The number of threads to use. A value of ``-1`` means as many threads as
        available cores.
    """
    orig_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(resolve_ncpu(thread_count))
        yield
    finally:
        numba.set_num_threads(orig_threads)


def _to_unit_in_place(
    q: U.Quantity, unit: U.UnitBase, equivalencies: list | None = None
) -> U.Quantity:
    """
    Convert a :class:`~astropy.units.Quantity` to a new unit, reusing its memory.

    :meth:`~astropy.units.Quantity.to` allocates new arrays. When the conversion is a pure
    scaling, the values are instead multiplied in place and the same memory is wrapped
    with the new unit. Otherwise (e.g. a conversion with an offset) this falls back to
    :meth:`~astropy.units.Quantity.to`. Must be used with caution because other references
    to the same memory with other units will be invalidated (but this is a general issue
    affecting :class:`~astropy.units.Quantity`).

    Parameters
    ----------
    q : ~astropy.units.Quantity
        The quantity to convert. Its values are modified in place.

    unit : ~astropy.units.UnitBase
        The unit to convert to.

    equivalencies : list, optional
        Equivalencies to use for the conversion.

    Returns
    -------
    ~astropy.units.Quantity
        The converted quantity, sharing memory with ``q`` if converted in place.
    """
    equivalencies = [] if equivalencies is None else equivalencies
    factor = (1.0 * q.unit).to_value(unit, equivalencies=equivalencies)
    is_scaling = (0.0 * q.unit).to_value(unit, equivalencies=equivalencies) == 0 and (
        np.isclose(
            (2.0 * q.unit).to_value(unit, equivalencies=equivalencies),
            2 * factor,
            rtol=1e-12,
            atol=0,
        )
    )
    if not is_scaling:
        return q.to(unit, equivalencies=equivalencies)
    values = q.view(np.ndarray)
    np.multiply(values, factor, out=values, casting="unsafe")
    return U.Quantity(values, unit, copy=False)


def _plan_segments(mem_estimate: np.ndarray, allowance: float) -> list[slice]:
    """
    Divide particles into contiguous batches that each fit within a memory allowance.

    Parameters
    ----------
    mem_estimate : ~numpy.ndarray
        Estimated memory required to process each particle.

    allowance : float
        Memory available to process each batch, in the same units as
        ``mem_estimate``.

    Returns
    -------
    list
        List of :class:`slice` objects selecting the particles in each batch.

    Raises
    ------
    ValueError
        If any single particle needs more memory than the allowance.
    """
    if np.any(mem_estimate > allowance):
        raise ValueError("Memory allowance insufficient for a single particle.")
    # Should always have a particle if we get here, but make very sure so we don't
    # enter an infinite loop:
    assert len(mem_estimate) > 0, (
        "Particles processing in batches but no particles present, this is an error."
    )
    segments = []
    segment_start = 0
    while segment_start < len(mem_estimate):
        cumsum = np.cumsum(mem_estimate[segment_start:])
        segment_length = np.searchsorted(cumsum, allowance, side="right")
        segment_end = segment_start + int(segment_length)
        segments.append(slice(segment_start, segment_end))
        segment_start = segment_end
    return segments
