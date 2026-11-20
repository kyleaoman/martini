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

HIfreq: U.Quantity[U.Hz] = 1.420405751e9 * U.Hz


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
