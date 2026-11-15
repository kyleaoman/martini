"""Provide miscellaneous global utilities."""

from warnings import warn
import contextlib
from typing import Generator

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
