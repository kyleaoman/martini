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


@contextlib.contextmanager
def numba_threads(thread_count: int) -> Generator[None, None, None]:
    """
    Temporarily set a desired thread count.

    Parameters
    ----------
    thread_count : int
        The number of threads to use.
    """
    orig_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(thread_count)
        yield
    finally:
        numba.set_num_threads(orig_threads)
