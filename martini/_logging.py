"""Provide a progress logger with timestamps and nested step timing."""

import contextlib
import time
from collections.abc import Generator
from datetime import datetime


def _format_duration(seconds: float) -> str:
    """
    Format a duration as ``HH:MM:SS.ffff``.

    Parameters
    ----------
    seconds : float
        The duration in seconds.

    Returns
    -------
    str
        The formatted duration (hours may exceed 24).
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:07.4f}"


class _Step:
    """
    Record details of a step in progress, used by :meth:`_ProgressLogger.step`.

    Parameters
    ----------
    end : str
        Text of the message logged when the step ends (the duration is appended).
    """

    end: str
    annotation: str

    def __init__(self, end: str) -> None:
        self.end = end
        self.annotation = ""
        return


class _ProgressLogger:
    """
    Log progress messages with timestamps and indentation.

    Nested steps are also supported.

    Each message is prefixed with the wall-clock time and indented according to how
    deeply nested the step is within enclosing steps. All messages are recorded in
    :attr:`records`, and printed unless ``quiet``.

    Parameters
    ----------
    indent : str
        Indentation added for each level of nesting.

    Attributes
    ----------
    records : list
        All messages logged, as they were (or would have been) printed.
    """

    records: list[str]

    def __init__(self, indent: str = "  ") -> None:
        self._indent = indent
        self._depth = 0
        self.records = []
        return

    def message(self, *lines: str, quiet: bool = False, extra_indent: int = 0) -> None:
        """
        Log one or more messages, one per line.

        Parameters
        ----------
        *lines : str
            The messages.

        quiet : bool
            If ``True``, record the messages without printing them.

        extra_indent : int
            Additional levels of indentation (e.g. to show details below a message).
        """
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        prefix = self._indent * (self._depth + extra_indent)
        for line in lines:
            record = f"{timestamp} {prefix}{line}"
            self.records.append(record)
            if not quiet:
                print(record)
        return

    @contextlib.contextmanager
    def step(
        self, start: str, end: str, quiet: bool = False
    ) -> Generator[_Step, None, None]:
        """
        Log the start and end (with duration) of a step, indenting messages within.

        The message logged at the end of the step can be changed within the step by
        setting the ``end`` attribute of the yielded object, and further details can
        be appended after the duration by setting its ``annotation`` attribute.

        Parameters
        ----------
        start : str
            Message logged at the start of the step.

        end : str
            Message logged at the end of the step (the duration is appended).

        quiet : bool
            If ``True``, record the messages without printing them.

        Yields
        ------
        _Step
            Details of the step that can be modified while it is in progress.
        """
        self.message(start, quiet=quiet)
        step = _Step(end)
        self._depth += 1
        start_time = time.perf_counter()
        try:
            yield step
        finally:
            # restore indentation even if the step fails (then no end message is logged)
            self._depth -= 1
        elapsed = _format_duration(time.perf_counter() - start_time)
        annotation = f" {step.annotation}" if step.annotation else ""
        self.message(f"{step.end}, took {elapsed}{annotation}", quiet=quiet)
        return
