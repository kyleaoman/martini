"""Tests of the progress logger."""

import re
import pytest
from martini._logging import _ProgressLogger, _format_duration

TIMESTAMP = r"^\[\d\d:\d\d:\d\d\] "


class TestFormatDuration:
    """Test formatting of durations."""

    @pytest.mark.parametrize(
        "seconds, expected",
        (
            (0.12345, "00:00:00.1235"),
            (3723.5, "01:02:03.5000"),
            (90061.0, "25:01:01.0000"),
            (360123.5, "100:02:03.5000"),  # wider than 2 digits for hours
        ),
    )
    def test_format(self, seconds, expected):
        """Check the HH:MM:SS.ffff format, including more than 24 hours."""
        assert _format_duration(seconds) == expected


class TestProgressLogger:
    """Test the progress logger."""

    def test_message(self, capsys):
        """Check that messages are timestamped, printed and recorded."""
        logger = _ProgressLogger()
        logger.message("first", "second")
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 2
        assert all(re.match(TIMESTAMP + r"(first|second)$", line) for line in lines)
        assert logger.records == lines

    def test_step_nesting(self, capsys):
        """Check indentation of nested steps and the end message with duration."""
        logger = _ProgressLogger()
        with logger.step("Outer...", "Outer done"):
            logger.message("inside")
            with logger.step("Inner...", "Inner done") as step:
                step.end = "Inner finished"
                step.annotation = "on 2 cores"
        lines = capsys.readouterr().out.splitlines()
        assert re.match(TIMESTAMP + "Outer...", lines[0])
        assert re.match(TIMESTAMP + "  inside", lines[1])
        assert re.match(TIMESTAMP + "  Inner...", lines[2])
        assert re.match(
            TIMESTAMP + r"  Inner finished, took \d\d:\d\d:\d\d\.\d{4} on 2 cores$",
            lines[3],
        )
        assert re.match(
            TIMESTAMP + r"Outer done, took \d\d:\d\d:\d\d\.\d{4}$", lines[4]
        )
        assert len(lines) == 5

    def test_extra_indent(self, capsys):
        """Check extra indentation of detail messages."""
        logger = _ProgressLogger()
        logger.message("detail", extra_indent=1)
        assert re.match(TIMESTAMP + "  detail", capsys.readouterr().out.splitlines()[0])

    def test_quiet(self, capsys):
        """Check that quiet messages are recorded but not printed."""
        logger = _ProgressLogger()
        with logger.step("Start...", "End", quiet=True):
            logger.message("inside", quiet=True)
        assert capsys.readouterr().out == ""
        assert len(logger.records) == 3

    def test_exception_restores_depth(self, capsys):
        """Check that a failing step restores indentation and logs no end message."""
        logger = _ProgressLogger()
        with pytest.raises(RuntimeError):
            with logger.step("Failing...", "Failed"):
                raise RuntimeError
        logger.message("after")
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 2
        assert re.match(TIMESTAMP + "Failing...", lines[0])
        assert re.match(TIMESTAMP + "after", lines[1])
