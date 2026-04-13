"""Test consistency of code version numbers."""

import importlib.metadata
import martini


class TestVersion:
    """Test consistency of code version numbers."""

    def test_code_version(self):
        """Check that code version matches pyproject.toml version."""
        assert importlib.metadata.version("astromartini") == martini.__version__

    def test_codemeta_version(self):
        """Check that codemeta version matches version file."""
        with open("codemeta.json") as f:
            codemeta_content = f.read()
        assert (
            f'"version": "{importlib.metadata.version("astromartini")}",'
            in codemeta_content
        )


class Reminder:
    """Trigger a test failure when 3.10 is no longer supported."""

    def test_310_support_dropped(self):
        """
        Make changes when python3.10 support is dropped.

        sources/combined_source.py has some logic to support astropy<7, because
        python3.10 does not support astropy>=7. When python3.10 is no longer supported
        this can be cleaned up (and then require astropy>=7).
        """
        assert "3.10" in importlib.metadata.metadata("astromartini")["Requires-Python"]
