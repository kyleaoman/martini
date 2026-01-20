"""Test for compliance with the FITS specification."""

import os
from astropy.io import fits


class TestFITSCompliant:
    """Test for compliance with the FITS specification."""

    def test_cube_compliant(self, m):
        """Check that written cube complies with FITS standards enfored by astropy."""
        filename = "test_cube_compliant.fits"
        try:
            m.write_fits(filename, overwrite=True)
            with fits.open(filename) as hdul:
                hdul.verify("exception")  # errors here if failing
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_beam_compliant(self, m):
        """Check that written beam complies with FITS standards enforced by astropy."""
        filename = "test_beam_compliant.fits"
        try:
            m.write_beam_fits(filename, overwrite=True)
            with fits.open(filename) as hdul:
                hdul.verify("exception")  # errors here if failing
        finally:
            if os.path.exists(filename):
                os.remove(filename)
