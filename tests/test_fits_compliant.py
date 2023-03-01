import pytest
import os
from astropy.io import fits


class TestFITSCompliant:
    @pytest.mark.parametrize("channels", ("frequency", "velocity"))
    def test_cube_compliant(self, m, channels):
        filename = "test_cube_compliant.fits"
        try:
            m.write_fits(filename, overwrite=True, channels=channels)
            with fits.open(filename) as hdul:
                hdul.verify("exception")  # errors here if failing
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    @pytest.mark.parametrize("channels", ("frequency", "velocity"))
    def test_beam_compliant(self, m, channels):
        filename = "test_beam_compliant.fits"
        try:
            m.write_beam_fits(filename, overwrite=True, channels=channels)
            with fits.open(filename) as hdul:
                hdul.verify("exception")  # errors here if failing
        finally:
            if os.path.exists(filename):
                os.remove(filename)
