"""Test for compliance with the FITS specification."""

from astropy.io import fits


class TestFITSCompliant:
    """Test for compliance with the FITS specification."""

    def test_cube_compliant(self, m, tmp_path_factory):
        """Check that written cube complies with FITS standards enfored by astropy."""
        filename = str(tmp_path_factory.mktemp("cube") / "cube.fits")
        m.write_fits(filename, overwrite=True)
        with fits.open(filename) as hdul:
            hdul.verify("exception")  # errors here if failing

    def test_beam_compliant(self, m, tmp_path_factory):
        """Check that written beam complies with FITS standards enforced by astropy."""
        filename = str(tmp_path_factory.mktemp("beam") / "beam.fits")
        m.write_beam_fits(filename, overwrite=True)
        with fits.open(filename) as hdul:
            hdul.verify("exception")  # errors here if failing
