import pytest
import os


class TestWrite:
    def test_write_fits_freqchannels(self, m):
        m.write_fits("cube_f.fits")
        raise AssertionError  # test something here
        os.remove("cube_f.fits")

    @pytest.mark.skip
    def test_write_fits_velchannels(self, m):
        m.write_fits("cube_v.fits", channels="velocity")

    @pytest.mark.skip
    def test_write_hdf5_freqchannels(self, m):
        m.write_hdf5("cube_f.hdf5")

    @pytest.mark.skip
    def test_write_hdf5_velchannels(self, m):
        m.write_hdf5('cube_v.hdf5', channels="velocity")
