import os
from astropy.io import fits
import h5py


class TestWrite:
    def test_write_fits_freqchannels(self, m):
        """
        Check that fits cube gets written with frequency channels.
        """
        filename = "cube_f.fits"
        try:
            m.write_fits(filename, channels="frequency")
            with fits.open(filename) as f:
                hdr = f[0].header
            assert "FREQ" in hdr["CTYPE3"]
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_write_fits_velchannels(self, m):
        """
        Check that fits cube gets written with velocity channels.
        """
        filename = "cube_v.fits"
        try:
            m.write_fits(filename, channels="velocity")
            with fits.open(filename) as f:
                hdr = f[0].header
            assert hdr["CTYPE3"] == "VRAD"
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_write_hdf5_freqchannels(self, m):
        """
        Check that hdf5 cube gets written with frequency channels.
        """
        filename = "cube_f.hdf5"
        try:
            m.write_hdf5(filename, channels="frequency")
            with h5py.File(filename, "r") as f:
                assert f["FluxCube"].attrs["VProjType"] == "FREQ"
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_write_hdf5_velchannels(self, m):
        """
        Check that hdf5 cube gets written with velocity channels.
        """
        filename = "cube_v.hdf5"
        try:
            m.write_hdf5(filename, channels="velocity")
            with h5py.File(filename, "r") as f:
                assert f["FluxCube"].attrs["VProjType"] == "VRAD"
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_write_fits_beam_freqchannels(self, m):
        """
        Check that fits beam cube gets written with frequency channels.
        """
        filename = "beam_f.fits"
        try:
            m.write_beam_fits(filename, channels="frequency")
            with fits.open(filename) as f:
                hdr = f[0].header
            assert "FREQ" in hdr["CTYPE3"]
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_write_fits_beam_velchannels(self, m):
        """
        Check that fits beam cube gets written with velocity channels.
        """
        filename = "beam_v.fits"
        try:
            m.write_beam_fits(filename, channels="velocity")
            with fits.open(filename) as f:
                hdr = f[0].header
            assert "VRAD" in hdr["CTYPE3"]
        finally:
            if os.path.exists(filename):
                os.remove(filename)
