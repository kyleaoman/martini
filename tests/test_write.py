import os
from astropy.io import fits
import h5py


class TestWrite:
    def test_write_fits_freqchannels(self, m):
        filename = "cube_f.fits"
        m.write_fits(filename, channels="frequency")
        with fits.open(filename) as f:
            hdr = f[0].header
        try:
            assert "FREQ" in hdr["CTYPE3"]
        finally:
            os.remove(filename)

    def test_write_fits_velchannels(self, m):
        filename = "cube_v.fits"
        m.write_fits(filename, channels="velocity")
        with fits.open(filename) as f:
            hdr = f[0].header
        try:
            assert "VOPT" in hdr["CTYPE3"]
        finally:
            os.remove(filename)

    def test_write_hdf5_freqchannels(self, m):
        filename = "cube_f.hdf5"
        m.write_hdf5(filename, channels="frequency")
        try:
            with h5py.File(filename, "r") as f:
                assert f["FluxCube"].attrs["VProjType"] == "FREQ"
        finally:
            os.remove(filename)

    def test_write_hdf5_velchannels(self, m):
        filename = "cube_v.hdf5"
        m.write_hdf5(filename, channels="velocity")
        try:
            with h5py.File(filename, "r") as f:
                assert f["FluxCube"].attrs["VProjType"] == "VOPT"
        finally:
            os.remove(filename)

    def test_write_fits_beam_freqchannels(self, m):
        filename = "beam_f.fits"
        m.write_beam_fits(filename, channels="frequency")
        with fits.open(filename) as f:
            hdr = f[0].header
        try:
            print(hdr)
            assert "FREQ" in hdr["CTYPE3"]
        finally:
            os.remove(filename)

    def test_write_fits_beam_velchannels(self, m):
        filename = "beam_v.fits"
        m.write_beam_fits(filename, channels="velocity")
        with fits.open(filename) as f:
            hdr = f[0].header
        try:
            print(hdr)
            assert "VOPT" in hdr["CTYPE3"]
        finally:
            os.remove(filename)
