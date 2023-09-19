from martini import demo
import os


class TestDemo:
    def test_demo(self):
        filenames = dict(
            cubefile="pytest_testcube.fits",
            beamfile="pytest_testbeam.fits",
            hdf5file="pytest_testcube.hdf5",
        )
        try:
            demo(**filenames)
        finally:
            for filename in filenames.values():
                if os.path.exists(filename):
                    os.remove(filename)
