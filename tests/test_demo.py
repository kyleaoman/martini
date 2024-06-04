from martini._demo import demo, demo_source
import os


class TestDemo:
    def test_demo(self):
        """
        Check that demo completes and produces output files.
        """
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

    def test_demo_source(self):
        """
        Check that demo source created and contains particles.
        """
        N = 100
        source = demo_source(N=N)
        assert source.mHI_g.size == N
