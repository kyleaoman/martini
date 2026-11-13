"""Test that demo function is working."""

from martini._demo import demo, demo_source


def test_demo(tmp_path_factory):
    """Check that demo completes and produces output files."""
    tp = tmp_path_factory.mktemp("demo")
    filenames = {
        "cubefile": str(tp / "testcube.fits"),
        "beamfile": str(tp / "testbeam.fits"),
        "hdf5file": str(tp / "testcube.hdf5"),
    }
    demo(**filenames)


def test_demo_source():
    """Check that demo source created and contains particles."""
    N = 100
    source = demo_source(N=N)
    assert source.mHI_g.size == N
