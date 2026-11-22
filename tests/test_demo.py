"""Test that demo function is working."""

from martini import Martini
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


def test_demo_without_h5py(tmp_path, monkeypatch, capsys):
    """Check that demo skips hdf5 output if h5py is unavailable."""

    def write_hdf5(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'h5py'")

    monkeypatch.setattr(Martini, "write_hdf5", write_hdf5)
    demo(
        cubefile=str(tmp_path / "testcube.fits"),
        beamfile=str(tmp_path / "testbeam.fits"),
        hdf5file=str(tmp_path / "testcube.hdf5"),
    )
    assert "skipping hdf5 output demo" in capsys.readouterr().out
    assert not (tmp_path / "testcube.hdf5").exists()


def test_demo_source():
    """Check that demo source created and contains particles."""
    N = 100
    source = demo_source(N=N)
    assert source.mHI_g.size == N
