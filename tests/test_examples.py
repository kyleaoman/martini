import os
from martini.__version__ import __version__
import pytest


class TestExamples:
    def test_source_injection_example_version(self):
        """
        Check that the source injection notebook installs current version of martini.
        """
        with open("examples/martini_source_injection.ipynb") as f:
            nb_content = f.read()
        print(nb_content)
        assert (
            '"!{sys.executable} -m pip install \\"astromartini[tngsource]=='
            + __version__
            + '\\"'
            in nb_content
        )

    @pytest.mark.skipif(
        not os.path.isfile("examples/NGC_2841_NA_CUBE_THINGS.FITS"),
        reason="sample data not locally available",
    )
    def test_source_injection_example(self):
        pytest.importorskip(
            "nbmake", reason="nbmake (optional dependency) not available"
        )
        from nbmake.nb_run import NotebookRun
        from nbmake.pytest_items import NotebookFailedException
        import pathlib

        files_to_cleanup = (
            "examples/NGC_2841_AND_TNG50-1_99_737963.FITS",
            "examples/sourceinjectiondemo.fits",
            "examples/martini-cutout-grnr-TNG50-1-99-737963.npy",
            "examples/martini-cutout-TNG50-1-99-2122.hdf5",
        )

        assert os.path.isfile("examples/NGC_2841_NA_CUBE_THINGS.FITS")
        for file_to_cleanup in files_to_cleanup:
            if os.path.isfile(file_to_cleanup):
                os.remove(file_to_cleanup)
        nbr = NotebookRun(pathlib.Path("examples/martini_source_injection.ipynb"), 3600)
        try:
            result = nbr.execute()
            if result.error is not None:
                raise NotebookFailedException(result)
        finally:
            for file_to_cleanup in files_to_cleanup:
                if os.path.isfile(file_to_cleanup):
                    os.remove(file_to_cleanup)
