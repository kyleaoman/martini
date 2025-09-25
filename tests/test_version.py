import importlib.metadata
import martini


class TestVersion:
    def test_code_version(self):
        """
        Check that code version matches pyproject.toml version.
        """
        assert importlib.metadata.version("astromartini") == martini.__version__

    def test_codemeta_version(self):
        """
        Check that codemeta version matches version file.
        """
        with open("codemeta.json") as f:
            codemeta_content = f.read()
        assert (
            f'"version": "{importlib.metadata.version("astromartini")}",'
            in codemeta_content
        )
