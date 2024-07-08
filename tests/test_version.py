from martini.__version__ import __version__


class TestVersion:
    def test_pyproject_version(self):
        """
        Check that pyproject version matches version file.
        """
        with open("pyproject.toml") as f:
            pyproject_content = f.read()
        assert 'version = "' + __version__ + '"' in pyproject_content

    def test_codemeta_version(self):
        """
        Check that codemeta version matches version file.
        """
        with open("codemeta.json") as f:
            codemeta_content = f.read()
        assert '"version": "' + __version__ + '",' in codemeta_content
