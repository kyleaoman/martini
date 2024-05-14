Contributing to Martini
=======================

Contributions to Martini should come as pull requests submitted to the [GitHub repository](https://github.com/kyleaoman/martini).

Contributions are always welcome, but you should make sure of the following:

+ Your contributions include type hints (`*.pyi` stub files are used for these).
+ Your contributions are formatted with the `black` formatter.
+ Your contributions do not have formatting or style issues identified by `flake8`.
+ Your contributions pass `mypy` checks and `stubtest` checks.
+ Your contributions pass all unit tests in `/tests`.
+ Your contributions add unit tests for new functionality.
+ Your contributions are documented fully under `/docs`.

Some brief quickstart-style notes are included below, but are not intended to replace consulting the documentation of each relevant toolset. We recognize that this can seem daunting to users new to collaborative development. Don't hesitate to get in touch for help if you want to contribute!

Black style
-----------

You can install the `black` formatter with `pip install black`. To check your copy of the repository you can then run `black --check --diff` in the same directory as the `setup.py` file. A message like `All done! ✨ 🍰 ✨` indicates that your working copy passes the checks, while `Oh no! 💥 💔 💥` indicates problems are present. You can also use `black` to automatically edit your copy of the repository to comply with the style rules by running `black` in the same directory as `setup.py`. Don't forget to commit any changes it makes. In most cases, code formatted with `black` will pass the `flake8` checks.

Flake8 style guide enforcement
------------------------------

You can install the `flake8` style guide compliance checker with `pip install flake8`. Contributions will be checked on github on python versions `3.8` to `3.12`. To check your copy of the repository for compliance with the [PEP8](https://peps.python.org/pep-0008/) style guide, run `flake8` in the `pyproject.toml` file. No message indicates success (if unsure, you can check the return code with `flake8;echo $?`, in this case a value of 0 indicates success). In case of failure, any errors and warnings will be printed. In general these should be addressed by editing the code to be compliant (running `black` will resolve most issues automatically). In some rare cases it may be appropriate to ignore an error by editing the `/.flake8` file. Note that the maximum line length for the project is set to 90 characters.

MyPy type checking
------------------

You can install the `mypy` static type checker with `pip install mypy`. In the same directory as the `pyproject.toml` file, run `mypy --install-types --non-interactive`. This will check for identifiable type conflicts based on the type hints in the stub (`*.pyi`) files, and type hints for external packages when available. Type hint coverage in the wider python ecosystem is still patchy so these checks are not exhaustive, but occasionally catch some inconsistencies.

Stubtest
--------

The `stubtest` utility is included in `mypy` (see above). It is used to check that declarations in the stub files (`*.pyi`) are consistent with those in the source code files. The command to run `stubtest` is included in the `run_stubtest` script in the same directory as the `pyproject.toml` file.

Pytest unit testing
-------------------

You can install the `pytest` unit testing toolkit with `pip install pytest`. Running the full test suite requires the dependencies in `requirements.txt` and also those in `optional_requirements.txt`. You can then run `pytest` in the same directory as the `pyproject.toml` file to run the existing unit tests. Any test failures will report detailed debugging information. Note that the tests on github are run with python versions `3.8`, `3.9`, `3.10`, `3.11` and `3.12` and the latest PyPI releases of the relevant dependencies (`astropy`, etc.). To run only tests in a specific file, you can do e.g. `pytest tests/test_creation.py`. The tests to be run can be further narrowed down with the `-k` argument to `pytest` (see `pytest --help`).

Documentation
-------------

The API documentation is built automatically from the docstrings of classes, functions, etc. in the source files. These follow the NumPy-style format. At a minimum all public (i.e. not starting in `_`) modules, functions, classes, methods, etc. should have an appropriate docstring, and preferably all functions and classes should be appropriately documented. Developers may wish to build a local copy of the documentation that includes everything (e.g. private member functions, abstract base classes, etc.). This can be done by adding the following to `/docs/source/conf.py`:

    autodoc_default_options = {
        "members": True,
        "undoc-members": True,
        "private-members": True
    }


In addition to this there is "narrative documentation" that should describe the features of the code. The docs are built with `sphinx` and use the "ReadTheDocs" theme. If you have the dependencies installed (check `/docs/requirements.txt`) you can build the documentation locally with `make html` in the `/docs` directory. Opening the `/docs/index.html` file with a browser will then allow you to browse the documentation and check your contributions.