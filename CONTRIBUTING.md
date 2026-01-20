Contributing to Martini
=======================

Contributions to Martini should come as pull requests submitted to the [GitHub repository](https://github.com/kyleaoman/martini).

Contributions are always welcome, but you should make sure of the following:

+ Your contributions include type hints (in function and method declarations).
+ Your contributions are formatted with the `ruff` formatter.
+ Your contributions pass `ruff` style checks.
+ Your contributions do not have formatting or style issues identified by `flake8`.
+ Your contributions are covered by tests and pass all unit tests in `/tests`.
+ Your contributions add unit tests for new functionality.
+ Your contributions are documented with docstrings and their style passes `numpydoc lint` checks.
+ Your contributions are documented fully under `/docs`.

Some brief quickstart-style notes are included below, but are not intended to replace consulting the documentation of each relevant toolset. We recognize that this can seem daunting to users new to collaborative development. Don't hesitate to get in touch for help if you want to contribute!

Ruff
----

You can install the `ruff` linter with `pip install ruff`. To check that your copy of the repository conforms to style rules you can run `ruff check` in the same directory as the `pyproject.toml` file. A message like `All tests passed!` indicates that your working copy passes the checks, otherwise a list of problems is given. Some might be automatically fixable with `ruff check --fix`. Don't forget to commit any automatic fixes.

`ruff` is also used to enforce code formatting, you can check this with `ruff format --check` and automatically format your copy of the code with `ruff format`. Again remember to commit any automatically formatted files.

Flake8 style guide enforcement
------------------------------

You can install the `flake8` style guide compliance checker with `pip install flake8`. Contributions will be checked on github on `python3.12`. To check your copy of the repository for compliance with the [PEP8](https://peps.python.org/pep-0008/) style guide, run `flake8` in the same directory as the `pyproject.toml` file. No message indicates success (if unsure, you can check the return code with `flake8;echo $?`, in this case a value of 0 indicates success). In case of failure, any errors and warnings will be printed. In general these should be addressed by editing the code to be compliant (running `black` will resolve most issues automatically). In some rare cases it may be appropriate to ignore an error by editing the `/.flake8` file. Note that the maximum line length for the project is set to 90 characters.

MyPy type checking
------------------

You can install the `mypy` static type checker with `pip install mypy`. In the same directory as the `pyproject.toml` file, run `mypy --install-types --non-interactive`. This will check for identifiable type conflicts based on the type hints, and type hints for external packages when available. Type hint coverage in the wider python ecosystem is still patchy so these checks are not exhaustive, but do catch some inconsistencies.

Pytest unit testing
-------------------

You can install the `pytest` unit testing toolkit with `pip install pytest`. Running the full test suite requires the dependencies in `requirements.txt` and also those in `optional_requirements.txt`. You can then run `pytest` in the same directory as the `pyproject.toml` file to run the existing unit tests. Any test failures will report detailed debugging information. Note that the tests on github are run with python versions `3.10`, `3.11`, `3.12`, `3.13` and `3.14` and the latest PyPI releases of the relevant dependencies (`astropy`, etc.) available on each python version. To run only tests in a specific file, you can do e.g. `pytest tests/test_creation.py`. The tests to be run can be further narrowed down with the `-k` argument to `pytest` (see `pytest --help`).

Test coverage is analysed with [Codecov](https://about.codecov.io/). First the `pytest-cov` module checks which lines of code are run during the execution of the test suite and produces a report. Codecov simply presents this report in a convenient format and adds some integration with Github. While having all code lines run (called "100% test coverage"), or nearly, does not guarantee good tests, it is still a useful benchmark. Pull requests opened on the repository will produce a test coverage report once the test suite finishes running. This will flag any new code lines not covered by tests, or previously covered lines that are no longer covered, etc. You should look at this and evaluate whether it reveals additional useful test cases that you hadn't considered. You can also generate a code coverage report locally with the `pytest-cov` extension for `pytest`.

Documentation
-------------

The API documentation is built automatically from the docstrings of classes, functions, etc. in the source files. These follow the NumPy-style format. All modules, functions, classes, methods, etc. should have an appropriate docstring. Developers may wish to build a local copy of the documentation that includes everything (e.g. private member functions, abstract base classes, etc.). This can be done by adding the following to `docs/source/conf.py`:

    autodoc_default_options = {
        "members": True,
        "undoc-members": True,
        "private-members": True
    }


In addition to this there is "narrative documentation" that should describe the features of the code. The docs are built with `sphinx` and use the "ReadTheDocs" theme. If you have the dependencies installed (check `pyproject.toml` or just `pip install .[docs]` in the project root directory) you can build the documentation locally with `make html` in the `docs` directory. Opening the `docs/index.html` file with a browser will then allow you to browse the documentation and check your contributions. Style issues in the documentation source files can be checked with `sphinx-lint --max-line-length 90 -e all docs/source/` (run from the project root directory).

Docstrings
----------

Ruff currently has limited support for [numpydoc](https://numpydoc.readthedocs.io/en/latest/index.html)-style docstrings. To run additional checks on docstrings use `numpydoc lint **/*.py` in the same directory as the `pyproject.toml` file. As more style rules become supported by `ruff` this will hopefully be phased out