# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import martini

# -- Project information -----------------------------------------------------

project = "Martini"
copyright = "2018, Kyle Oman"
author = "Kyle Oman"

# The full version, including alpha/beta/rc tags
release = martini.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = [".rst", ".md"]
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


def setup(app):
    app.add_css_file("custom.css")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

intersphinx_mapping = dict(
    multiprocess=("https://multiprocess.readthedocs.io/en/latest/", None),
    h5py=("https://docs.h5py.org/en/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    astropy=("https://docs.astropy.org/en/stable/", None),
    swiftgalaxy=("https://swiftgalaxy.readthedocs.io/en/latest", None),
    swiftsimio=("https://swiftsimio.readthedocs.io/en/latest", None),
)
