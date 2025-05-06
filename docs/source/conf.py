# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import noctis

project = "noctis"
copyright = "2023 Syngenta Group Co. Ltd."
author = "Nataliya Lopanitsyna"
show_authors = True
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
modindex_common_prefix = ["noctis."]

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.viewcode"]

templates_path = ["templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True


version = noctis.__version__
# The full version, including dev info
release = version.replace("_", "")

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "bizstyle"
html_static_path = ["static"]
html_logo = "static/logo.png"


html_sidebars = {
    "**": [
        "localtoc.html",
        "relations.html",
        "searchbox.html",
        "authors.html",
    ]
}
