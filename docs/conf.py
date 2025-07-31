import os 
import sys
import numpy

project = 'WxSysLib'
copyright = '2025, 21st Century Weather'
version = '1.0'
release = '1.0.0'

#extensions = [
#    'sphinx.ext.autodoc',
#    'sphinx.ext.napoleon', # For Google/NumPy style docstrings
#    'sphinx.ext.intersphinx',
#    'sphinx.ext.viewcode',
#]

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # Adjust as needed
sys.path.insert(0, project_root)

extensions = [
    'sphinx.ext.autodoc',       # Core for documenting Python objects
    'sphinx.ext.napoleon',      # For Google/NumPy style docstrings
    'sphinx.ext.doctest',       # For runnable examples
    'sphinx.ext.intersphinx',   # If you link to other Sphinx docs (e.g., Python stdlib)
    'sphinx.ext.todo',          # For TODO items
    'sphinx.ext.coverage',      # To check documentation coverage
    'sphinx.ext.mathjax',       # If you have math in your docstrings/docs
    'sphinx.ext.ifconfig',      # Conditional content
    'sphinx.ext.viewcode',      # Links to source code
    'sphinx.ext.autosectionlabel', # Allows referencing sections by title
    'sphinx_rtd_theme',         # Common theme for Read the Docs
]


# The root document.
root_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Napoleon settings (if you're using Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme' # Or another theme like 'pydata_sphinx_theme'

