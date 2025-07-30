project = 'WxSysLib'
copyright = '2025, 21st Century Weather'
version = '1.0'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # For Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]
# The root document.
root_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme' # Or another theme like 'pydata_sphinx_theme'

