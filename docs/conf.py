import sys
import os
from importlib.metadata import version as pkg_version

# Ensure the src directory is on sys.path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

project = 'MiniLLMLib'
copyright = '2025, Quentin Feuillade--Montixi'
author = 'Quentin Feuillade--Montixi'

# Use setuptools_scm to get the version from git tags
try:
    release = pkg_version("minillmlib")
    version = release  # Sphinx expects both to be strings
except Exception:
    release = version = 'unknown'

extensions = [
    'myst_parser',  # Markdown support
    # 'sphinx.ext.autodoc',  # Uncomment if you want API docs
    # 'sphinx.ext.napoleon', # Uncomment for Google/Numpy docstring style
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
