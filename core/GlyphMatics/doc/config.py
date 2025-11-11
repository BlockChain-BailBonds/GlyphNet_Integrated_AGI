import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

project = 'GlyphMatics'
copyright = f'{datetime.now().year}, xxNine1Eightxx'
author = 'xxNine1Eightxx'
release = '0.1.0'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = "GlyphMatics Documentation"
