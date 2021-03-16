from __future__ import absolute_import, print_function

import os
import re
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup



from kids_ggl_pipeline.helpers.setup_helpers import (
    data_files_recursively, find_location, find_version, read)


setup(name='kids_ggl_pipeline',
      version=find_version('kids_ggl_pipeline/__init__.py'),
      description='KiDS Galaxy-Galaxy Lensing Pipeline',
      author='Margot Brouwer, Andrej Dvornik, Cristobal Sifon',
      author_email='dvornik@strw.leidenuniv.nl',
      long_description=read('README.md'),
      url='https://github.com/KiDS-WL/KiDS-GGL',
      packages=['kids_ggl_pipeline',
                'kids_ggl_pipeline/esd_production',
                'kids_ggl_pipeline/halomodel',
                'kids_ggl_pipeline/hod',
                'kids_ggl_pipeline/sampling',
                'kids_ggl_pipeline/helpers',
                'kids_ggl_pipeline/helpers/configuration'],
      #package_data={'': ['demo/*', 'docs/*', 'README.md']},
      #data_files=[('.', ['LICENSE', 'README.md']),
                  #('demo', data_files_recursively('demo')),
                  #('docs', data_files_recursively('docs'))],
      include_package_data=True,
      scripts=['bin/kids_ggl'],
      install_requires=['astropy>=1.2.0',
                        'emcee>3.0',
                        'hmf>=3.1',
                        'mpmath>=0.19',
                        'numpy>=1.5.0',
                        'scipy>=0.16.0',
                        'healpy>1.13.0',
                        'psutil>=3.2.1',
                        'gitpython>=3.1',
                        'dill>=0.3'],
      zip_safe=False
      )

