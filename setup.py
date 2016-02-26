import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Taken from the Python docs:
# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='kids_ggl_pipeline',
      version='1.1.0',
      description='KiDS Galaxy-Galaxy Lensing Pipeline',
      author='Margot Brouwer, Andrej Dvornik, Cristobal Sifon',
      author_email='dvornik@strw.leidenuniv.nl',
      long_description=read('README.md'),
      url='https://github.com/KiDS-WL/KiDS-GGL',
      packages=['kids_ggl_pipeline',
                'kids_ggl_pipeline/esd_production',
                'kids_ggl_pipeline/halomodel',
                'kids_ggl_pipeline/sampling'],
      package_data={'': ['demo/*', 'README.md']},
      scripts=['bin/kids_ggl'],
      install_requires=['astropy>=1.1.0',
                        'emcee>=2.1.0',
                        'hmf==1.7.0',
                        'mpmath>=0.19',
                        'numpy>=1.5.0']
      )
