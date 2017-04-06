import os
import re
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))


#Taken from the Python docs:
#Utility function to read the README file.
#Used for the long_description.  It's nice, because now 1) we have a
#top level README file and 2) it's easier to type in the README file
#than to put a raw string in below
def read(fname):
    return open(os.path.join(here, fname)).read()


#this function copied from pip's setup.py
#https://github.com/pypa/pip/blob/1.5.6/setup.py
#so that the version is only set in the __init__.py and then read here
#to be consistent
def find_version(fname):
    version_file = read(fname)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='kids_ggl_pipeline',
      #version='1.1.1',
      version=find_version('kids_ggl_pipeline/__init__.py'),
      description='KiDS Galaxy-Galaxy Lensing Pipeline',
      author='Margot Brouwer, Andrej Dvornik, Cristobal Sifon',
      author_email='dvornik@strw.leidenuniv.nl',
      long_description=read('README.md'),
      url='https://github.com/KiDS-WL/KiDS-GGL',
      packages=['kids_ggl_pipeline',
                'kids_ggl_pipeline/esd_production',
                'kids_ggl_pipeline/halomodel',
                'kids_ggl_pipeline/sampling',
                'kids_ggl_pipeline/helpers',
                'kids_ggl_pipeline/The-wiZZ'],
      package_data={'': ['demo/*', 'README.md']},
      scripts=['bin/kids_ggl'],
      install_requires=['astropy>=1.2.0',
                        'emcee>=2.1.0',
                        'hmf==1.7.0',
                        'mpmath>=0.19',
                        'numpy>=1.5.0',
                        'scipy>=0.16.0',
                        'psutil==3.2.1']
      )
