from __future__ import absolute_import, print_function

from glob import glob
import os
import re
import sys


def data_files_recursively(path):
    """
    Recursively find all files and subfolders for `data_files`.
    This is easily done in python 3.5+ but not so much in older versions.
    See https://goo.gl/WbKRDZ

    Not fully tested
    """
    try:
        return sorted(glob(os.path.join(path, '**'), recursive=True))
    except TypeError:
        pairs = []
        for root, dirnames, filenames in os.walk(path):
            files = [os.path.join(root, f) for f in filenames]
            pairs.append([root, files])
            print(pairs[-1])
            print()
    return pairs



def find_location(file):
    return os.path.abspath(os.path.dirname(file))


def find_version(filename):
    """Find package version
    The version should be defined as variable `__version__` (typically
    in an `__init__.py`). This way it only needs to be specified in
    one place.
    Copied from pip's setup.py,
    https://github.com/pypa/pip/blob/1.5.6/setup.py
    """
    version_file = read(filename)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read(filename):
    """Utility function to read the entire contents of a file
    Used mostly to pass the contents of a README file to
    `long_description` in the setup script.
    Copied from the Python docs
    """
    return open(filename).read()