from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import imp
import numpy as np
import os
import sys
from glob import glob
from numpy import array, inf, iterable, loadtxt

if sys.version_info[0] == 3:
    xrange = range

# local
from . import nfw, nfw_stack, satellites, halo, halo_2, halo_2_mc
try:
    import models
except ImportError:
    pass

# working directory
try:
    import models
except ImportError:
    pass


def read_function(module, function):
    #import pickle
    #print module, function,
    #module = imp.load_module(module, *imp.find_module(module))
    #print module.__file__,
    #function = getattr(module, function)
    # this works for now but is of very limited functionality
    if module == 'satellites':
        function = getattr(satellites, function)
    elif module == 'nfw':
        function = getattr(nfw, function)
    elif module == 'nfw_stack':
        function = getattr(nfw_stack, function)
    elif module == 'halo':
        function = getattr(halo, function)
    elif module == 'halo_2':
        function = getattr(halo_2, function)
    elif module == 'halo_2_mc':
        function = getattr(halo_2_mc, function)
    elif module == 'models':
        function = getattr(models, function)
    print('Successfully imported {0}'.format(function))
    #pickle.dumps(function)
    #print 'Pickled!'
    return function


def depth(iterable):
    return sum(1 + depth(item) if hasattr(item, '__iter__') else 1
               for item in iterable)


def make_array(val):
    val = array(val)
    for i, v in enumerate(val):
        if len(v) == 1 and depth(v) == 1:
            val[i] = v[0]
    return val

