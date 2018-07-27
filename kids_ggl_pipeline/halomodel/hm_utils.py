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
from . import nfw, nfw_stack, satellites, halo, halo_2, halo_2_mc, halo_sz
try:
    import models
except ImportError:
    pass

# working directory
try:
    import models
except ImportError:
    pass



def read_config(config_file, version='0.5.7'):
    valid_types = ('normal', 'lognormal', 'uniform', 'exp',
                   'fixed', 'read', 'function')
    path = ''
    params = []
    param_types = []
    prior_types = []
    val1 = []
    val2 = []
    val3 = []
    val4 = []
    starting = []
    hm_functions = []
    meta_names = []
    fits_format = []
    njoin = 1
    nparams = 0
    join = [[]]
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'model':
            model = line[1].split('.')
            model = [line[1], read_function(*model)]
        elif line[0] == 'path':
            if len(line) > 1:
                path = line[1]
                if path[-1] == '/':
                    path = path[:-1]
            else:
                path = ''
        # also read param names
        elif line[0] == 'hm_param':
            if line[2] not in valid_types:
                msg = 'ERROR: Please provide only valid prior types in the' \
                      ' parameter file ({0}). Value {1} is invalid.' \
                      ' Valid types are {2}'.format(
                            config_file, line[1], valid_types)
                print(msg)
                sys.exit()
            params.append(line[1])
            prior_types.append(line[2])
            if line[2] == 'function':
                val1.append(read_function(*(line[3].split('.'))))
                val2.append(-1)
            elif line[2] == 'read':
                filename = line[3]
                if path:
                    filename = filename.replace('<path>', path)
                v = loadtxt(filename, usecols=(int(line[4]),))
                if not iterable(v):
                    v = array([v])
                val1.append(v)
                val2.append(-1)
            else:
                val1.append(float(line[3]))
                if len(line) > 4:
                    try:
                        val2.append(float(line[4]))
                    except:
                        val2.append(-1)
                else:
                    val2.append(-1)
            if line[2] in ('normal', 'lognormal'):
                if len(line) > 5:
                    val3.append(float(line[5]))
                    val4.append(float(line[6]))
                else:
                    val3.append(-inf)
                    val4.append(inf)
                starting.append(float(line[3]))
            else:
                val3.append(-inf)
                val4.append(inf)
            if line[2] == 'uniform':
                try:
                    starting.append(float(line[-1]))
                except:
                    starting.append(float(line[-2]))
            # these are to enable automatic handling of the number of bins
            if line[-1] == 'join{0}'.format(njoin):
                join[-1].append(nparams)
            elif line[-1] == 'join{0}'.format(njoin+1):
                join[-1] = array(join[-1])
                join.append([])
                join[-1].append(nparams)
                njoin += 1
            nparams += 1
            # to be able to handle comma-separated hm_params
            if not hasattr(val1[-1], '__iter__'):
                val1[-1] = [val1[-1]]
            val2[-1] = [val2[-1]]
            val3[-1] = [val3[-1]]
            val4[-1] = [val4[-1]]
        elif line[0] == 'hm_params':
            if line[2] != 'fixed':
                msg = 'ERROR: Arrays can only contain fixed values.'
                print(msg)
                exit()
            param_types.append(line[0])
            params.append(line[1])
            prior_types.append(line[2])
            val1.append(array(line[3].split(','), dtype=float))
            val2.append([-1])
            val3.append([-inf])
            val4.append([inf])
            nparams += 1
        elif line[0] == 'hm_functions':
            # check if there are comments at the end first
            if '#' in line:
                j = line.index('#')
            else:
                j = len(line)
            # how many entries before the comments?
            if j == 2:
                f = [read_function(i) for i in line[1].split(',')]
            else:
                f = [read_function(line[1]+'.'+i)
                     for i in line[2].split(',')]
            for i in f:
                hm_functions.append(i)
        elif line[0] == 'hm_output':
            fmt = line[2].split(',')
            n = int(fmt[0]) if len(fmt) == 2 else 0
            fmt = fmt[-1]
            if n == 0:
                meta_names.append(line[1])
                fits_format.append(line[2])
            else:
                for i in xrange(1, n+1):
                    meta_names.append('{0}{1}'.format(line[1], i))
                    fits_format.append(fmt)
    if len(hm_functions) > 0:
        hm_functions = (func for func in hm_functions)
    if njoin == 1 and len(join[0]) == 0:
        join = None

    out = (model, array(params), array(param_types), array(prior_types),
           make_array(val1), make_array(val2), make_array(val3),
           make_array(val4), join, hm_functions, array(starting),
           array(meta_names), fits_format)
    return out


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
    elif module == 'halo_sz':
        function = getattr(halo_sz, function)
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
