import imp
import os
from glob import glob
from numpy import array, inf, loadtxt

#import sys
#sys.path.append('halomodel')

# local
import nfw, nfw_stack, satellites, halo


def read_config(config_file, version='0.5.7'):
    valid_types = ('normal', 'lognormal', 'uniform', 'exp',
                   'fixed', 'read', 'function')
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
            model = read_function(*model)
        # also read param names - follow the satellites Early Science function
        elif line[0] == 'hm_param':
            if line[2] not in valid_types:
                msg = 'ERROR: Please provide only valid prior types in the'
                msg += ' parameter file (%s). Value %s is invalid.' \
                       %(paramfile, line[1])
                msg = ' Valid types are %s' %valid_types
                print msg
                exit()
            params.append(line[1])
            prior_types.append(line[2])
            if line[2] == 'function':
                val1.append(read_function(*(line[3].split('.'))))
                val2.append(-1)
            elif line[2] == 'read':
                filename = line[3]
                val1.append(loadtxt(filename, usecols=(int(line[4]),)))
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
        elif line[0] == 'hm_params':
            if line[2] != 'fixed':
                msg = 'ERROR: Arrays can only contain fixed values.'
                print msg
                exit()
            param_types.append(line[0])
            params.append(line[1])
            prior_types.append(line[2])
            val1.append(array(line[3].split(','), dtype=float))
            val2.append(-1)
            val3.append(-inf)
            val4.append(inf)
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
            meta_names.append(line[1].split(','))
            fits_format.append(line[2].split(','))
    if len(hm_functions) > 0:
        hm_functions = (func for func in hm_functions)
    if njoin == 1 and len(join[0]) == 0:
        join = None
    out = (model, array(params), array(param_types), array(prior_types),
           array(val1), array(val2), array(val3), array(val4), join,
           hm_functions, array(starting), array(meta_names), fits_format)
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
    print function
    #pickle.dumps(function)
    #print 'Pickled!'
    return function
