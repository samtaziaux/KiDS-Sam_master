import os
from glob import glob
#from ConfigParser import SafeConfigParser

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
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if line[0] == 'model':
            model = read_function(line[1])
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
                val1.append(read_function(line[3]))
                val2.append(-1)
            if line[2] == 'read':
                filename = os.path.join(path, line[3])
                val1.append(loadtxt(filename, usecols=(int(line[4]),)))
                val2.append(-1)
            else:
                val1.append(float(line[3]))
                if len(line) > 4:
                    val2.append(float(line[4]))
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
                starting.append(float(line[-1]))
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
    out = (model, params, param_types, prior_types,
           val1, val2, val3, val4, hm_functions,
           starting, meta_names, fits_format)
    return out

def read_function(function):
    function_path = function.split('.')
    if len(function_path) < 2:
        msg = 'ERROR: the parent module(s) must be given with'
        msg += 'a function'
        print msg
        exit()
    else:
        module = __import__(function)
        for attr in function_path:
            func = getattr(func, attr)
    return func
