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
    meta_names = []
    fits_format = []
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if line[0] == 'data':
            datafiles = sorted(glob(line[1]))
            datacols = [int(i) for i in line[2].split(',')]
        elif line[0] == 'covariance':
            covfile = glob(line[1])
            if len(covfile) > 1:
                msg = 'More than one matching covariance filename'
                raise ValueError(msg)
            covfile = covfile[0]
            covcols = [int(i) for i in line[2].split(',')]
        elif line[0] == 'sampler':
            sampler = sorted(glob(line[1]))
        elif line[0] == 'nwalkers':
            nwalkers = glob(line[1])
        elif line[0] == 'nsteps':
            nsteps = glob(line[1])
        elif line[0] == 'nburn':
            nburn = glob(line[1])
        elif line[0] == 'thin':
            thin = glob(line[1])
        elif line[0] == 'k':
            k = glob(line[1])
        elif line[0] == 'threads':
            threads = glob(line[1])
        elif line[0] == 'sampler_type':
            sampler_type = glob(line[1])

    out = (datafiles, datacols, covfile, covcols,
           sampler,nwalkers,nsteps,nburn,thin,k,threads,
           sampler_type)
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
