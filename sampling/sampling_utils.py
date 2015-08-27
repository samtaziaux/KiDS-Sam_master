import os
from glob import glob
#from ConfigParser import SafeConfigParser

def read_config(config_file, version='0.5.7',
                path_data='', path_covariance=''):
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
        if len(line) == 0:
            continue
        if line[0] == 'path_data':
            path_data = line[1]
        if line[0] == 'data':
            datafiles = line[1]
            datacols = [int(i) for i in line[2].split(',')]
            if len(datacols) not in (2,3):
                msg = 'datacols must have either two or three elements'
                raise ValueError(msg)
        if line[0] == 'path_covariance':
            path_covariance = line[1]
        elif line[0] == 'covariance':
            covfile = line[1]
            covcols = [int(i) for i in line[2].split(',')]
            if len(covcols) not in (1,2):
                msg = 'covcols must have either one or two elements'
                raise ValueError(msg)
        elif line[0] == 'sampler_output':
            output = line[1]
            if output[-5:].lower() != '.fits' and \
                output[-4:].lower() != '.fit':
                output += '.fits'
        # all of this will have to be made more flexible to allow for
        # non-emcee options
        elif line[0] == 'sampler':
            sampler = line[1]
        elif line[0] == 'nwalkers':
            nwalkers = int(line[1])
        elif line[0] == 'nsteps':
            nsteps = int(line[1])
        elif line[0] == 'nburn':
            nburn = int(line[1])
        elif line[0] == 'thin':
            thin = int(line[1])
        # this k is only needed for mis-centred groups in my implementation
        # so maybe it should go in the hm_utils?
        elif line[0] == 'k':
            k = int(line[1])
        elif line[0] == 'threads':
            threads = int(line[1])
        elif line[0] == 'sampler_type':
            sampler_type = line[1]
    if path_data:
        datafiles = os.path.join(path_data, datafiles)
    datafiles = sorted(glob(datafiles))
    if path_covariance:
        covfile = os.path.join(path_covariance, covfile)
    covfile = glob(covfile)
    if len(covfile) > 1:
        msg = 'ambiguous covariance filename'
        raise ValueError(msg)
    covfile = covfile[0]

    out = (datafiles, datacols, covfile, covcols, output,
           sampler, nwalkers, nsteps, nburn, thin, k, threads,
           sampler_type)
    return out

def read_function(function):
    print 'Reading function', function
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
