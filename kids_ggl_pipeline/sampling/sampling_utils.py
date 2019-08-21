from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import six
import sys
from glob import glob

if sys.version_info[0] == 2:
    input = raw_input
    range = xrange


def get_autocorrelation(sampler):
    return


def initialize_fail_value(metadata, value=9999):
    fail_value = []
    for m in metadata:
        shape = list(m.shape)
        shape.remove(max(shape))
        fail_value.append(np.zeros(shape))
    # the last numbers are data chi2, lnLdata
    for i in range(3):
        fail_value.append(value)
    return fail_value


def initialize_metadata(options, output, shape):
    meta_names, fits_format = output
    # this assumes that all parameters are floats -- can't imagine a
    # different scenario
    metadata = [[] for m in meta_names]
    for j, fmt in enumerate(fits_format):
        n = 1 if len(fmt) == 1 else int(fmt[0])
        # is this value a scalar?
        if len(fmt) == 1:
            size = options['nwalkers'] * options['nsteps'] \
                // options['thin']
        else:
            size = [options['nwalkers']*options['nsteps']//options['thin'],
                    int(fmt[:-1])]
            # only for ESDs. Note that there will be trouble if outputs
            # other than the ESD have the same length, so avoid them at
            # all cost.
            if options['exclude'] is not None \
                    and size[1] == shape[-1]+len(options['exclude']):
                size[1] -= len(options['exclude'])
        metadata[j].append(np.zeros(size))
    metadata = [np.array(m) for m in metadata]
    metadata = [m[0] if m.shape[0] == 1 else m for m in metadata]
    return metadata, meta_names, fits_format


def read_function(function):
    print('Reading function', function)
    function_path = function.split('.')
    if len(function_path) < 2:
        print('ERROR: the parent module(s) must be given with a function')
        exit()
    else:
        module = __import__(function)
        for attr in function_path:
            func = getattr(func, attr)
    return func


def setup_integrand(R, k=7):
    """
    These are needed for integration and interpolation and should always
    be used. k=7 gives a precision better than 1% at all radii

    """
    if R.shape[0] == 1:
        Rrange = np.logspace(np.log10(0.99*R.min()),
                                np.log10(1.01*R.max()), 2**k)
        # this assumes that a value at R=0 will never be provided, which is
        # obviously true in real observations
        R = np.array([np.append(0, R)])
        Rrange = np.append(0, Rrange)
    else:
        Rrange = [np.logspace(np.log10(0.99*Ri.min()),
                                 np.log10(1.01*Ri.max()), 2**k)
                  for Ri in R]
        R = [np.append(0, Ri) for Ri in R]
        Rrange = [np.append(0, Ri) for Ri in Rrange]
        R = np.array(R)
        Rrange = np.array(Rrange)
    return R, Rrange

