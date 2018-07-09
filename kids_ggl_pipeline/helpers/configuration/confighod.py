from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import stats

from .core import *
from ...halomodel.hod import relations, scatter
from ...sampling.priors import (
    fixed_priors, free_priors, nargs as prior_nargs, valid_priors)


def append_setup(parameters, nparams, setup):
    for i in setup:
        parameters[0].append(i)
    nparams.append(len(setup[0]))
    return parameters, nparams


def flatten_parameters(parameters):
    flat = [[] for p in parameters]
    nparams = [len(p) for p in parameters[0]]
    # first the four sets for the priors
    for i, params in enumerate(parameters):
        for par in params:
            for p in par:
                flat[i].append(p)
    flat = [np.array(i) for i in flat]
    return flat, nparams


def hod_entries(line, section, names, parameters, priors, starting):
    """
    `line` should be a `configline` object
    """
    if line.words[0] == 'name':
        names, parameters, priors = hod_function(
            names, parameters, priors, section, line)
    else:
        names, parameters, priors, starting = \
            hod_parameters(names, parameters, priors, starting, line)
    return names, parameters, priors, starting


def hod_function(names, parameters, priors, section, line):
    if 'mor' in section:
        parameters[0].append(getattr(relations, line.words[1]))
    elif 'scatter' in section:
        parameters[0].append(getattr(scatter, line.words[1]))
    parameters[1].append(0)
    parameters[2].append(-np.inf)
    parameters[3].append(np.inf)
    if len(line.words) == 3:
        names.append(line.words[2])
    else:
        names.append(':'.join([section.name, line.words[1]]))
    priors.append('function')
    return names, parameters, priors


def hod_parameters(names, parameters, priors, starting, line):
    """
    To deal with parameters with priors (including fixed values)

    `line` should be a `configline` object

    need to consider `join` instances here.
    """
    words = line.words
    if words[0] in ('name', 'function'):
        priors.append('function')
    elif words[1] in ('array', 'read'):
        priors.append(words[1])
    elif words[1] in fixed_priors:
        priors.append('fixed')
    elif words[1] in valid_priors:
        priors.append(words[1])
    else:
        assert prior_is_valid(line)
    names.append(words[0])

    if priors[-1] == 'array':
        parameters[0].append(np.array(words[2].split(','), dtype=float))
    elif priors[-1] == 'read':
        parameters[0].append(
            np.loadtxt(words[2], usecols=','.split(words[3])).T)
    elif priors[-1] in fixed_priors:
        parameters[0].append(float(words[2]))
    elif prior_nargs[priors[-1]] > 0:
        parameters[0].append(float(words[2]))
    else:
        parameters[0].append(-1)

    if priors[-1] in fixed_priors or priors[-1] in ('exp', 'jeffrey'):
        parameters[1].append(-1)
        starting = starting_values(starting, parameters, line)
    else:
        if prior_nargs[priors[-1]] == 2:
            parameters[1].append(float(words[3]))
            if len(words) > 5:
                parameters[2].append(float(words[4]))
                parameters[3].append(float(words[5]))
        else:
            parameters[1].append(-1)
            if prior_nargs[priors[-1]] == 1 and len(words) > 4:
                parameters[2].append(float(words[3]))
                parameters[3].append(float(words[4]))
            elif prior_nargs[priors[-1]] == 0 and len(words) > 3:
                parameters[2].append(float(words[2]))
                parameters[3].append(float(words[3]))
        if priors[-1] == 'uniform':
            parameters[2].append(parameters[0][-1])
            parameters[3].append(parameters[1][-1])
        starting = starting_values(starting, parameters, line)
    if len(parameters[2]) < len(parameters[1]):
        parameters[2].append(-np.inf)
        parameters[3].append(np.inf)

    return names, parameters, priors, starting


def ingredients(ingr, words):
    assert words[1] in ('True', 'False'), \
        'Value {1} for parameter {0} in hod/ingredients not valid.' \
        ' Must be True or False'.format(*(words))
    ingr[words[0]] = (words[1] == 'True')
    return ingr


def observables(words):
    binning = np.array(words[1].split(','), dtype=float)
    means = np.array(words[2].split(','), dtype=float)
    return [binning[:-1], binning[1:], means]


def prior_is_valid(line):
    assert line.words[1] in valid_priors, \
        'prior {1} not valid; must be one of {0}'.format(
            valid_priors, line.words[1])
    return True


def starting_values(starting, parameters, line):
    words = line.words
    prior = words[1]
    if prior in fixed_priors:
        return starting
    # these ones take two parameters
    if prior in ('lognormal', 'normal', 'uniform'):
        if len(words) in (5,7):
            starting.append(float(words[-1]))
        elif prior == 'normal':
            starting.append(np.random.normal(
                parameters[0][-1], parameters[1][-1], 1)[0])
        elif prior == 'lognormal':
            starting.append(10**np.random.normal(
                np.log10(parameters[0][-1]),
                np.log10(parameters[1][-1]), 1)[0])
        elif prior == 'uniform':
            starting.append(np.random.uniform(
                parameters[0][-1], parameters[1][-1], 1)[0])
    # these take one parameter
    elif prior in ('student',):
        if len(words) in (4,6):
            starting.append(float(words[-1]))
        if prior == 'student':
            starting.append(stats.t.rvs(float(line.words[2]), 1))
    # these take no parameters
    elif prior in ('exp', 'jeffrey',):
        if len(words) in (3,5):
            starting.append(float(words[-1]))
        if prior == 'exp':
            satrting.append(stats.expon.rvs(1))
        elif prior == 'jeffrey':
            # until we have a random number generator with this prior
            starting.append(np.random.uniform(
                parameters[0][-1], parameters[1][-1], 1)[0])
    return starting

