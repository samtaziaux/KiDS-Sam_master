from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .core import *
from ...halomodel.hod import relations, scatter


valid_priors = ('fixed', 'lognormal', 'normal', 'uniform')


def append_entries(section, parameters, priors, starting, line):
    """
    `line` should be a `configline` object
    """
    if section.name == 'hod/observables':
        #parameters[0] = observables(parameters, line)
        parameters[0].append(observables(line.words))
    elif section.name == 'hod/ingredients':
        parameters[0].append(ingredients(line.words))
    # all other cosmo,hod sections
    else:
        if line.words[0] == 'name':
            if 'mor' in section:
                parameters[0].append(getattr(relations, line.words[1]))
            elif 'scatter' in section:
                parameters[0].append(getattr(scatter, line.words[1]))
            parameters[1].append(0)
            parameters[2].append(-np.inf)
            parameters[3].append(np.inf)
        else:
            parameters, priors, starting = \
                append_hod_parameters(parameters, priors, starting, line)
    return parameters, priors, starting


def append_hod_parameters(parameters, priors, starting, line):
    """
    To deal with parameters with priors (including fixed values)

    `line` should be a `configline` object

    need to consider `join` instances here.
    """
    words = line.words
    if words[0] not in ('name', 'function'):
        assert prior_is_valid(line)
    priors.append(words[1])
    parameters[0].append(float(words[2]))
    # keep adapting the code above to here
    if priors[-1] == 'fixed':
        parameters[1].append(-1)
    #if priors[-1] in ('lognormal', 'normal', 'uniform'):
    else:
        parameters[1].append(float(words[3]))
        if len(words) > 5:
            parameters[2].append(float(words[4]))
            parameters[3].append(float(words[5]))
        # starting only applies to free parameters
        starting = append_starting(starting, parameters, line)
    if len(parameters[2]) < len(parameters[1]):
        parameters[2].append(-np.inf)
        parameters[3].append(np.inf)
    return parameters, priors, starting


def append_starting(starting, parameters, line):
    words = line.words
    prior = words[1]
    if prior == 'fixed':
        return starting
    if prior == 'uniform':
        if len(words) == 5:
            starting.append(float(words[4]))
        else:
            starting.append(np.random.uniform(
                parameters[0][-1], parameters[1][-1], 1)[0])
    elif prior in ('lognormal', 'normal'):
        if len(words) in (5,7):
            starting.append(words[-1])
        elif prior == 'normal':
            starting.append(np.random.normal(
                parameters[0][-1], parameters[1][-1], 1)[0])
        else:
            starting.append(10**np.random.normal(
                np.log10(parameters[0][-1]),
                np.log10(parameters[1][-1]), 1)[0])
    return starting


def ingredients(words):
    assert words[1] in ('True', 'False'), \
        'Value {1} for parameter {0} in hod/ingredients not valid.' \
        ' Must be True or False'.format(*(words))
    return (words[1] == 'True')


def observables(words):
    binning = np.array(words[1].split(','), dtype=float)
    means = np.array(words[2].split(','), dtype=float)
    return [binning[:-1], binning[1:], means]


def prior_is_valid(line):
    assert line.words[1] in valid_priors, \
        'prior {1} not valid; must be one of {0}'.format(
            valid_priors, line.words[1])
    return True


def starting_value(line):
    """
    `line` must be a `configline` object
    """
    assert prior_is_valid(line)
    words = line.words
    prior = words[1]
    v1 = float(words[2])
    v2 = float(words[3])
    if prior == 'fixed':
        return v1
    if prior == 'uniform':
        if len(words) == 5:
            return float(words[4])
        return np.random.uniform(v1, v2, 1)[0]
    if prior in ('normal', 'lognormal'):
        if len(words) in (5, 7):
            return float(words[-1])
        if prior == 'normal':
            return np.random.normal(v1, v2, 1)[0]
        return 10**np.random.normal(np.log10(v1), np.log10(v2), 1)[0]

