from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import stats

from .core import *
from ...hod import relations, scatter
from ...halomodel import concentration
from ...halomodel.observables import Observable
from ...helpers import io
from ...sampling.priors import (
    define_limits, draw, fixed_priors, free_priors, nargs as prior_nargs,
    valid_priors)


def add_default_ingredients(ingredients):
    options = ('centrals', 'pointmass', 'miscentring', 'satellites',
               'twohalo', 'haloexclusion', 'nzlens')
    default = {key: False for key in options}
    for key in ingredients:
        if key not in options:
            msg = 'ingredient {0} not a valid entry. Valid entries are' \
                  ' {1}'.format(key, options)
            raise ValueError(msg)
    for key, val in default.items():
        if key not in ingredients:
            ingredients[key] = val
    return ingredients


def append_setup(parameters, nparams, setup):
    for i in setup:
        parameters[0].append(i)
    nparams.append(len(setup[0]))
    return parameters, nparams


def flatten_parameters(parameters, names, repeat, join):
    flat = [[] for p in parameters]
    flatnames = []
    # first the four sets for the priors
    for i, sections in enumerate(parameters):
        # this iterates over sections
        for s, sect in enumerate(sections):
            pstart = sum([len(x) for x in sections[:s]])
            for p, param in enumerate(sect):
                flat[i].append(param)
                if i > 0:
                    continue
                flatnames.append(names[s][p])
                rloc = pstart + p
                if repeat[rloc] != -1:
                    r = repeat[rloc]
                    repeat[rloc] = \
                        sum([len(x) for x in sections[:r[0]]]) + r[1]
    flat = [np.array(i) for i in flat]
    # count number of parameters per section
    nparams = [len(p) for p in parameters[0]]
    # remove joined parameters from count
    join = np.array([ji for j in join for ji in j])
    end = 0
    for i, npar in enumerate(nparams):
        # previous values in nparams have changed so we cannot
        # sum over its elements every time
        start = end
        end = start + npar
        joined = np.sum((join >= start) & (join < end))
        if joined > 0:
            nparams[i] -= joined - 1
    return flat, flatnames, repeat, nparams


def format_join(join):
    join = np.array(join)
    rng = np.arange(join.size)
    join_labels = np.unique(join[join != '-1'])
    join_arrays = []
    for label in join_labels:
        j = (join == label)
        join[j] = rng[j][0]
        join_arrays.append(rng[j])
    return join_arrays


def hod_entries(line, section, names, parameters, priors, starting, join):
    """
    `line` should be a `configline` object
    """
    # record 'join' first just in case we want to add this
    # functionality to repeat parameters
    join.append(line.join_label if line.join_label else '-1')
    if line.words[0] == 'name':
        names, parameters, priors = hod_function(
            line, names, parameters, priors, section)
    else:
        names, parameters, priors, starting = \
            hod_parameters(line, names, parameters, priors, starting)
    return names, parameters, priors, starting, join


def hod_function(line, names, parameters, priors, section):
    if 'mor' in section:
        parameters[0].append(getattr(relations, line.words[1]))
    elif 'scatter' in section:
        parameters[0].append(getattr(scatter, line.words[1]))
    elif 'concentration' in section:
        parameters[0].append(getattr(concentration, line.words[1]))
    # felixibility not yet implemented
    elif 'miscentring' in section:
        parameters[0].append(line.words[1])
    parameters[1].append(0)
    parameters[2].append(-np.inf)
    parameters[3].append(np.inf)
    # option to provide a custom name to the function
    # NOTE: not documented!
    if len(line.words) == 3:
        names.append(line.words[2])
    else:
        names.append(line.words[1])
    priors.append('function')
    return names, parameters, priors


def hod_parameters(line, names, parameters, priors, starting):
    """
    To deal with parameters with priors (including fixed values)

    `line` should be a `configline` object

    need to consider `join` instances here.
    """
    words = line.words
    # reading a parameter from a different section. In this case,
    # just assign None placeholders to everything except the name
    if '.' in words[0]:
        names.append(words[0])
        for i in range(len(parameters)):
            parameters[i].append(None)
        priors.append(None)
        return names, parameters, priors, starting
    # check that the prior is valid
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
        p0 = np.array(words[2].split(','), dtype=float)
    elif priors[-1] == 'read':
        cols = np.array(words[3].split(','), dtype=int)
        p0 = io.read_ascii(words[2], columns=cols)
        if len(cols) == 1:
            p0 = p0[0]
    elif priors[-1] == 'repeat':
        p0 = -1
    elif priors[-1] in fixed_priors:
        p0 = float(words[2])
    elif prior_nargs[priors[-1]] > 0:
        p0 = float(words[2])
    else:
        p0 = -99
    parameters[0].append(p0)

    if priors[-1] == 'repeat' or priors[-1] in fixed_priors \
            or priors[-1] in ('exp', 'jeffreys'):
        parameters[1].append(-99)
        starting = starting_values(starting, parameters, line)
    else:
        if prior_nargs[priors[-1]] == 2:
            parameters[1].append(float(words[3]))
            if len(words) > 5:
                parameters[2].append(float(words[4]))
                parameters[3].append(float(words[5]))
        else:
            parameters[1].append(-99)
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
        if priors[-1] in fixed_priors:
            parameters[2].append(-np.inf)
            parameters[3].append(np.inf)
        else:
            lims = define_limits(priors[-1], words[2:])
            parameters[2].append(lims[0])
            parameters[3].append(lims[1])

    return names, parameters, priors, starting


def ingredients(ingr, words):
    assert words[1] in ('True', 'False'), \
        'Value {1} for parameter {0} in hod/ingredients not valid.' \
        ' Must be True or False'.format(*(words))
    ingr[words[0]] = (words[1] == 'True')
    return ingr


def observables(words):
    options = ('gm', 'gg', 'mm', 'mlf')
    default = {key: False for key in options}
    for key in words[1]:
        if key not in options:
            ingr = 'gm'
            binning = np.array(words[1].split(','), dtype=float)
            means = np.array(words[2].split(','), dtype=float)
        else:
            ingr = words[1]
            binning = np.array(words[2].split(','), dtype=float)
            means = np.array(words[3].split(','), dtype=float)
    return Observable(words[0], ingr, binning, means)


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
    if prior == 'repeat':
        starting.append(-999)
        return starting
    # if the starting points are defined in the config file
    if (len(words) - 2 - prior_nargs[prior]) % 2 == 1:
        starting.append(float(words[-1]))
    else:
        args = [p[-1] for p in parameters[:2]]
        bounds = [parameters[2][-1], parameters[3][-1]]
        starting.append(draw(prior, args, bounds, size=None))
    return starting

