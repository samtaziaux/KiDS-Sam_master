from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import stats

from .core import *
from ...hod import relations, scatter
from ...halomodel import concentration
from ...halomodel.observables import Observable
from ...sampling.priors import (
    draw, fixed_priors, free_priors, nargs as prior_nargs, valid_priors)


def append_setup(parameters, nparams, setup):
    for i in setup:
        parameters[0].append(i)
    nparams.append(len(setup[0]))
    return parameters, nparams


def flatten_parameters(parameters, names, repeat):
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
    nparams = [len(p) for p in parameters[0]]
    return flat, flatnames, repeat, nparams


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


def hod_parameters(names, parameters, priors, starting, line):
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
        #starting.append(None)
        return names, parameters, priors, starting
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
    elif priors[-1] == 'repeat':
        parameters[0].append(-1)
    elif priors[-1] in fixed_priors:
        parameters[0].append(float(words[2]))
    elif prior_nargs[priors[-1]] > 0:
        parameters[0].append(float(words[2]))
    else:
        parameters[0].append(-1)

    if priors[-1] == 'repeat' or priors[-1] in fixed_priors \
            or priors[-1] in ('exp', 'jeffrey'):
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
    #return [binning, means]
    return Observable(words[0], binning, means)


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


class HODParams(object):
    """Class to manage list of HOD parameters

    Attributes
    ----------
    names : list of str
        list of parameter names
    values : list
        list of parameters or set of parameters

    Methods
    -------
    read_section
        Given a section name, return the data

    """

    def __init__(self, names, values):
        self.names = names
        self.values = values

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    def read_section(self, name):
        return self.values[self.section_index(name)]

    def section_index(self, name):
        return self.names.index(name)
