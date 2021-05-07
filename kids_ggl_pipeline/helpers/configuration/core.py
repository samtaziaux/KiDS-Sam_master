from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from itertools import count, permutations
import numpy as np
from six import string_types
import sys

from . import confighod, configsampler, configsetup, configcovar
from ...halomodel import nfw, nfw_stack, halo, cmbhalo, covariance
from ...halomodel.observables import ModelObservables, Observable
from ...halomodel.selection import Selection
# debugging
from .._debugging import import_icecream
ic = import_icecream()

# local, if present
try:
    import models
except ImportError:
    pass


class ConfigLine(str):

    def __init__(self, line):
        self.line = line
        self._remove_comments()
        self._join_label = None
        self._section = None
        self._words = None
        super(ConfigLine, self).__init__()
        return

    ### attributes ###

    @property
    def join_label(self):
        if self._join_label is None:
            if 'join' in self.words[-1]:
                self._join_label = self.words[-1].split(':')[1]
            else:
                self._join_label = None
        return self._join_label

    @property
    def section(self):
        if self._section is None and self.is_section():
            self._section = self.line[1:self.line.index(']')]
        return self._section

    @property
    def words(self):
        if self._words is None:
            self._words = self.line.split()
            if self.join_label is not None:
                self._words = self._words[:-1]
        return self._words

    ### hidden methods ###

    def _remove_comments(self):
        if '#' in self.line:
            self.line = self.line[:self.line.index('#')]

    ### methods ###

    def is_comment(self):
        if self.is_empty():
            return False
        return self.line.lstrip()[0] == '#'

    def is_empty(self):
        return len(self.line.lstrip()) == 0

    def is_section(self):
        return self.line[0] == '['

    def split_words(self):
        _line = '{0}'.format(self.line)
        if isinstance(_line, string_types):
            _line = _line.split()
        return _line


class ConfigSection(str):
    """Object that handles a section in the configuration file"""

    def __init__(self, name=''):
        self.name = name
        self._parent = None
        super(ConfigSection, self).__init__()
        return

    @property
    def parent(self):
        if self._parent is None:
            self._parent = self.name.split('/')[0]
        return self._parent

    def append_entry_output(self, line, output):
        """
        `line` should be a `ConfigLine` object
        """
        output.append(line.words[0])
        return output

    def append_parameters(
            self, names, parameters, priors, repeat, section_names,
            these_names, these_params, these_priors):
        """Append parameters to the section once we're done reading it

        Repeat parameters are processed here
        """
        # cosmological parameters must be reordered
        if self.name == 'cosmo':
            cosmo = CosmoSection(name='cosmo')
            # remember that these_params also contains values relating
            # to the priors and starting values
            cosmo.set_values(
                **{key: val for key, val in zip(these_names, these_params[0])})
            # need to sort priors accordingly
            cosmo_priors = ['fixed'] * len(cosmo.names)
            # the 1+len(cosmo.values) inclu
            cosmo_params = [cosmo.values] \
                + [len(cosmo.values)*[i] for i in (None,-np.inf,np.inf)]
            # also need to set prior parameters and starting values
            for j, name in enumerate(cosmo.names):
                if name in these_names:
                    i = these_names.index(name)
                    cosmo_priors[j] = these_priors[i]
                    for k in range(3):
                        cosmo_params[1+k][j] = these_params[1+k][i]
                    #these_priors[i] = cosmo_priors[j]
                    #for k in range(3):
                        #these_params
            these_names = cosmo.names
            these_params = cosmo_params
            these_priors = cosmo_priors
            ic()
            ic(these_params)
        names.append(these_names)
        if self.name is None:
            return names, parameters, priors
        # not sure I need this - will check later
        if len(parameters) == 0:
            parameters = [[] for i in these_params]
        if self.name not in ('ingredients', 'observables'):
            #if self.name == 'cosmo':
                #parameters
            for i, p in enumerate(these_params):
                parameters[i].append(p)
            for i, n, pr in zip(count(), these_names, these_priors):
                # then the parameter is repeated from a previous section
                if '.' in n:
                    n = n.split('.')
                    paramsection = '/'.join(n[:-1])
                    # the 3 here is to ignore observables, selection and
                    # ingredients but I need to find a better way to do it
                    j = section_names.index(paramsection) - 3
                    assert n[-1] in names[j], \
                        'parameter {0} not found in section {1}'.format(
                            n[-1], paramsection)
                    jj = names[j].index(n[-1])
                    repeat.append([j,jj])
                    priors.append('repeat')
                    for ip in range(len(parameters)):
                        parameters[ip][-1][i] = parameters[ip][j][jj]
                else:
                    priors.append(pr)
                    repeat.append(-1)
        return names, parameters, priors

    def is_parent(self):
        return self.name == self.parent


class ConfigFile(object):

    def __init__(self, filename):
        """Initialize configuration file"""
        self.filename = filename
        self._data = None
        self._valid_modules = None

    @property
    def data(self):
        if self._data is None:
            with open(self.filename) as file:
                data = file.readlines()
            _data = []
            for line in data:
                line = ConfigLine(line)
                if line.is_empty() or line.is_comment():
                    continue
                _data.append(line.line)
            self._data = _data
        return self._data

    @property
    def valid_modules(self):
        if self._valid_modules is None:
            _modules = {
                'nfw': nfw, 'nfw_stack': nfw_stack, 'halo': halo,
                'covariance': covariance, 'cmbhalo': cmbhalo}
            try:
                _modules['models'] = models
            except NameError:
                pass
            self._valid_modules = _modules
        return self._valid_modules

    def initialize_names(self):
        return []

    def initialize_parameters(self):
        # the four elements correspond to [mean, std, lower, upper]
        return [[] for i in range(4)]

    def initialize_priors(self):
        return []

    def read(self):
        """Read the configuration file"""
        section = ConfigSection()
        model = None
        preamble = None
        cov = None
        section_names = []
        observables = []
        names = []
        parameters = []
        priors = []
        repeat = []
        join = []
        # starting values for free parameters
        starting = []
        ingredients = {}
        covar = {}
        setup = {}
        # dictionaries don't preserve order
        output = []
        sampling = {}
        path = ''
        for line in self.data:
            line = ConfigLine(line)
            if line.words[0] == 'model':
                model = self.read_function(line.words[1])
                continue
            if line.words[0] == 'preamble':
                preamble = self.read_function(line.words[1])
            if line.words[0] == 'cov':
                cov = self.read_function(line.words[1])
                continue
            # we reach this once we're done with the previous section
            if line.is_section():
                if section.name == 'cosmo' or section.name[:3] == 'hod':
                    names, parameters, priors = section.append_parameters(
                        names, parameters, priors, repeat, section_names,
                        these_names, these_params, these_priors)
                # stored all parameters, now we initialize the new section
                section = ConfigSection(line.section)
                section_names.append(section.name)
                these_names = self.initialize_names()
                these_params = self.initialize_parameters()
                these_priors = self.initialize_priors()
                continue
            if section == 'observables':
                observables.append(Observable(*line.words))
            elif section == 'selection':
                selection = Selection(*line.words)
            elif section == 'ingredients':
                ingredients = confighod.ingredients(
                    ingredients, line.words)
            elif section.parent in ('cosmo', 'hod'):
                new = confighod.hod_entries(
                    line, section, these_names, these_params, these_priors,
                    starting, join)
                these_names, these_params, these_priors, starting, join = new
            elif section == 'covariance':
                covar = configcovar.append_entry(line, covar)
            elif section == 'output':
                output = section.append_entry_output(line, output)
            elif section == 'setup':
                setup = configsetup.append_entry(line, setup)
            elif section == 'sampler':
                sampling = configsampler.sampling_dict(line, sampling)
        join = confighod.format_join(join)
        parameters, names, repeat, nparams = \
            confighod.flatten_parameters(parameters, names, repeat, join)
        sampling = configsampler.add_defaults(sampling)
        # add defaults and check types
        ingredients = confighod.add_default_ingredients(ingredients)
        observables = ModelObservables(observables)
        setup = configsetup.check_setup(setup)
        # for now - necessary for k-space filtering functionality
        setup['exclude'] = sampling['exclude']
        #
        covar = configcovar.check_covar(covar)
        hod_param_names = ['observables', 'selection', 'ingredients',
                           'parameters', 'setup', 'covariance']
        hod_params = [
            hod_param_names,
            [observables, selection, ingredients, parameters, setup, covar]]
        hm_params = [model, cov, preamble, hod_params, np.array(names),
                     np.array(priors), np.array(nparams), np.array(repeat),
                     join, np.array(starting), output]
        return hm_params, sampling

    def read_function(self, path):
        # maybe this should be in `ConfigLine` instead
        module, func = path.split('.')
        assert module in self.valid_modules, \
            'Implemented modules are {0}'.format(
                np.sort(list(self.valid_modules.keys())))
        return getattr(self.valid_modules[module], func)


class CosmoSection:

    """Cosmology section object

    Used to store cosmological parameters so that all parameters are
    optional in the configuration file and their order does not matter

    Can only handle flat cosmologies for now
    """
    # TO DO: add options to specify A_s, Oc0, and Ode0 (some
    # commented code already exists)

    def __init__(self, name='cosmo', **kwargs):
        self.name = name
        # this order is fixed
        self._names = \
            ['Om0', 'Ob0', 'h', 'sigma8', 'n_s', 'm_nu', 'Neff',
             'w0', 'wa', 'Tcmb0', 'z', 'n(z)', 'z_mlf', 'z_s']
        # Planck 18 by default
        # TT,TE,EE+lowE+lensing+BAO from Table 2
        self._default = \
            [0.311, 0.049, 0.677, 0.810, 0.967, 0.06, 3.046, -1, 0, 2.725,
             None, None, None, None]
        # this is only a debugging test, will never happen to the user
        if len(self._names) != len(self._default):
            raise ValueError('inconsistent definitions of _names and _default!')
        self._values = None

    @property
    def names(self):
        return self._names

    @property
    def parameters(self):
        return {key: val for key, val in zip(self.names, self.values)}

    @property
    def values(self):
        if self._values is None:
            return self._default
        return self._values

    def set_values(self, **kwargs):
        # cannot specify all of them!
        #if 'A_s' in kwargs and 'sigma8' in kwargs:
            #raise ValueError('Cannot provide both A_s and sigma8')
        # we don't need so much flexibility for now
        #if ('Om0' in kwargs) + ('Ob0' in kwargs) + ('Oc0' in kwargs) \
                #not in (0,2):
            #raise ValueError(
                #'Must provide exactly two of {Om0,Ob0,Oc0} or none at all')
        #if self.flat:
            #if ('Ode0' in kwargs \
                    #and ('Om0' in kwargs \
                        #or ('Ob0' in kwargs and 'Oc0' in kwargs))):
                #msg = 'cannot provide both Ode0 and the total matter density' \
                      #' (either Om0 or both Ob0 and Oc0) for a flat cosmology'
                #raise ValueError(msg)
        # initialize empty
        _values = [None] * len(self.names)
        # assign provided values
        for key, val in kwargs.items():
            if key not in self.names:
                msg = f'parameter {key} not known. Available parameters are' \
                      f' {self.names}'
                raise KeyError(msg)
            _values[self.index(key)] = val
        # add defaults where no value has been specified
        for i, (name, val, default) \
                in enumerate(zip(self.names, _values, self.values)):
            if val is None:
                _values[i] = default
        # note that these things don't matter here because we would
        # run into the same issue at every step in the chain
        # sigma8 will have been filled with the default number above
        # but we don't want that if A_s was provided!
        #if 'A_s' in kwargs:
             #values[self.index('sigma8')] = None
        # make sure the matter density parameters make sense
        #if 'Om0' is None:
            #_values[self.index('Om0')] \
                #= _values[self.index('Ob0')] + _values[self.index('Oc0')]
        #if 'Oc0' is None:
            #_values[self.index('Oc0')] \
                #= _values[self.index('Om0')] - _values[self.index('Ob0')]
        #if 'Ob0' is None:
            #_values[self.index('Ob0')] \
                #= _values[self.index('Om0')] - _values[self.index('Oc0')]

        # finally, assign!
        self._values = _values

    def index(self, param_name):
        return self.names.index(param_name)
