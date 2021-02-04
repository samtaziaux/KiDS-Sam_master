from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from itertools import count
import numpy as np
from six import string_types
import sys

from . import confighod, configsampler, configsetup, configcovar
from ...halomodel import nfw, nfw_stack, halo, cmbhalo, covariance
from ...halomodel.observables import ModelObservables, Observable
from ...halomodel.selection import Selection

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
        names.append(these_names)
        if self.name is None:
            return names, parameters, priors
        # not sure I need this - will check later
        if len(parameters) == 0:
            parameters = [[] for i in these_params]
        if self.name not in ('ingredients', 'observables'):
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
<<<<<<< HEAD
                'halo_2': halo_2, 'covariance': covariance,
                'cmbhalo': cmbhalo}
=======
                'covariance': covariance}
>>>>>>> master
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
            if line.words[0] == 'cov':
                cov = self.read_function(line.words[1])
                continue
            if line.is_section():
                if section.name == 'cosmo' or section.name[:3] == 'hod':
                    names, parameters, priors = section.append_parameters(
                        names, parameters, priors, repeat, section_names,
                        these_names, these_params, these_priors)
                # initialize new section
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
        covar = configcovar.check_covar(covar)
        hod_param_names = ['observables', 'selection', 'ingredients',
                           'parameters', 'setup', 'covariance']
        hod_params = [
            hod_param_names,
            [observables, selection, ingredients, parameters, setup, covar]]
        hm_params = [model, cov, hod_params, np.array(names), np.array(priors),
                     np.array(nparams), np.array(repeat), join,
                     np.array(starting), output]
        return hm_params, sampling

    def read_function(self, path):
        # maybe this should be in `ConfigLine` instead
        module, func = path.split('.')
        assert module in self.valid_modules, \
            'Implemented modules are {0}'.format(
                np.sort(list(self.valid_modules.keys())))
        return getattr(self.valid_modules[module], func)

