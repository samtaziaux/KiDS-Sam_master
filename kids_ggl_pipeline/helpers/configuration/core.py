from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from six import string_types

from . import confighod, configsampler
from ...halomodel import nfw, nfw_stack, halo, halo_2, halo_2_mc
from ...halomodel.observables import Observable
from ...halomodel.selection import Selection

# local, if present
try:
    import models
except ImportError:
    pass


class ConfigLine(str):

    def __init__(self, line):
        self.line = line
        self._join_label = None
        self._section = None
        self._words = None
        super(ConfigLine, self).__init__()
        return

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

    def is_comment(self):
        if self.is_empty():
            return False
        return self.line.lstrip()[0] == '#'

    def is_empty(self):
        return len(self.line.lstrip()) == 0

    def is_section(self):
        return self.line[0] == '['

    def remove_comments(self):
        if '#' in self.line:
            return self.line[:self.line.index('#')]
        return self.line

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
        output[0].append(line.words[0])
        output[1].append(line.words[1])
        return output

    def append_entry_setup(self, line, setup):
        """
        `line` should be a `ConfigLine` object
        """
        #setup[0].append(line.words[0])
        for dtype in (int, float, str):
            try:
                #setup[1].append(dtype(line.words[1]))
                setup[line.words[0]] = dtype(line.words[1])
                break
            except ValueError:
                pass
        return setup

    def append_subsection_names(self, names, these_names):
        for n in these_names:
            names.append(n)
        return names

    def append_subsection_priors(self, priors, these_priors):
        if self.name is None \
                or self.name in ('ingredients', 'observables'):
            return priors
        #priors.append(np.array([these_priors]))
        for p in these_priors:
            priors.append(p)
        return priors

    def append_subsection_parameters(self, parameters, these_params):
        if self.name is None:
            return parameters
        # initialize
        if len(parameters) == 0:
            parameters = [[] for i in these_params]
        for i, p in enumerate(these_params):
            parameters[i].append(p)
        return parameters

    def is_parent(self):
        return self.name == self.parent


class ConfigFile:

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
                if '#' in line:
                    line = line.remove_comments()
                _data.append(line)
            self._data = _data
        return self._data

    @property
    def valid_modules(self):
        if self._valid_modules is None:
            _modules = {
                'nfw': nfw, 'nfw_stack': nfw_stack, 'halo': halo,
                'halo_2': halo_2, 'halo_2_mc': halo_2_mc}
            try:
                _modules['models'] = models
            except NameError:
                pass
            self._valid_modules = _modules
        return self._valid_modules

    def initialize_names(self):
        return []

    def initialize_parameters(self):
        return [[] for i in range(4)]

    def initialize_priors(self):
        return []

    def read(self):
        """Read the configuration file"""
        section = ConfigSection()
        observables = []
        names = []
        parameters = []
        priors = []
        # starting values for free parameters
        starting = []
        ingredients = {}
        setup = {}
        # dictionaries don't preserve order
        output = [[], []]
        sampling = {}
        path = ''
        for line in self.data:
            line = ConfigLine(line)
            if line.words[0] == 'model':
                model = self.read_function(line.words[1])
                continue
            if line.is_section():
                if section.name == 'cosmo' or section.name[:3] == 'hod':
                    parameters = section.append_subsection_parameters(
                        parameters, these_params)
                    priors = section.append_subsection_priors(
                        priors, these_priors)
                    names = section.append_subsection_names(
                        names, these_names)
                # initialize new section
                section = ConfigSection(line.section)
                these_names = self.initialize_names()
                these_params = self.initialize_parameters()
                these_priors = self.initialize_priors()
                continue
            if section == 'observables':
                #observables = confighod.observables(line.words)
                observables.append(Observable(*line.words))
            elif section == 'selection':
                selection = Selection(*line.words)
            elif section == 'ingredients':
                ingredients = confighod.ingredients(
                    ingredients, line.words)
            elif section.parent in ('cosmo', 'hod'):
                new = confighod.hod_entries(
                    line, section, these_names, these_params, these_priors,
                    starting)
                these_names, these_params, these_priors, starting = new
            elif section == 'setup':
                setup = section.append_entry_setup(line, setup)
            elif section == 'output':
                output = section.append_entry_output(line, output)
            elif section == 'sampler':
                sampling = configsampler.sampling_dict(line, sampling)
        parameters, nparams = confighod.flatten_parameters(parameters)
        sampling = configsampler.add_defaults(sampling)
        #hod_params = confighod.HODParams(
            #'observables,selection,ingredients,parameters,setup'.split(','),
            #[observables, selection, ingredients, parameters, setup])
        #hod_params = {
            #'observables': observables, 'selection': selection,
            #'ingredients': ingredients, 'parameters': parameters,
            #'setup': setup}
        hod_params = [
            'observables,selection,ingredients,parameters,setup'.split(','),
            [observables, selection, ingredients, parameters, setup]]
        hm_params = [model, hod_params, np.array(names), np.array(priors),
                     np.array(nparams), np.array(starting), np.array(output)]
        return hm_params, sampling

    def read_function(self, path):
        # maybe this should be in `ConfigLine` instead
        module, func = path.split('.')
        assert module in self.valid_modules, \
            'Implemented modules are {0}'.format(
                np.sort(list(self.valid_modules.keys())))
        return getattr(self.valid_modules[module], func)

