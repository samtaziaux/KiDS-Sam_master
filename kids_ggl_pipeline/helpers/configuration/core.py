from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from six import string_types

from . import confighod, configsampler


class configline(str):

    def __init__(self, line):
        self.line = line
        self._join_label = None
        self._section = None
        self._words = None
        super().__init__()
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


class configsection(str):

    def __init__(self, name=''):
        self.name = name
        self._parent = None
        super().__init__()
        return

    @property
    def parent(self):
        if self._parent is None:
            self._parent = self.name.split('/')[0]
        return self._parent

    def append_entry_output(self, line, output):
        """
        `line` should be a `configline` object
        """
        output[0].append(line.words[0])
        output[1].append(line.words[1])
        return output

    def append_entry_setup(self, line, setup):
        """
        `line` should be a `configline` object
        """
        setup[0].append(line.words[0])
        for dtype in (int, float, str):
            try:
                setup[1].append(dtype(line.words[1]))
                break
            except ValueError:
                pass
        return setup

    def append_subsection_priors(self, priors, these_priors):
        if self.name is None \
                or self.name in ('hod/ingredients', 'hod/observables'):
            return priors
        if self.name.count('/') == 0:
            priors.append(these_priors)
        elif self.name.count('/') == 1:
            priors[-1].append(these_priors)
        elif self.name.count('/') == 2:
            priors[-1][-1].append(these_priors)
        return priors

    def append_subsection_parameters(self, parameters, these_params):
        if self.name is None:
            return parameters
        if len(parameters) == 0:
            for i, p in enumerate(these_params):
                parameters.append([p])
        elif self.name.count('/') == 0:
            for i, p in enumerate(these_params):
                parameters[i].append(p)
        elif self.name.count('/') == 1:
            for i, p in enumerate(these_params):
                parameters[i][-1].append(p)
        elif self.name.count('/') == 2:
            for i, p in enumerate(these_params):
                parameters[i][-1][-1].append(p)
        return parameters

    def is_parent(self):
        return self.name == self.parent


class ConfigFile:

    def __init__(self, filename):
        """Initialize configuration file"""
        self.filename = filename
        self._data = None

    @property
    def data(self):
        if self._data is None:
            with open(self.filename) as file:
                data = file.readlines()
            _data = []
            for line in data:
                line = configline(line)
                if line.is_empty() or line.is_comment():
                    continue
                if '#' in line:
                    line = line.remove_comments()
                _data.append(line)
            self._data = _data
        return self._data

    @property
    def valid_modules(self):
        return {'satellites': satellites, 'nfw': nfw, 'nfw_stack': nfw_stack,
                'halo': halo, 'halo_2': halo_2, 'halo_2_mc': halo_2_mc,
                'halo_sz': halo_sz, 'halo_sz_modular': halo_sz_modular}

    def initialize_parameters(self):
        return [[] for i in range(4)]

    def initialize_priors(self):
        return []

    def read(self):
        """
        Need to reshape parameters to return val1, val2, val3, val4
        instead of cosmo, hod separately.
        """
        section = configsection()
        parameters = []
        priors = []
        # starting values for free parameters
        starting = []
        # I wanted to use dictionaries but
        # the problem is they don't preserve order!
        setup = [[], []]
        output = [[], []]
        sampling = {}
        path = ''
        for line in self.data:
            line = configline(line)
            if line[0] == 'model':
                model = self.read_function(line.words[1])
                continue
            if line.is_section():
                if section.name == 'cosmo' or section.name[:3] == 'hod':
                    parameters = section.append_subsection_parameters(
                        parameters, these_params)
                    priors = section.append_subsection_priors(
                        priors, these_priors)
                # initialize new section
                section = configsection(line.section)
                these_params = self.initialize_parameters()
                these_priors = self.initialize_priors()
                continue
            if section.parent in ('cosmo', 'hod'):
                new = confighod.hod_entries(
                    line, section, these_params, these_priors, starting)
                these_params, these_priors, starting = new
            elif section == 'setup':
                setup = section.append_entry_setup(line, setup)
            elif section == 'output':
                output = section.append_entry_output(line, output)
            elif section == 'sampler':
                sampling = configsampler.sampling_dict(line, sampling)
        sampling = configsampler.add_defaults(sampling)
        return [parameters, priors, starting, setup, output], sampling

    def read_function(self, path):
        # maybe this should be in `configline` instead
        module, func = path.split('.')
        assert module in self.valid_modules, \
            'Implemented modules are {0}'.format(
                np.sort(list(self.valid_modules.keys())))
        return getattr(self.valid_modules[module], func)

