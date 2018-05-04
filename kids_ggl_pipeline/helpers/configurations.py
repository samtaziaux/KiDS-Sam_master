from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import yaml

# local
from ..halomodel.hod import relations, scatter


class Config(object):

    def __init__(self, filename, format='yaml'):
        """Configuration file object"""
        self.filename = filename
        self.format = format
        self.data = self.read()
        self._valid_types = None
        self._components = None
        self.

    @property
    def components(self):
        if self._components is None:
            return list(self.data.keys())

    @property
    def valid_types(self):
        if self._valid_types is None:
            return ('esd', 'halomodel', 'sampler')


    def parse_params(self, component):
        for key, value in self.data[component].items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.data[key][k] = self.parse_value(k, v, parent=key)
            else:
                self.data[key] = self.parse_value(key, value)


    def parse_value(self, name, value, parent=None):
        # where does read_function come from?
        if name == 'model':
            return read_function(*value.split('.'))
        if name == 'path':
            return value
        """
        if name in ('mor', 'scatter'):
            value = value.split('.')
            # then it's definitely a custom function
            if len(value) == 2:
                # need to specify the place where the user can create
                # these functions
                return read_function(*value)
            if name == 'mor':
                # will raise error if non existent (maybe should
                # modify error message though)
                return getattr(relations, value)
            if name == 'scatter':
                return getattr(scatter, value)
        """

        v = value.split()
        if v[0] == 'fixed

    def read(self):
        # new yaml format
        if self.format == 'yaml':
            with open(self.filename) as cfg:
                contents = cfg.read()
            self.data = yaml.safe_load(contents)
            for key in self.data.keys():
                self.parse_params(key)
                setattr(self, key, self.data[key])
        # old custom format
        else:
            with open(self.filename) as cfg:
                data = [line.split() for line in cfg.readlines()
                        if not (line[0] == '#' or line == '\n')]
            # finish up

    def set_params(self, component):
        assert component in self.valid_types, \
            'please specify a valid configuration type: {0}'.format(
                self.valid_types)
        assert component in self.components, \
            'pipeline component {0} not in configuration file'.format(
                component)
        params = []

    #def setup_model(self):
        
