from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import yaml


class Config(object):

    def __init__(self, filename):
        """Configuration file object"""
        self.filename = filename
        self.data = self.read()
        self._valid_types = None
        self._components = None

    @property
    def components(self):
        if self._components is None:
            return list(self.data.keys())

    @property
    def valid_types(self):
        if self._valid_types is None:
            return ('esd', 'halomodel', 'sampler')

    def read(self):
        with open(self.filename) as cfg:
            contents = cfg.read()
        self.data = yaml.safe_load(contents)
        for key in self.data.keys():
            setattr(self, key, self.data[key])

    def set_params(self, component):
        assert component in self.valid_types, \
            'please specify a valid configuration type: {0}'.format(
                self.valid_types)
        assert component in self.components, \
            'pipeline component {0} not in configuration file'.format(
                component)
        params = []

    #def setup_model(self):
        
