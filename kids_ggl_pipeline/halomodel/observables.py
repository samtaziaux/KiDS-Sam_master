from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import six


class Observable(object):
    """Observable object

    """

    def __init__(self, name, obstype, binlows, binhighs, notes='', sampling_size=100):
        """Construct an observable object from the information in the
        configuration file

        """
        self.name = name
        self.obstype = obstype
        self._binlows = binlows
        self._binhighs = binhighs
        assert self.binlows.size == self.binhighs.size, \
            'Must provide the same number of lower and upper observable bin' \
            ' boundaries'
        self._nbins = None
        assert notes in ('', 'log'), \
            'the fourth column in the observable definition is optional' \
            ' but may only be given the value "log" if present. Found' \
            ' "{0}" instead.'.format(notes)
        self.notes = notes
        self.is_log = (self.notes == 'log')
        self.sampling_size = sampling_size
        self._sampling = self.sample()

    @property
    def _valid_obstypes(self):
        return ('gg', 'gm', 'mlf', 'mm')

    @property
    def binlows(self):
        if isinstance(self._binlows, six.string_types):
            self._binlows = np.array(self._binlows.split(','), dtype=float)
        return self._binlows

    @property
    def binhighs(self):
        if isinstance(self._binhighs, six.string_types):
            self._binhighs = np.array(self._binhighs.split(','), dtype=float)
        return self._binhighs

    """
    @property
    def binning(self):
        if isinstance(self._binning, six.string_types):
            self._binning = np.array(self._binning.split(','), dtype=float)
        return self._binning

    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, means):
        if isinstance(means, six.string_types):
            means = means.split(',')
        self._means = np.array(means, dtype=float)
    """

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def obstype(self):
        return self._obstype

    @obstype.setter
    def obstype(self, obstype):
        assert obstype in self._valid_obstypes, \
            f'type "{obstype}" for observable {self.name} not valid.' \
            ' Remember that starting in v2.0.0 you must specify the' \
            ' type of observable you are modelling. Valid observables' \
            f'are {self._valid_obstypes}.'
        self._obstype = obstype

    @property
    def nbins(self):
        if self._nbins is None:
            self._nbins = self.binlows.size
        return self._nbins

    @property
    def sampling_size(self):
        return self._sampling_size

    @sampling_size.setter
    def sampling_size(self, size):
        assert size > 0
        self._sampling_size = size

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sample):
        self._sampling = sample

    ## methods ##

    def sample(self, size=None):
        """Linearly sample each observable bin

        Parameters
        ----------
        size : int, optional
            size of the sample. If not specified, will use
            ``self.sampling_size``

        Returns
        -------
        samples : array of floats, shape ``(nbins,size)``
        """
        if size is None:
            size = self.sampling_size
        return np.array(
            [np.linspace(lo, hi, size)
             for lo, hi in zip(self.binlows, self.binhighs)])





