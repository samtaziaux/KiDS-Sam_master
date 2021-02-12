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
    def nbins(self):
        if self._nbins is None:
            self._nbins = self.binlows.size
        return self._nbins

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


class ModelObservables:

    def __init__(self, observables):
        self.observables = observables
        self.obstypes = [obs.obstype for obs in self.observables]
        # note that this is a list with number of bins per observable
        self._nbins = [obs.nbins for obs in self.observables]
        # and this the total number of bins
        self.nbins = sum(self._nbins)
        self.sampling = np.concatenate(
            [obs.sampling for obs in self.observables if obs.obstype != 'mlf'],
            axis=0)
        self.cmblens = self._init_obs('cmblens')
        self.gg = self._init_obs('gg')
        self.gm = self._init_obs('gm')
        self.mlf = self._init_obs('mlf')
        self.mm = self._init_obs('mm')

    def __getitem__(self, i):
        if isinstance(i, (int,np.integer)):
            return self.observables[i]
        elif isinstance(i, str):
            if i not in self.obstypes:
                err = f'{i} not a valid observable, must be one of {self.obstypes}'
                raise KeyError(err)
            return self.observables[self.obstypes.index(i)]
        else:
            err = 'index must be either int or str, received ' \
                  f'{type(i).__name__} instead'
            raise TypeError(err)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self.observables):
            next = self.observables[self._i]
            self._i += 1
            return next
        else: raise StopIteration

    ### hidden methods ###

    def _init_obs(self, obs):
        if obs in self.obstypes:
            return self._obs_attribute(self.obstypes.index(obs))
        else:
            return False

    def _add_R(self, R):
        for obs in self.observables:
            obs.R = [r[1:].astype('float64') for r in R[obs.idx]]
            obs.size = np.array([len(r) for r in obs.R])

    def _obs_attribute(self, i):
        obs = self.observables[i]
        obs.idx = self._obs_idx(i)
        return obs

    def _obs_idx(self, i):
        prior = sum(self._nbins[:i])
        return np.s_[prior:prior+self._nbins[i]]


