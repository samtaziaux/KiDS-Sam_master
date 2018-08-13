from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import six


class Observable():
    """Observable object

    """

    def __init__(self, name, binning, means, sampling_size=100):
        """Construct an observable object from the information in the
        configuration file

        """
        self.name = name
        self.binning = binning
        self.means = means
        self._nbins = None
        self.sampling_size = sampling_size
        self._sampling = self.sample()

    @property
    def binning(self):
        return self._binning

    @binning.setter
    def binning(self, binning):
        if isinstance(binning, six.string_types):
            binning = binning.split(',')
        self._binning = np.array(binning, dtype=float)

    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, means):
        if isinstance(means, six.string_types):
            means = means.split(',')
        self._means = np.array(means, dtype=float)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def nbins(self):
        if self._nbins is None:
            self._nbins = self.binning.size - 1
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

    def is_log(self):
        """Check whether the observable is defined in log space

        Returns
        -------
        """
        return self.name.startswith('log')

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
            [np.linspace(self.binning[i-1], self.binning[i], size)
             for i in range(1, self.binning.size)])





