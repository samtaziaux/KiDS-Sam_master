from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glob import glob
import numpy as np
import os

from .core import *


def sampling_dict(line, sampling):
    words = line.words
    assert len(words) >= 2, \
        'Value for quantity {0} not provided'.format(words[0])
    if words[0] in ('data', 'covariance'):
        assert len(words) >= 3, \
            'line {0} must have at least three entries ({1} instead)'.format(
                line.line, len(line.line))
        path_label = 'path_{0}'.format(words[0])
        if path_label not in sampling:
            sampling[path_label] = ''
        path = os.path.join(sampling[path_label], words[1])
        if words[0] == 'data':
            print(os.path.join(sampling[path_label], words[1]))
            sampling[words[0]] = \
                [sorted(glob(os.path.join(sampling[path_label], words[1])))]
        else:
            sampling[words[0]] = [path]
        sampling[words[0]].append(np.array(words[2].split(','), dtype=int))
    elif ',' in words[1]:
        for dtype in (int, float, str):
            try:
                sampling[words[0]] = np.array(
                    words[1].split(','), dtype=dtype)
                break
            except ValueError:
                pass
    else:
        assert len(words) >= 2, \
            'line {0} must have at least two entries ({1} instead)'.format(
                line.line, len(line.line))
        for dtype in (int, float, str):
            try:
                sampling[words[0]] = dtype(words[1])
                break
            except ValueError:
                pass
    return sampling


def add_defaults(sampling):
    # replace old names for backward compatibility
    if 'exclude_bins' in sampling:
        sampling['exclude'] = sampling.pop('exclude_bins')
    if 'sampler_output' in sampling:
        sampling['output'] = sampling.pop('sampling_output')
    if 'update_freq' in sampling:
        sampling['update'] = sampling.pop('update_freq')
    if 'exclude' not in sampling:
        sampling['exclude'] = None
    # in case a single bin is excluded
    if sampling['exclude'] is not None:
        if not hasattr(sampling['exclude'], '__iter__'):
            sampling['exclude'] = np.array([sampling['exclude']])
    return sampling

