from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glob import glob
import numpy as np
import os

from .core import *


def sampling_dict(line, sampling):
    words = line.words
    if words[0] in ('data', 'covariance'):
        path_label = 'path_{0}'.format(words[0])
        if path_label not in sampling:
            sampling[path_label] = ''
        path = os.path.join(sampling[path_label], words[1])
        if words[0] == 'data':
            sampling[words[0]] = \
                [sorted(glob(os.path.join(sampling[path_label], words[1])))]
        else:
            sampling[words[0]] = [path]
        sampling[words[0]].append(np.array(words[2].split(','), dtype=int))
    else:
        for dtype in (int, float, str):
            try:
                sampling[words[0]] = dtype(words[1])
                break
            except ValueError:
                pass
    return sampling


def add_defaults(sampling):
    if 'exclude' not in sampling:
        sampling['exclude'] = None
    return sampling

