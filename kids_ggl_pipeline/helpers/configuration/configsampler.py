from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os

from .core import *


def sampling_dict(line, sampling):
    words = line.words
    if words[0] in ('data', 'covariance'):
        path_label = 'path_{0}'.format(words[0])
        if path_label not in sampling:
            sampling[path_label] = ''
        sampling[words[0]] = [
            os.path.join(sampling[path_label], words[1]),
            np.array(words[2].split(','), dtype=int)]
    else:
        for dtype in (int, float, str):
            try:
                sampling[words[0]] = dtype(words[1])
                break
            except ValueError:
                pass
    return sampling


def add_defaults(sampling):
    if 'exclude_bins' not in sampling:
        sampling['exclude_bins'] = None
    if 'precision' not in sampling:
        sampling['precision'] = sampling['k']
    return sampling

