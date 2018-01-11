from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import psutil
import os
import numpy as np


def test():     # return the memory usage in MB
	process = psutil.Process(os.getpid())
	memfrac = psutil.virtual_memory().percent
	mem = process.memory_info()[0] / 2**20
	print('Memory:', mem, 'MB, ', memfrac, '%')
	return memfrac
