from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np


def powerlaw(M, A, B, Mo=1e12):
    return A * (M/Mo)**B


def powerlaw_mz(M, z, A, B, C, Mo=1e12):
    return A * (M/Mo)**B * (1+z)**C

## Default relations

def duffy08_crit(M, z, h=1):
    return powerlaw_mz(M, z, 6.71, -0.091, -0.44, Mo=2e12/h)


def duffy08_mean(M, z, h=1):
    return powerlaw_mz(M, z, 10.14, -0.081, -1.01, Mo=2e12/h)


def dutton14(M, h=1):
    return powerlaw(M, 8.035, -0.101, Mo=1e12/h)


