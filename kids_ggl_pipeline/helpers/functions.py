"""Set of generic functions

This module contains generic functions to be used in different
components of the halo model

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import log10

from .decorators import logify


@logify
def double_powerlaw(M, logM0, logM1, a, b, return_log=True):
    m = M / 10**logM1
    return logM0 + a*m - (a-b)*log10(1+m)


@logify
def powerlaw(M, logM0, a, b, return_log=True):
    return a + b*(log10(M)-logM0)


@logify
def powerlaw_mz(M, z, logM0, z0, a, b, c, return_log=True):
    return a + b*(log10(M)-logM0) + c*log10((1+z)/(1+z0))


