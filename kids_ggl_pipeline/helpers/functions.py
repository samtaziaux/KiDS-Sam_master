"""Set of generic functions

This module contains generic functions to be used in different
components of the halo model. By "function" we mean here any
operation that returns the observable given the mass, or the
other way around. 

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import log10

from .decorators import logfunc


@logfunc
def double_powerlaw(M, logM0, logM1, a, b, return_log=True):
    m = M / 10**logM1
    return logM0 + a*log10(m) - (a-b)*log10(1+m)


@logfunc
def powerlaw(M, logM0, a, b, return_log=True):
    return a + b*(log10(M)-logM0)


@logfunc
def powerlaw_mz(M, z, logM0, z0, a, b, c, return_log=True):
    return a + b*(log10(M)-logM0) + c*log10((1+z)/(1+z0))


