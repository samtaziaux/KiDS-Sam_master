"""Set of generic functions

This module contains generic functions to be used in different
components of the halo model. By "function" we mean here any
operation that returns the observable given the mass, or the
other way around. 

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import expand_dims, iterable, log10

from .decorators import logfunc


@logfunc
def double_powerlaw(M, logM0, logM1, a, b, norm=1.0, return_log=True):
    m = M / 10**logM1
    norm = log10(norm)
    return norm + logM0 + a*log10(m) - (a-b)*log10(1+m)


@logfunc
def powerlaw(M, logM0, a, b, norm=1.0, return_log=True):
    norm = log10(norm)
    return norm + a + b*(log10(M)-logM0)


@logfunc
def powerlaw_mz(M, z, logM0, z0, a, b, c, norm=1.0, return_log=True):
    """
    M must be a vector of shape (N,)
    z can be a vector of arbitrary shape, and all other
    quantities are assumed scalar
    """
    norm = log10(norm)
    if iterable(z):
        z = expand_dims(z, -1)
    return  norm + (a + b*(log10(M)-logM0)) + c*log10((1+z)/(1+z0))


