"""Mass-observable relations

"""

from __future__ import absolute_import, division, print_function

from numpy import log10


#don't forget to update this!
__all__ = ('powerlaw', 'double_powerlaw', 'double_powerlaw_scaled')


def powerlaw(M, logM0, a, b, obs_is_log=False):
    x = a + b*(log10(M)-logM0)
    if obs_is_log:
        return x
    return 10**x


def double_powerlaw(M, logM0, logM1, a, b, obs_is_log=False):
    m = M / 10**logM1
    x = logM0 + a*m - (a-b)*log10(1+m)
    if obs_is_log:
        return x
    return 10**x


def double_powerlaw_scaled(M, logM0, logM1, a, b, A, obs_is_log=False):
    """Double power-law with additional scale, e.g. for satellites"""
    return A * double_powerlaw(M, logM0, logM1, a, b, obs_is_log=obs_is_log)

