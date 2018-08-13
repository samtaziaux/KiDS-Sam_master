from __future__ import absolute_import, division, print_function

from numpy import log10


#don't forget to update this!
__all__ = ('powerlaw', 'double_powerlaw', 'double_powerlaw_scaled')


def powerlaw(M, M0, a, b, return_log=False):
    x = a + b*log10(M/M0)
    if return_log:
        return x
    return 10**x


def double_powerlaw(M, logM0, logM1, a, b, return_log=False):
    m = M / 10**logM1
    return 10**logM0 * m**a / (1+m)**(a-b)


def double_powerlaw_scaled(M, logM0, logM1, a, b, A, return_log=False):
    """Double power-law with additional scale, e.g. for satellites"""
    return A * double_powerlaw(M, logM0, logM1, a, b, return_log=return_log)

