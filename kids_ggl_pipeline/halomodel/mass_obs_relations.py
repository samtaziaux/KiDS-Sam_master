from __future__ import absolute_import, division, print_function

from numpy import log10


def powerlaw(M, logM0, a, b):
    return 10**(a + b*log10(M/10**logM0))


def double_powerlaw(M, logM0, logM1, a, b):
    m = M / 10**logM1
    return 10**logM0 * m**a / (1+m)**(a-b)


def double_powerlaw_scaled(M, logM0, logM1, a, b, A):
    """Double power-law with additional scale, e.g. for satellites"""
    return A * double_powerlaw(M, logM0, logM1, a, b)
