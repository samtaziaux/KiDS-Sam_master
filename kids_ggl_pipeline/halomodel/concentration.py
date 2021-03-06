from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import log10

from ..helpers.functions import powerlaw, powerlaw_mz


"""Default relations for the mass-concentration relation"""

# Note that relations that do not include redshift must add a leading
# empty dimension (see, e.g., dutton14)


def duffy08_crit(M, z, f, h=1):
    return f * powerlaw_mz(
        M, z, 12.301, 0, 0.8267, -0.091, -0.44,
        return_log=False) # h-factor already accounted for in the M, cancels out! (12.301-log10(h))


def duffy08_mean(M, z, f, h=1):
    return f * powerlaw_mz(
        M, z, 12.301, 0, 1.006, -0.081, -1.01,
        return_log=False) # h-factor already accounted for in the M, cancels out! (12.301-log10(h))


def dutton14(M, f, h=1):
    """At z=0"""
    # h-factor already accounted for in the M, cancels out!
    return f * powerlaw(M, 12, 0.905, -0.101, return_log=False)[None]
    #return f * powerlaw(M, 12-log10(h), 0.905, -0.101, return_log=False)


def cm_powerlaw(M, logM0, a, b):
    return powerlaw(M, logM0, a, b, return_log=False)[None]


def cmz_powerlaw(M, z, logM0, z0, a, b, c):
    return powerlaw_mz(M, z, logM0, z0, a, b, c, return_log=False)
