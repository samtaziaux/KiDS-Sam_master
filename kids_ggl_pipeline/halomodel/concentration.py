from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import log10, newaxis

from ..helpers.functions import powerlaw, powerlaw_mz


##
## Default relations
##

"""
def duffy08_crit(M, z, f, h=1):
    return f * powerlaw_mz(M, z, 12.301-log10(h), 0, 6.71, -0.091, -0.44)


def duffy08_mean(M, z, f, h=1):
    return f * powerlaw_mz(M, z, 12.301-log10(h), 0, 10.14, -0.081, -1.01)


def dutton14(M, f, h=1):
    return f * powerlaw(M, 12-log10(h), 8.035, -0.101)
"""

"""
These below serve as a placeholder until we implement the ability to
use the same parameter in multiple places, at which point the
z[:,newaxis] operation will probably happen in halo.model
"""


def duffy08_crit(M, z, f, h=1):
    return f * powerlaw_mz(
        M, z[:,newaxis], 12.301-log10(h), 0, 0.8267, -0.091, -0.44,
        return_log=False)


def duffy08_mean(M, z, f, h=1):
    return f * powerlaw_mz(
        M, z[:,newaxis], 12.301-log10(h), 0, 1.006, -0.081, -1.01,
        return_log=False)


def dutton14(M, f, h=1):
    """At z=0"""
    return f * powerlaw(M, 12-log10(h), 0.905, -0.101, return_log=False)

