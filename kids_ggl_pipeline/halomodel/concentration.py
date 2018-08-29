from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import log10, newaxis

from ..helpers.functions import powerlaw, powerlaw_mz


##
## Default relations
##


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

