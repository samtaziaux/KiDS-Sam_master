from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import log10

from ..helpers.functions import powerlaw, powerlaw_mz


##
## Default relations
##


def duffy08_crit(M, z, f, h=1):
    return f * powerlaw_mz(M, z, 12.301-log10(h), 0, 6.71, -0.091, -0.44)


def duffy08_mean(M, z, f, h=1):
    return f * powerlaw_mz(M, z, 12.301-log10(h), 0, 10.14, -0.081, -1.01)


def dutton14(M, f, h=1):
    return f * powerlaw(M, 12-log10(h), 8.035, -0.101)


