"""Set of generic distributions

This module contains generic distributions to be used in different
components of the halo model. Methods here take *both* observable and
halo mass to return a probability distribution, as opposed to methods
in `functions.py` which return one given the other.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import arange, array, exp, expand_dims, log, log10, pi

from .decorators import logdist


@logdist
def lognormal(obs, Mo, Mh, sigma, obs_is_log=False):
    """Log-normal scatter

    Eq. 15 of van Uitert et al. (2016)

    Parameters
    ----------
    obs : float array, shape ([nbins,]N)
        observable. ``nbins`` is the number of observable bins
    Mo : float array, shape (P,)
        observable mass, from mass-observable relation
    Mh : float array, shape (P,)
        halo mass
    sigma : float or float array with shape (nbins,)
        scatter
    obs_is_log : bool, optional
        whether the observable is in log-space. Note that if this is
        the case then ``hod.probability`` will pass logM here, so
        it is assumed that a log-space observable implies a log-space
        halo mass.

    Returns
    -------
    distribution : float array, shape ([nbins,]N,P)
        lognormal distribution of the observable about halo mass

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    if np.iterable(sigma):
        sigma = expand_dims(expand_dims(sigma, -1), -1)
    obs = expand_dims(obs, -1)
    return exp(-((log10(obs/Mo)**2.0) / (2.0*sigma**2.0))) \
        / ((2.0*pi)**0.5 * sigma * obs * log(10.0))


@logdist
def modschechter(obs, Mo, Mh, logMref, alpha, b, obs_is_log=False):
    """Modified Schechter scatter (eq. 17 in van Uitert et al. 2016)

    This is generally used for satellite galaxies, and the mor
    would be `double_powerlaw_scaled`

    Parameters
    ----------
    obs : float array, shape ([nbins,]N)
        observable. ``nbins`` is the number of observable bins
    Mo : float array, shape (P,)
        observable mass, from mor relation
    Mh : float array, shape (P,)
        halo mass
    logMref : float
        log of the characteristic mass
    alpha : float
        low-mass slope
    b : float array, arbitrary length (1D)
        polynomial terms for phi_s, equation 18 in van Uitert+16
        whether the observable is in log-space. Note that if this is
        the case then ``hod.probability`` will pass logM here, so
        it is assumed that a log-space observable implies a log-space
        halo mass.

    Returns
    -------
    distribution : float array, shape ([nbins,]N,P)
        distribution of observable values given halo mass

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    obs = expand_dims(obs, -1)
    logMphi = log10(Mh)-logMref
    # this has the same shape as Mh
    phi_s = 10.0**np.sum([bi * logMphi**i for i, bi in enumerate(b)], axis=0)
    return ((phi_s/Mo) * (obs/Mo)**alpha * exp(-(obs/Mo)**2.0))

