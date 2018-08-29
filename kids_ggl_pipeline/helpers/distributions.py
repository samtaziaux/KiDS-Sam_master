"""Set of generic distributions

This module contains generic distributions to be used in different
components of the halo model. Methods here take *both* observable and
halo mass to return a probability distribution, as opposed to methods
in `functions.py` which return one given the other.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import array, exp, log, log10, pi

from .decorators import logdist


@logdist
def lognormal(obs, M, sigma, obs_is_log=False):
    """Log-normal scatter

    Parameters
    ----------
    obs : float array, shape (nbins,N)
        observable. ``nbins`` is the number of observable bins
    M : float array, shape (P,)
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
    distribution : float array, shape (nbins,P,N)
        lognormal distribution of the observable about halo mass
    """
    return array([exp(-((log10(obs/Mi)**2) / (2*sigma**2))) \
                     / ((2*pi)**0.5 * sigma * obs * log(10))
                  for Mi in M])


@logdist
def schechter_mod(obs, M, logMref, alpha, b0, b1, b2, obs_is_log=False):
    """Modified Schechter scatter (eq. 17 in van Uitert et al. 2016)

    This is generally used for satellite galaxies, and the mor
    would be `double_powerlaw_scaled`

    Parameters
    ----------
    obs : float array, shape (nbins,N)
        observable. ``nbins`` is the number of observable bins
    M : float array, shape (P,)
        halo mass
    logMref : float
        log of the characteristic mass
    alpha : float
        low-mass slope
    b0, b1, b2 : floats
        polynomial terms for phi_s, equation 18 in van Uitert+16
        whether the observable is in log-space. Note that if this is
        the case then ``hod.probability`` will pass logM here, so
        it is assumed that a log-space observable implies a log-space
        halo mass.

    Returns
    -------
    distribution : float array, shape (nbins,P,N)
        distribution of observable values given halo mass
    """
    #phi_s = 10**(b[:,newaxis] * logMref**arange(b.size)[:,newaxis])
    phi_s = 10**(b0 + b1*logMref + b2*logMref**2)
    return array([phi_s_i/obs * (obs/Mi)**alpha * exp(-(obs/Mi)**2)
                  for phi_s_i, Mi in zip(phi_s, M)])

