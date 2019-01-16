"""HOD population functions"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import array, iterable, newaxis, ones_like, log, log10, moveaxis
from scipy.integrate import trapz

from ..halomodel.tools import Integrate
from ..helpers.decorators import logfunc


@logfunc
def Mh_effective(mass_function, pop_number, Mh, return_log=True):
    """Effective halo mass in each observable bin

    Average halo mass, weighted by the number of objects expected in
    each observable bin, nbar

    Parameters
    ----------
    mass_function : array of floats, shape (nbins,P)
        mass function, dn/dMh. Here, ``nbins`` represents the number
        of observable bins
    pop_number : array of floats, shape (nbins,P)
        expected number of objects given the mass Mh, <N_x|M>
    Mh : array of floats, shape (P,)
        halo mass bins for the integration

    Returns
    -------
    Mh_eff : array of floats, shape (nbins,)
        effective halo mass in each observable bin
    """
    return log10(trapz(Mh*mass_function * pop_number, Mh, axis=-1) \
                 / trapz(mass_function * pop_number, Mh, axis=-1))


def nbar(mass_function, pop_number, Mh):
    """Number of objects expected in each observable bin

    Equal to the expected number of objects integrated over the
    halo mass function

    Parameters
    ----------
    mass_function : array of floats, shape (nbins,P)
        mass function, dn/dMh. Here, ``nbins`` represents the number
        of observable bins
    pop_number : array of floats, shape (nbins,P)
        expected number of objects given the mass Mh, <N_x|M>
    Mh : array of floats, shape (P,)
        halo mass bins for the integration

    Returns
    -------
    nbar : array of floats, shape (nbins,)
        average number of objects in each observable bin
    """
    return trapz(mass_function * pop_number, Mh, axis=-1)


def number(obs, Mh, mor, scatter_func, mor_args, scatter_args, selection=None,
           obs_is_log=False):
    """Expected number of objects (of a given type) of a given mass,
    < N_x | M >

    Parameters
    ----------
    obs : array, shape (nbins,N)
        locations at which the observable is evaluated for the
        trapezoidal rule integration
    Mh : array, shape (P,)
        mean halo mass of interest
    mor : callable
        Mass-observable relation function
    scatter_func : callable
        (Intrinsic) scatter function
    mor_args : array-like
        arguments passed to `mor`
    scatter_args : array-like
        arguments passed to `scatter_func`
    selection : array of floats, shape (nbins,N), optional
        completeness as a function of observable value
    obs_is_log : bool, optional
        whether the observable provided is in log-space

    Returns
    -------
    nc : array, shape (nbins,P)
        number of galaxies for each value of Mh
    """
    if selection is None:
        selection = ones_like(obs)
    prob = probability(
        obs, Mh, mor, scatter_func, mor_args, scatter_args,
        obs_is_log=obs_is_log)
    if obs_is_log:
        obs = 10**obs
    number = moveaxis(
        Integrate(prob*selection[newaxis], obs[newaxis], axis=-1), -1, -2)
    return number


def obs_effective(obs, Mh, mor, scatter_func, mor_args, scatter_args,
                  selection=None, obs_is_log=False):
    """Effective average observable

    Parameters
    ----------
    obs : array, shape (nbins,N)
        locations at which the observable is evaluated for the
        trapezoidal rule integration
    Mh : array, shape (P,)
        mean halo mass of interest
    mor : callable
        Mass-observable relation function
    scatter_func : callable
        (Intrinsic) scatter function
    mor_args : array-like
        arguments passed to `mor`
    scatter_args : array-like
        arguments passed to `scatter_func`

    Returns
    -------
    avg : array, shape (nbins,)
        average observable in each bin in Mh
    """
    if selection is None:
        selection = ones_like(obs)
    prob = probability(
        obs, Mh, mor, scatter_func, mor_args, scatter_args,
        obs_is_log=obs_is_log)
    if obs_is_log:
        obs = 10**obs
    return Integrate(prob*(selection*obs)[newaxis], obs[newaxis], axis=2).T


def probability(obs, Mh, mor, fscatter, mor_args, scatter_args,
                obs_is_log=True):
    """Occupation probability, Phi(obs|M)

    Parameters
    ----------
    obs : array, shape (nbins,N)
        locations at which the observable is evaluated for the
        trapezoidal rule integration. nbins is the number of observable
        bins
    Mh : array, shape (P,)
        mean halo mass of interest
    mor : callable
        Mass-observable relation function
    fscatter : callable
        (Intrinsic) scatter function
    mor_args : array-like
        arguments passed to `mor`
    scatter_args : array-like
        arguments passed to `scatter_func`
    obs_is_log : bool, optional
        whether the observable (and halo mass) are in log space

    Returns
    -------
    phi : array, shape (nbins,P)
        occupation probability for each Mh
    """
    if not iterable(Mh):
        Mh = array([Mh])
    Mo = mor(Mh, *mor_args, return_log=obs_is_log)
    return fscatter(obs, Mo, Mh, *scatter_args, obs_is_log=obs_is_log)

