"""HOD population functions"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import array, iterable, newaxis, ones_like
from scipy.integrate import trapz

from ..halomodel.tools import Integrate



def number(obs, Mh, mor, scatter_func, mor_args, scatter_args, selection=None):
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
    selection : array of floats, shape (nbins,N)
        completeness as a function of observable value

    Returns
    -------
    nc : array, shape (P,)
        number of galaxies for each value of Mh
    """
    if selection is None:
        selection = ones_like(obs)
    prob = probability(obs, Mh, mor, scatter_func, mor_args, scatter_args)
    number = Integrate(prob*selection[newaxis], obs[newaxis], axis=2).T
    return number


def obs_average(obs, Mh, mor, scatter_func, mor_args, scatter_args, selection):
    """Effective average observable

    NOTE: I think this is giving the effective average *halo mass*

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
    avg : array, shape (P,)
        average observable in each bin in Mh
    """
    if selection is None:
        selection = ones_like(obs)
    prob = [probability(oi, Mh, mor, scatter_func, mor_args, scatter_args)
            for oi in obs]
    return array(
        [trapz(p*s*o, o) for p, s, o in zip(prob, selection, obs)])


def probability(obs, Mh, mor, scatter_func, mor_args, scatter_args):
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
    scatter_func : callable
        (Intrinsic) scatter function
    mor_args : array-like
        arguments passed to `mor`
    scatter_args : array-like
        arguments passed to `scatter_func`

    Returns
    -------
    phi : array, shape (nbins,P)
        occupation probability for each Mh
    """
    if not iterable(Mh):
        Mh = array([Mh])
    Mo = mor(Mh, *mor_args)
    return scatter_func(obs, Mo, *scatter_args)

