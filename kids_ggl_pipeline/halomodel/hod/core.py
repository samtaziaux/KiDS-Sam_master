from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import array, iterable

from ..tools import Integrate


"""
# Population functions - average number of galaxies

There should be two files that the user can edit: one with the mass-observable 
scaling and one with the HOD (there could be standard functions with names such as 
'power law'). Then we should identify these in the configuration file

"""



def number(mor, scatter_func, m, Mh, mor_args, scatter_args):
    """Expected number of objects (of a given type) of a given mass,
    < N_x | M >

    Parameters
    ----------
    mor : callable
        Mass-observable relation function
    scatter_func : callable
        (Intrinsic) scatter function
    m : array
        locations at which the integrand is evaluated for the
        trapezoidal rule integration
    Mh : array
        mean halo mass of interest
    mor_args : iterable
        arguments passed to `mor`
    scatter_args : iterable
        arguments passed to `scatter_func`

    Returns
    -------
    nc : array
        number of galaxies for each value of Ms
    """
    prob = probability(mor, scatter_func, m, Mh, mor_args, scatter_args)
    return array([Integrate(pi, m) for pi in prob])


def probability(mor, scatter_func, m, Mh, mor_args, scatter_args):
    """Occupation probability, Phi(obs|M)

    For list of parameters see `number`
    """
    if not iterable(Mh):
        Mh = array([Mh])
    Mo = mor(Mh, *mor_args)
    return scatter_func(m, Mo, *scatter_args)

