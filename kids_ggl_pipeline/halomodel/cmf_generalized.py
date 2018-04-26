from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import sys
from numpy import array, exp, iterable, log, log10, pi
from scipy.integrate import simps, trapz
import scipy.special as sp

if sys.version_info[0] == 3:
    izip = zip
    xrange = range

from .tools import Integrate


"""
# Population functions - average number of galaxies 
# (central or satelites or all) in a halo of mass M - from HOD or CLF!

There should be two files that the user can edit: one with the mass-observable 
scaling and one with the HOD (there could be standard functions with names such as 
'power law'). Then we should identify these in the configuration file

Should also make the scatters independent functions so that they can
be modified if needed. Not critical for now though

"""



def N_given_M(Phi, m, Mh, sigma, mor, mor_args):
    """Expected number of objects (of a given type) of a given mass,

    < N_x | M >

    Parameters
    ----------
    Phi : function
        halo occupation function
    m : array
        locations at which the integrand is evaluated for the
        trapezoidal rule integration
    M : array
        mean halo mass of interest
    sigma : float
        scatter in the halo occupation distribution
    mor : function
        mass-observable relation
    mor_args : tuple
        arguments passed to the mass-observable relation, `mor`

    Returns
    -------
    nc : array
        number of galaxies for each value of M
    """
    Phi_int = Phi(m, Mh, sigma, *mor_args)
    return np.array([Integrate(Phi_i, m) for Phi_i in Phi_int])


def Phi_lognormal(mor, m, Mh, sigma, mor_args):
    """log-normal halo occupation distribution"""
    if not iterable(Mh):
        Mh = array([Mh])
    Mo = mor(Mh, *mor_args)
    return array([exp(-((log10(m/Mi)**2) / (2*sigma**2)) \
                     / ((2*pi)**0.5 * sigma * m * log(10))
                 for Mi in Mo])


def Phi_double_schechter(mor, m, Mh,  alpha, b_0, b_1, b_2, mor_args):
    """Modified Schechter scatter (eq. 17 in van Uitert et al. 2016)

    This is generally used for satellite galaxies, and the mor
    would be `double_powerlaw_scaled`
    """
    if not iterable(Mh):
        Mh = array([Mh])
    Mo = mor(Mh, *mor_args)
    # normalization - note that Mh is normalized by 1e12 as opposed
    # to 1e13 in van Uitert et al.
    phi_s = b_0 + b_1*log10(Mh/1e12) + b_2*log10(Mh/1e12)**2
    return array([phi_s_i * (m/Mi)**(1.+alpha) * exp(-(m/Mi)**2 / m
                  for phi_s_i, Mi in izip(phi_s, Mo)])

