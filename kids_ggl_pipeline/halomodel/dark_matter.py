#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  dark.py
#
#  Copyright 2014 Andrej Dvornik <dvornik@dommel.strw.leidenuniv.nl>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
import matplotlib.pyplot as pl
import numpy as np
import sys
from numpy import cos, expand_dims, pi, sin, piecewise
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp

if sys.version_info[0] == 3:
    xrange = range

from .tools import (Integrate, Integrate1, extrap1d, extrap2d, fill_nan,
                    virial_mass, virial_radius)


"""
# NFW profile and corresponding parameters.
# All these NFW functions should either be moved to the nfw.py or removed from here. Still some cleaning to do!


"""

def nfw(rho_mean, c, R_array):

    r_max = R_array[-1]
    d_c = nfw_dc(200.0, c)
    r_s = nfw_rs(c, r_max)

    # When using baryons, rho_mean gets is rho_dm!

    profile = (rho_mean*d_c)/((R_array/r_s)*((1.0+R_array/r_s)**2.0))

    #print ("NFW.")
    #print ("%s %s %s", d_c, r_s, r_max)

    return profile


def nfw_rs(c, r_max):
    #print ("RS.")
    return r_max/c


def nfw_dc(delta_h, c):
    #print ("DC.")
    return (delta_h*(c**3.0))/(3.0*(np.log(1.0+c)-(c/(1.0+c))))


# Fourier transform of NFW profile - analytic!

def nfw_f(z, rho_mean, f, m_x, r_x, k_x, c=None):
    #if len(m_x.shape) == 0:
        #m_x = np.array([m_x])
        #r_x = np.array([r_x])
    u_k = np.zeros((k_x.size,m_x.size))
    if c is None:
        c = con(z, m_x, f)
    else:
        c = c * np.ones(m_x.size)
    for i in xrange(m_x.size):
        r_s = nfw_rs(c[i], r_x[i])
        K = k_x*r_s
        bs, bc = sp.sici(K)
        asi, ac = sp.sici((1+c[i]) * K)
        u_k[:,i] = 4.0 * pi * rho_mean * NFW_Dc(200.0, c[i]) * r_s**3.0 * \
                   ((sin(K) * (asi - bs)) - \
                    (sin(c[i]*K) / ((1.0 + c[i]) * K)) + \
                    (cos(K) * (ac - bc))) / m_x[i]
    return u_k


def miscenter(p_off, r_off, m_x, r_x, k_x, c=None, shape=None):
    if shape is None:
        shape = (k_x.size, m_x.size)
    u_k = np.zeros(shape)
    if c is None:
        c = con(z, m_x, f)
    #else:
        #c = c * np.ones(m_x.size)
    rs = nfw_rs(c, r_x)
    u_k = 1 - p_off + p_off*np.exp(-0.5*(k_x**2)*(rs*r_off)**2)
    return u_k


def con(z, M, f, scaling='duffy08'):
    #duffy rho_crit
    #c = f * 6.71 * (M/2e12)**-0.091 * (1+z)**-0.44
    #duffy rho_mean
    c = f * 10.14 / (M/2e12)**0.081 / (1+z)**1.01
    #maccio08
    #c = 10**0.830 / (M*0.3/(1e12))**0.098
    #zehavi
    #c = ((M / 1.5e13) ** -0.13) * 9.0 / (1 + z)
    #bullock_rescaled
    #c = (M / 10 ** 12.47) ** (-0.13) * 11 / (1 + z)
    #c = c0 * (M/M0) ** b
    #c = f * np.ones(M.shape)
    # more general version -- these are all calculated at delta=200
    """
    cM = {'duffy08':
            {'crit': 6.71 / (M/2e12/h)**0.091 / (1+z)**0.44,
             'mean': 10.14 / (M/2e12/h)**0.081 / (1+z)**1.01},
          'maccio08':
            {'crit': 10**0.83 / (M/1e12/h)**0.098},
          'dutton14':
            {'crit': 10**0.905 / (M/1e12/h)**0.101}
         }
    """
    return c


def delta_nfw(z, rho_mean, f, M, r):

    c = con(z, M, f)
    r_vir = virial_radius(M, rho_mean, 200.0)
    r_s = NFW_RS(c, r_vir)
    d_c = NFW_Dc(200.0, c)

    x = r/r_s

    g = np.ones(len(x))

    for i in range(len(x)):
        if x[i]<1.0:
            g[i] = (8.0*np.arctanh(np.sqrt((1.0 - x[i])/(1.0 + x[i]))))/ \
            ((x[i]**2.0)*np.sqrt(1.0 - x[i]**2.0)) + (4.0*np.log(x[i]/2.0))/ \
            (x[i]**2.0) - 2.0/(x[i]**2.0 - 1.0) + (4.0*np.arctanh(np.sqrt((1.0- \
            x[i])/(1.0 + x[i]))))/((x[i]**2.0 - 1.0)*np.sqrt(1.0 - x[i]**2.0))

        elif x[i]==1.0:
            g[i] = 10.0/3.0 + 4.0*np.log(0.5)

        elif x[i]>=1.0:
            g[i] = (8.0*np.arctan(np.sqrt((x[i] - 1.0)/(1.0 + x[i]))))/ \
            ((x[i]**2.0)*np.sqrt(x[i]**2.0 - 1.0)) + (4.0*np.log(x[i]/2.0))/ \
            (x[i]**2.0) - 2.0/(x[i]**2.0 - 1.0) + (4.0*np.arctan(np.sqrt((x[i]- \
            1.0)/(1.0 + x[i]))))/((x[i]**2.0 - 1.0)**(3.0/2.0))


    return r_s * d_c * rho_mean * g


def av_delta_nfw(mass_func, z, rho_mean, f, hod, M, r):

    integ = np.ones((len(M), len(r)))
    average = np.ones(len(r))
    prob = hod*mass_func

    for i in range(len(M)):

        integ[i,:] = delta_nfw(z, rho_mean, f, M[i], r)

    for i in range(len(r)):

        average[i] = Integrate(prob*integ[:,i], M)

    av = average/Integrate(prob, M) # Needs to be normalized!

    return av


"""
# Some bias functions
"""

def bias_ps(hmf, r_x):
    """
    PS bias - analytic
        
    """
    bias = 1.0+(hmf.nu-1.0)/(hmf.growth*hmf.delta_c)
    return bias


def bias_tinker10_func(hmf):
    """
    Tinker 2010 bias - empirical
        
    """
    nu = hmf.nu**0.5
    y = np.log10(hmf.mdef_params['overdensity'])
    A = 1.0 + 0.24 * y * np.exp(-(4. / y) ** 4.)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4. / y) ** 4.)
    c = 2.4
    #print y, A, a, B, b, C, c
    return 1 - A * nu**a / (nu**a + hmf.delta_c**a) + B * nu**b + C * nu**c


def bias_tinker10(hmf):
    """
    Tinker 2010 bias - empirical
        
    """
    nu0 = hmf.nu**0.5
    min, max, step = hmf.Mmin, hmf.Mmax, hmf.dlog10m
    min_0 = 2.
    hmf.update(Mmin = min_0, Mmax = max, dlog10m = (max-min_0)/500.0)
    nu = hmf.nu**0.5
    
    f_nu = hmf.fsigma / nu
    b_nu = bias_tinker10_func(hmf)
    norm = trapz(f_nu * b_nu, nu)
    
    func = bias_tinker10_func(hmf)
    func_i = interp1d(nu, func, bounds_error=False, fill_value='extrapolate')
    func = func_i(nu0)
    hmf.update(Mmin = min, Mmax = max, dlog10m = step)
    return func / norm
    

def two_halo_gm(hmf, ngal, population, m_x, **kwargs):
    """
    Note that I removed the argument k_x which was required but not
    used
    """
    #print('in twohalo:')
    #print('dndlnm =', hmf.dndlnm.shape)
    #if np.iterable(ngal):
        #print('ngal =', ngal.shape)
    #else:
        #print('ngal =', ngal)
    #print('population =', population.shape)
    #print('r_x =', r_x.shape)
    #print('m_x =', m_x.shape)
    #print('integrand =', (hmf.dndlnm * population * Bias_Tinker10(hmf, r_x) / m_x).shape)
    b_g = trapz(
        hmf.dndm * population * bias_tinker10(hmf),
        m_x, **kwargs) / ngal
    #print('b_g =', b_g.shape)#, b_g[:,None].shape)
    #try:
        #print('output =', (hmf.power*b_g[:,None]).shape)
    #except ValueError:
        #print('output =', (hmf.power*b_g).shape)
    #print()
    #return (hmf.power * np.expand_dims(b_g, -1)), b_g
    return (hmf.power * b_g), b_g
    
    
def two_halo_gg(hmf, ngal, population, m_x, **kwargs):
    """
    Note that I removed the argument k_x which was required but not
    used
    """
    b_g = trapz(
        hmf.dndm * population * bias_tinker10(hmf),
        m_x, **kwargs) / ngal

    return (hmf.power * b_g**2.0), b_g**2.0


def halo_exclusion(xi, r, meff, rho_dm, delta):
    """ Halo exclusion function that makes sure that in 2-halo term
        neighbouring haloes do not overlap and are at least Rvir apart
        from each other. This prescription follows Giocoly et al. 2010
        but maybe a better model might be implemented
    
    Parameters
    ----------
    xi : float array, shape (nbins,P)
        2-halo correlation function
    r : float array, shape (P,)
        x-axis array of radial values for xi
    meff : float array, shape (nbins,)
        average halo mass in observable bin
    rho_dm : float or float array, shape (nbins,)
        average background density
    delta : float
        overdensity factor

    Returns
    -------
    xi_exc : float array, shape (nbins,P)
        modified 2-halo correlation function
    """
    rvir = virial_radius(10.0**meff, rho_dm, delta)
    #filter = np.array([piecewise(r, [r <= rvir_i, r > rvir_i], [0.0, 1.0]) for rvir_i in rvir])
    filter = np.array([sp.erf(r/rvir_i) for rvir_i in rvir]) # Implemented with err function to smooth out step! 
    #xi_exc = ((1.0 + xi) * filter) - 1.0 # Given how sigma function in lens.py is coded up, the simple method is fine!
    xi_exc = xi * filter
    return xi_exc


"""
# Spectrum 1-halo components for dark matter.
"""

def mm_analy(dndm, uk, rho_dm, Mh):
    """Analytical calculation of the matter-matter power spectrum

    Uses the trapezoidal rule to integrate using halo mass samples

    Parameters
    ----------
    dndm : float array, shape (P,)
        differential mass function
    uk : float array, shape (P,K)
        Fourier transform of the matter density profile
    rho_dm : float or float array, shape (nbins,)
        average background density
    Mh : float array, shape (P,)
        halo mass samples over which to integrate

    Returns
    -------
    Pk : float array, shape (nbins,K)
        Central galaxy-matter power spectrum

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    norm = rho_dm
    if np.iterable(rho_dm):
        rho_dm = expand_dims(rho_dm, -1)
    return trapz(expand_dims(Mh**2.0 * dndm, -1) * uk**2.0,
             Mh, axis=1) / (rho_dm**2.0)


def gm_cen_analy(dndm, uk, rho_dm, population, ngal, Mh):
    """Analytical calculation of the galaxy-matter power spectrum

    Uses the trapezoidal rule to integrate using halo mass samples

    Parameters
    ----------
    dndm : float array, shape (P,)
        differential mass function
    uk : float array, shape (P,K)
        Fourier transform of the matter density profile
    rho_dm : float or float array, shape (nbins,)
        average background density
    population : float array, shape (nbins,P)
        occupation probability as a function of halo mass
    ngal : float or float array, shape (nbins,)
        number of galaxies in each observable bin
    Mh : float array, shape (P,)
        halo mass samples over which to integrate

    Returns
    -------
    Pk : float array, shape (nbins,K)
        Central galaxy-matter power spectrum

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    norm = rho_dm * ngal
    if np.iterable(norm):
        norm = expand_dims(norm, -1)
    dndlnm = dndm * Mh
    if len(population.shape) >= 3:
        dndlnm = expand_dims(dndlnm, -2)
        uk = expand_dims(uk, -3)
    #return trapz(expand_dims(dndlnm*population, -1) * expand_dims(uk, -3),
    return trapz(expand_dims(dndlnm*population, -1) * uk,
                 Mh, axis=-2) / norm

def gm_sat_analy(dndm, uk_m, uk_s, rho_dm, population, ngal, Mh):
    """Analytical calculation of the galaxy-matter power spectrum

    Uses the trapezoidal rule to integrate using halo mass samples

    Parameters
    ----------
    dndm : float array, shape (P,)
        differential mass function
    uk_m : float array, shape (P,K)
        Fourier transform of the matter density profile
    uk_s : float array, shape (P,K)
        Fourier transform of the satellite galaxy density profile
    rho_dm : float or float array, shape (nbins,)
        average background density
    population : float array, shape (nbins,P)
        occupation probability as a function of halo mass
    ngal : float or float array, shape (nbins,)
        number of galaxies in each observable bin
    Mh : float array, shape (P,)
        halo mass samples over which to integrate

    Returns
    -------
    Pk : float array, shape (nbins,K)
        Central galaxy-matter power spectrum

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    return gm_cen_analy(dndm, uk_m*uk_s, rho_dm, population, ngal, Mh)


def gg_cen_analy(dndm, ncen, ngal, shape, Mh):
    """Analytical calculation of the galaxy-galaxy power spectrum

    Uses the trapezoidal rule to integrate using halo mass samples

    Parameters
    ----------
    dndm : float array, shape (P,)
        differential mass function
    ncen : float or float array, shape (nbins,)
        number of central galaxies in each observable bin
    ngal : float or float array, shape (nbins,)
        number of galaxies in each observable bin
    shape : tuple of integers, (nbins,K)
        shape of resulting power spectra
    Mh : float array, shape (P,)
        halo mass samples over which to integrate

    Returns
    -------
    Pk : float array, shape (nbins,K)
        Central galaxy-galaxy power spectrum

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    return np.ones(shape) * ncen / (ngal**2.0)


def gg_sat_analy(dndm, uk_s, population_sat, ngal, beta, Mh):
    """Analytical calculation of the galaxy-gaalxy power spectrum

    Uses the trapezoidal rule to integrate using halo mass samples

    Parameters
    ----------
    dndm : float array, shape (P,)
        differential mass function
    uk_s : float array, shape (P,K)
        Fourier transform of the satellite galaxy density profile
    population_sat : float array, shape (nbins,P)
        occupation probability of satellite galaxies as a function of halo mass
    ngal : float or float array, shape (nbins,)
        number of galaxies in each observable bin
    beta : float or float array, shape (nbins,)
        Poisson parameter
    Mh : float array, shape (P,)
        halo mass samples over which to integrate

    Returns
    -------
    Pk : float array, shape (nbins,K)
        Central galaxy-matter power spectrum

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    if np.iterable(ngal):
        ngal = expand_dims(ngal, -1)
    if np.iterable(beta):
        beta = expand_dims(beta, -1)
    if len(population_sat.shape) >= 3:
        dndm = expand_dims(dndm, -2)
        uk_s = expand_dims(uk_s, -3)

    return trapz(beta * expand_dims(dndm*population_sat**2.0, -1) * uk_s**2.0,
                 Mh, axis=1) / (ngal**2.0)


def gg_cen_sat_analy(dndm, uk_s, population_cen, population_sat, ngal, Mh):
    """Analytical calculation of the galaxy-gaalxy power spectrum

    Uses the trapezoidal rule to integrate using halo mass samples

    Parameters
    ----------
    dndm : float array, shape (P,)
        differential mass function
    uk_s : float array, shape (P,K)
        Fourier transform of the satellite galaxy density profile
    population_cen : float array, shape (nbins,P)
        occupation probability of central galaxies as a function of halo mass
    population_sat : float array, shape (nbins,P)
        occupation probability of satellite galaxies as a function of halo mass
    ngal : float or float array, shape (nbins,)
        number of galaxies in each observable bin
    Mh : float array, shape (P,)
        halo mass samples over which to integrate

    Returns
    -------
    Pk : float array, shape (nbins,K)
        Central galaxy-matter power spectrum

    Notes
    -----
        There may be additional dimensions before the ones mentioned
        above, such as one for redshift distributions. These will be
        carried through and will remain at their location.
    """
    if np.iterable(ngal):
        ngal = expand_dims(ngal, -1)
    if len(population_sat.shape) >= 3:
        dndm = expand_dims(dndm, -2)
        uk_s = expand_dims(uk_s, -3)

    return trapz(expand_dims(dndm*population_sat*population_cen, -1) * uk_s,
                 Mh, axis=1) / (ngal**2.0)


def dm_mm_spectrum(mass_func, z, rho_dm, rho_mean, n, k_x, r_x, m_x, T):

    """
    Calculates the power spectrum for the component given in the name. 
    Following the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small 
    it is not worth doing it.
    Extrapolates the power spectrum to get rid of the knee, 
    which is a Taylor series artifact.
    """

    n = n + 2

    k_x = np.longdouble(k_x)

    #T = np.ones((n/2, len(m_x)))
    spec = np.ones(len(k_x))
    integ = np.ones((n/2, len(m_x)))
    T_comb = np.ones((n/2, len(m_x)))
    comp = np.ones((n/2, len(k_x)), dtype=np.longdouble)

    # Calculating all the needed T's!
    """
    for i in range(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/((T[0,:])**2.0)

    for k in range(0, n/2, 1):
        T_combined = np.ones((k+1, len(m_x)))

        for j in range(0, k+1, 1):

            T_combined[j,:] = T[j,:] * T[k-j,:]

        T_comb[k,:] = np.sum(T_combined, axis=0)

        #print T_comb[k,:]

        integ[k,:] = norm*(m_x*mass_func.dndlnm*T_comb[k,:])/((rho_dm**2.0))
        comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2.0)) * (-1.0)**(k)

    spec = np.sum(comp, axis=0)
    spec[spec >= 10.0**10.0] = np.nan
    spec[spec <= 0.0] = np.nan

    spec_ext = extrap1d(np.float64(k_x), np.float64(spec), 0.001, 2)

    return spec_ext


def gm_cen_spectrum(mass_func, z, rho_dm, rho_mean, n, population, \
                    ngal, k_x, r_x, m_x, T, T_tot):

    """
    Calculates the power spectrum for the component given in the name. 
    Following the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small 
    it is not worth doing it.
    Extrapolates the power spectrum to get rid of the knee, 
    which is a Taylor series artifact.
    """

    n = n + 2

    k_x = np.longdouble(k_x)

    #T = np.ones((n/2, len(m_x)))
    spec = np.ones(len(k_x))
    integ = np.ones((n/2, len(m_x)))
    T_comb = np.ones((n/2, len(m_x)))
    comp = np.ones((n/2, len(k_x)), dtype=np.longdouble)

    # Calculating all the needed T's!
    """
    for i in range(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/(T_tot[0,:])
    for k in xrange(n/2):
        integ[k,:] = norm*(population*mass_func.dndlnm*T[k,:])/(rho_dm*ngal)
        comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2.0)) * (-1.0)**(k)
    spec = np.sum(comp, axis=0)
    spec[(spec >= 1e10) | (spec <= 0.0)] = np.nan
    spec_ext = extrap1d(np.float64(k_x), np.float64(spec), 0.001, 3)#0.001,3
    return spec_ext


def gm_sat_spectrum(mass_func, z, rho_dm, rho_mean, n, population, \
                    ngal, k_x, r_x, m_x, T, T_tot):

    """
    Calculates the power spectrum for the component given in the name. 
    Following the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small 
    it is not worth doing it.
    Extrapolates the power spectrum to get rid of the knee, 
    which is a Taylor series artifact.
    """

    n = n + 2

    k_x = np.longdouble(k_x)

    #T = np.ones((n/2, len(m_x)))
    spec = np.ones(len(k_x))
    integ = np.ones((n/2, len(m_x)))
    T_comb = np.ones((n/2, len(m_x)))
    comp = np.ones((n/2, len(k_x)), dtype=np.longdouble)

    # Calculating all the needed T's!
    """
    for i in range(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/((T_tot[0,:])**2.0)

    for k in range(0, n/2, 1):
        T_combined = np.ones((k+1, len(m_x)))

        for j in range(0, k+1, 1):

            T_combined[j,:] = T[j,:] * T[k-j,:]

        T_comb[k,:] = np.sum(T_combined, axis=0)

        integ[k,:] = norm*(population*mass_func.dndlnm*T_comb[k,:])/(rho_dm*ngal)
        comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2.0)) * (-1.0)**(k)

    spec = np.sum(comp, axis=0)
    spec[spec >= 10.0**10.0] = np.nan
    spec[spec <= 0.0] = np.nan

    spec_ext = extrap1d(np.float64(k_x), np.float64(spec), 0.001, 2)#0.001,2


    return spec_ext

if __name__ == '__main__':
    main()

