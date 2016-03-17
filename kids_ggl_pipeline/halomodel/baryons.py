#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  barions.py
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

import time
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp
import halo
import cmf

from tools import Integrate, Integrate1, extrap1d, extrap2d, fill_nan


"""
# --------- Mass Fractions ----------
"""

"""
# FEDELI
"""

def f_dm(omegab, omegac):
    f = 1.0 - (omegab/(omegab+omegac))
    return f


"""
# CMF
"""

def f_stars(m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
            b_0, b_1, b_2, Ac2s):
    """
    m - stellar mass, M - halo mass

    """
    central = cmf.av_cen(m, M, sigma, A, M_1, gamma_1, gamma_2)
    satelite = cmf.av_sat(m, M, alpha, A, M_1, gamma_1, gamma_2,
                          b_0, b_1, b_2, Ac2s)
    return (central + satelite) / M


def f_gas(omegab, omegac, m):
    sigma = 3.0
    m_0 = 10.0**12.26
    x = np.float64(m/m_0)
    #f = omegab/(omegab+omegac) * sp.erf(np.log10(x)/sigma)
    #f = np.piecewise(x, [x < 1.0, x >= 1.0],
                     #[lambda t: 0.0, lambda t: \
                          #omegab/(omegab+omegac) * sp.erf(np.log10(t)/sigma)])
    #Because erf becomes negative for m<m0, we replace all these values with 0!
    f = np.where(x >= 1,
                 omegab/(omegab+omegac) * sp.erf(np.log10(x)/sigma), 0.0)
    return f


"""
# ----------- Normalizations
"""


"""
All normalizations should add up to 1, and one precisely!!!
"""

"""
# FEDELI
"""

def rhoDM(mass_func, M, omegab, omegac):
    integ = mass_func.dndlnm*f_dm(mass_func.omegab, mass_func.omegac)
    return Integrate(integ, M)

def rhoSTARS(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
             b_0, b_1, b_2, Ac2s):
    integ = mass_func.dndlnm * f_stars(m, M, sigma, alpha, A, M_1,
                                       gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)
    return Integrate(integ, M)

def rhoGAS(mass_func, rho_crit, omegab, omegac, m, M, sigma, alpha, A,
           M_1, gamma_1, gamma_2, b_0, b_1, b_2, Ac2s):
    rho = rho_crit * omegab - rhoSTARS(mass_func, m, M, sigma, alpha, A, M_1,
                                       gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)
    integ = mass_func.dndlnm * f_gas(omegab, omegac, M)
    F = 1.0#Integrate(integ, M)/rho
    #~ integ = mass_func.dndlnm*f_gas(omegab, omegac, M)
    #~ rho = Integrate(integ, M)
    #~ F = 1.0
    #print rho, F
    return rho, F


"""
# ----------- Gas and stars density profiles -----------
"""

"""
# FEDELI
"""

def u_g(r_x, beta, r_c0, omegab, omegac, m):
    """
    Beta-model for gas
    Calculates profile for only one mass! Be careful, how to call it!
    r_c can be changed! Fedeli -> 0.05

    """
    r_c = r_c0 * r_x[-1]
    x = r_x/r_c
    x_delta = r_x[-1]/r_c
    rho_c = (3. * m * f_gas(omegab, omegac, m)) / \
            (4. * np.pi * (r_c**3.) * \
             sp.hyp2f1(1.5, (3.*beta) / 2, 5. / 2., - (x_delta**2.0)) * \
             (x_delta**3.))
    return rho_c / ((1 + (x**2.))**(1.5*beta))


def u_s(r_x, alpha, r_t0, m, M, sigma, alpha_hod, A, M_1, gamma_1, gamma_2,
        b_0, b_1, b_2, Ac2s):
    """
    NFW for smaller scales, more to the center, exponential decline for the
    outer regions, which can be varied with alpha parameter!
    r_t can be changed! Fedeli -> 0.03

    """
    r_t = r_t0 * r_x[-1]
    x = r_x/r_t
    x_delta = r_x[-1]/r_t
    # absolute value, yes / no? - still not completely solved! BE AWARE!
    nu_alpha = 1 - (2. / alpha)
    E_alpha = sp.expn(np.absolute(nu_alpha), (x_delta**alpha))
    G_alpha = sp.gamma(1. - nu_alpha)
    rho_t = (M * f_stars(m, M, sigma, alpha_hod, A, M_1, gamma_1, gamma_2,
                         b_0, b_1, b_2, Ac2s) * alpha) / \
            (4 * np.pi * (r_t**3.0) * (G_alpha - (x_delta**2) * E_alpha))
    return (rho_t / x) * np.exp(-(x**alpha))


# --------- Power spectra for gas and stars ------------

"""
# Analytical for stars and gas - for alpha = 1 and beta = 2/3 only!
"""


def star_f(z, rho_mean, m_x, r_x, k_x, r_t0):
    m_x = np.float64(m_x)
    r_x = np.float64(r_x)
    k_x = np.float64(k_x)
    if len(m_x.shape) == 0:
        m_x = np.array([m_x])
        r_x = np.array([r_x])
    alpha = 1.0
    nu_alpha = 1 - (2 / alpha)
    G_alpha = sp.gamma(1 - nu_alpha)
    u_k = np.zeros((k_x.size, m_x.size))
    #def f_stars(m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
                #b_0, b_1, b_2, Ac2s)
    for i in xrange(m_x.size):
        r_t = r_t0[i] * r_x[i]
        x = r_x[i]/r_t
        x_delta = r_x[i]/r_t
        K = r_t * k_x
        E_alpha = sp.expn(np.absolute(nu_alpha), (x_delta ** alpha))
        rho_t = (m_x[i] * f_stars(m_x, np.array([m_x[i]]), 1.0) * alpha) / \
                (4 * np.pi * (r_t**3) * (G_alpha - (x_delta**2) * E_alpha))
        u_k[:,i] = (4.0 * np.pi * (r_t**3.0) * rho_t) / \
                   (m_x[i] * (1.0 + (K**2.0)))
    return u_k


def gas_f(z, rho_mean, m_x, r_x, k_x, r_c0):
    m_x = np.float64(m_x)
    r_x = np.float64(r_x)
    k_x = np.float64(k_x)
    if len(m_x.shape) == 0:
        m_x = np.array([m_x])
        r_x = np.array([r_x])
    u_k = np.zeros((k_x.size, m_x.size))
    beta = 2./3
    for i in xrange(m_x.size):
        r_c = r_c0[i] * r_x[i]
        x = r_x[i]/r_c
        x_delta = r_x[i]/r_c
        K = r_c * k_x
        rho_c = (3.0 * m_x[i] * \
                f_gas(0.0455, 0.226, m_x[i])) / \
                (4 * np.pi * (r_c**3.0) * \
                (3 * x_delta - 3 * np.arctan(x_delta)))
        u_k[:,i] = (4 * (np.pi**2) * (r_c**3) * rho_c * np.exp(-K)) / \
                   (2 * m_x[i] * K)
    return u_k


def GS_cen_analy(mass_func, z, rho_stars, rho_mean, n, population, ngal,
                 k_x, r_x, m_x, T_stars, T_tot, r_t):
    spec = np.ones(k_x.size)
    integ = np.ones((k_x.size, m_x.size))
    u_k = star_f(z, rho_mean, m_x, r_x, k_x, r_t)
    for i in xrange(k_x.size):
        integ[i,:] = mass_func.dndlnm * population * u_k[i,:]
        spec[i] = Integrate(integ[i,:], m_x)
    return spec / (rho_stars * ngal)


def Ggas_cen_analy(mass_func, z, F, rho_gas, rho_mean, n, population, ngal,
                   k_x, r_x, m_x, T_g, T_tot, r_c):
    spec = np.ones(k_x.size)
    integ = np.ones((k_x.size, m_x.size))
    u_k = gas_f(z, rho_mean, m_x, r_x, k_x, r_c)
    for i in xrange(k_x.size):
        integ[i,:] = mass_func.dndlnm * population * u_k[i,:]
        spec[i] = Integrate(integ[i,:], m_x)
    return spec / (rho_gas * ngal)

"""
# --------- Stars --------------------------------------
"""


def GS_cen_spectrum(mass_func, z, rho_stars, rho_mean, n, population, ngal,
                    k_x, r_x, m_x, T_stars, T_tot):
    """
    Calculates the power spectrum for the component given in the name.
    Following the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small it is not worth
    doing it.
    Extrapolates the power spectrum to get rid of the knee, which is a Taylor
    series artifact.
    """
    n = n + 2
    k_x = np.longdouble(k_x)
    #T = np.ones((n/2, m_x.size))
    spec = np.ones(k_x.size)
    integ = np.ones((n/2, m_x.size))
    T_comb = np.ones((n/2, m_x.size))
    comp = np.ones((n/2, k_x.size), dtype=np.longdouble)

    # Calculating all the needed T's!
    """
    for i in xrange(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/(T_tot[0,:])

    for k in xrange(0, n/2, 1):
        T_comb[k,:] = T_stars[k,:]
        integ[k,:] = norm * (population * mass_func.dndlnm * T_comb[k,:]) / \
                     (rho_stars * ngal)
        comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2)) * (-1)**k
    spec = np.sum(comp, axis=0)
    spec[spec >= 10.0**10.0] = np.nan
    spec[spec <= 0.0] = np.nan
    return extrap1d(np.float64(k_x), np.float64(spec), 0.01, 3)


def GS_sat_spectrum(mass_func, z, rho_stars, rho_mean, n, population, ngal,
                    k_x, r_x, m_x, T_dm, T_s, T_tot):

    """
    Calculates the power spectrum for the component given in the name.
    Following the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small it is not worth
    doing it.
    Extrapolates the power spectrum to get rid of the knee, which is a Taylor
    series artifact.

    """
    n = n + 2
    k_x = np.longdouble(k_x)
    #T = np.ones((n/2, m_x.size))
    spec = np.ones(k_x.size)
    integ = np.ones((n/2, m_x.size))
    T_comb1 = np.ones((n/2, m_x.size))
    T_comb2 = np.ones((n/2, m_x.size))
    comp = np.ones((n/2, k_x.size), dtype=np.longdouble)

    # Calculating all the needed T's!
    """
    for i in xrange(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/((T_tot[0,:])**2.0)
    for k in xrange(0, n/2, 1):
        T_combined1 = np.ones((k+1, m_x.size))
        T_combined2 = np.ones((k+1, m_x.size))
        for j in xrange(0, k+1, 1):
            T_combined1[j,:] = T_dm[j,:] * T_s[k-j,:]
        T_comb1[k,:] = np.sum(T_combined1, axis=0)
        T_comb2[k,:] = 1.0#np.sum(T_combined2, axis=0)
        integ[k,:] = norm * (population * mass_func.dndlnm * T_comb1[k,:] * \
                             T_comb2[k,:]) / (rho_stars * ngal)
        comp[k,:] = Integrate(integ[k,:], m_x) * (k_x**(k*2)) * (-1)**(k)
    spec = np.sum(comp, axis=0)
    spec[spec >= 10.0**10.0] = np.nan
    spec[spec <= 0.0] = np.nan
    return extrap1d(np.float64(k_x), np.float64(spec), 0.01, 2)

def GS_TwoHalo(mass_func, z, rho_stars, rho_mean, n, k_x, r_x, m_x,
               T_s, T_tot):
    """
    # This is ok!
    Calculates the power spectrum for the component given in the name.
    Following the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small it is not worth
    doing it.
    Extrapolates the power spectrum to get rid of the knee, which is a Taylor
    series artifact.
    """

    """
    Norm = rho_stars in this case!
    """
    n = n + 2
    k_x = np.longdouble(k_x)
    #T = np.ones((n/2, m_x.size))
    spec = np.ones(k_x.size)
    integ = np.ones((n/2, m_x.size))
    T_comb = np.ones((n/2, m_x.size))
    comp = np.ones((n/2, k_x.size))

    # Calculating all the needed T's!
    """
    for i in xrange(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/(T_tot[0,:])
    for k in xrange(0, n/2, 1):
        T_comb[k,:] = T_s[k,:]
        # Divided by m or not?
        integ[k,:] = norm * (mass_func.dndlnm * T_comb[k] * \
                     halo.Bias_Tinker10(mass_func,r_x)) / \
                     (rho_stars)#/m_x)/(rho_stars)
        comp[k,:] = Integrate(integ[k], m_x) * (k_x**(k*2)) * (-1)**k

    spec = np.sum(comp, axis=0)
    spec[spec >= 10.0**10.0] = np.nan
    spec[spec <= 0.0] = np.nan
    spec_ext = extrap1d(np.float64(k_x), np.float64(spec), 0.01, 2)

    #P2 = (np.exp(mass_func.power)/norm) * \
         #(Integrate((mass_func.dndlnm * T_stars2 * \
          #Bias_Tinker10(mass_func,r_x) / m_x), m_x))
    #print (Integrate((mass_func.dndlnm * population * \
            #Bias_Tinker10(mass_func,r_x) / m_x), m_x)) / norm
    print ("Two halo term calculated - stars.")
    return spec_ext * np.exp(mass_func.power)


"""
# ------------ Gas ----------------------
"""


def GGas_cen_spectrum(mass_func, z, F, rho_gas, rho_mean, n, population,
                      ngal, k_x, r_x, m_x, T_g, T_tot):

    """
    Calculates the power spectrum for the component given in the name. Following
    the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small it is not worth doing
    it.
    Extrapolates the power spectrum to get rid of the knee, which is a Taylor
    series artifact.
    """

    n = n + 2
    k_x = np.longdouble(k_x)

    #T = np.ones((n/2, m_x.size))
    spec = np.ones(k_x.size)
    integ = np.ones((n/2, m_x.size))
    T_comb = np.ones((n/2, m_x.size))
    comp = np.ones((n/2, k_x.size), dtype=np.longdouble)

    # Calculating all the needed T's!
    """
    for i in xrange(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/(T_tot[0,:])
    for k in xrange(n/2):
        T_comb[k] = T_g[k]
        integ[k] = norm * (population * mass_func.dndlnm * T_comb[k]) / \
                     (F * rho_gas * ngal)
        comp[k] = Integrate(integ[k], m_x) * (k_x**(k*2)) * (-1)**k

    spec = np.sum(comp, axis=0)
    spec[spec >= 1e10] = np.nan
    spec[spec <= 0.0] = np.nan
    return extrap1d(np.float64(k_x), np.float64(spec), 0.002, 2)

def GGas_sat_spectrum(mass_func, z, F, rho_gas, rho_mean, n, population,
                      ngal, k_x, r_x, m_x, T_dm, T_g, T_tot):
    """
    Calculates the power spectrum for the component given in the name. Following
    the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small it is not worth doing
    it.
    Extrapolates the power spectrum to get rid of the knee, which is a Taylor
    series artifact.
    """

    n = n + 2
    k_x = np.longdouble(k_x)
    #T = np.ones((n/2, m_x.size))
    spec = np.ones(k_x.size)
    integ = np.ones((n/2, m_x.size))
    T_comb1 = np.ones((n/2, m_x.size))
    T_comb2 = np.ones((n/2, m_x.size))
    comp = np.ones((n/2, k_x.size), dtype=np.longdouble)

    # Calculating all the needed T's!
    """
    for i in xrange(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/((T_tot[0])**2)
    for k in xrange(n/2):
        #print ('k:'), k
        #print ('-----------')
        T_combined1 = np.ones((k+1, m_x.size))
        T_combined2 = np.ones((k+1, m_x.size))
        for j in xrange(k+1):
            T_combined1[j] = T_dm[j] * T_g[k-j]
        T_comb1[k] = np.sum(T_combined1, axis=0)
        T_comb2[k] = 1.0#T_g[k,:]#np.sum(T_combined2, axis=0)
        integ[k] = norm * (population * mass_func.dndlnm * T_comb1[k] * \
                             T_comb2[k]) / (F * rho_gas * ngal)
        comp[k] = Integrate(integ[k], m_x) * (k_x**(k*2)) * (-1)**k

    spec = np.sum(comp, axis=0)
    spec[spec >= 1e10] = np.nan
    spec[spec <= 0.0] = np.nan
    return extrap1d(np.float64(k_x), np.float64(spec), 0.002, 2)

def GGas_TwoHalo(mass_func, z, F, rho_gas, rho_mean, n, k_x, r_x, m_x,
                 T_g, T_tot): # This is ok!
    """
    Calculates the power spectrum for the component given in the name. Following
    the construction from Mohammed, but to general power of k!
    In practice the contributions from k > 50 are so small it is not worth doing
    it.
    Extrapolates the power spectrum to get rid of the knee, which is a Taylor
    series artifact.
    """

    """
    Norm = rho_stars in this case!
    """

    n = n + 2
    k_x = np.longdouble(k_x)
    #T = np.ones((n/2, m_x.size))
    spec = np.ones(k_x.size)
    integ = np.ones((n/2, m_x.size))
    T_comb = np.ones((n/2, m_x.size))
    comp = np.ones((n/2, k_x.size))

    # Calculating all the needed T's!
    """
    for i in xrange(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x)
    """
    norm = 1.0/(T_tot[0])
    for k in xrange(n/2):
        T_comb[k] = T_g[k]
        # Divided by m or not?
        integ[k] = norm * (mass_func.dndlnm * T_comb[k] * \
                             halo.Bias_Tinker10(mass_func,r_x)) / \
                     (F * rho_gas)#/m_x)/(rho_stars)
        comp[k] = Integrate(integ[k], m_x) * (k_x**(k*2)) * (-1)**k

    spec = np.sum(comp, axis=0)
    spec[spec >= 1e10] = np.nan
    spec[spec <= 0.0] = np.nan
    spec_ext =  extrap1d(np.float64(k_x), np.float64(spec), 0.01, 2)

    #P2 = (np.exp(mass_func.power) / norm) * \
         #(Integrate((mass_func.dndlnm * T_stars2 * \
          #Bias_Tinker10(mass_func,r_x) / m_x), m_x))
    #print Integrate((mass_func.dndlnm * population * \
           #Bias_Tinker10(mass_func,r_x) / m_x), m_x) / norm
    print ("Two halo term calculated - gas.")
    return spec_ext*np.exp(mass_func.power)


if __name__ == '__main__':
    main()
