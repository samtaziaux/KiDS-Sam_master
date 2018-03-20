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
from numpy import cos, pi, sin
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp

if sys.version_info[0] == 3:
    xrange = range

from tools import (Integrate, Integrate1, extrap1d, extrap2d, fill_nan,
                   virial_mass, virial_radius)


"""
# NFW profile and corresponding parameters.
"""

def NFW(rho_mean, c, R_array):

    r_max = R_array[-1]
    d_c = NFW_Dc(200.0, c)
    r_s = NFW_RS(c, r_max)

    # When using baryons, rho_mean gets is rho_dm!

    profile = (rho_mean*d_c)/((R_array/r_s)*((1.0+R_array/r_s)**2.0))

    #print ("NFW.")
    #print ("%s %s %s", d_c, r_s, r_max)

    return profile


def NFW_RS(c, r_max):
    #print ("RS.")
    return r_max/c


def NFW_Dc(delta_h, c):
    #print ("DC.")
    return (delta_h*(c**3.0))/(3.0*(np.log(1.0+c)-(c/(1.0+c))))


# Fourier transform of NFW profile - analytic!

def NFW_f(z, rho_mean, f, m_x, r_x, k_x, c=None):
    #if len(m_x.shape) == 0:
        #m_x = np.array([m_x])
        #r_x = np.array([r_x])
    u_k = np.zeros((k_x.size,m_x.size))
    if c is None:
        c = Con(z, m_x, f)
    else:
        c = c * np.ones(m_x.size)
    for i in xrange(m_x.size):
        r_s = NFW_RS(c[i], r_x[i])
        K = k_x*r_s
        bs, bc = sp.sici(K)
        asi, ac = sp.sici((1+c[i]) * K)
        u_k[:,i] = 4.0 * pi * rho_mean * NFW_Dc(200.0, c[i]) * r_s**3.0 * \
                   ((sin(K) * (asi - bs)) - \
                    (sin(c[i]*K) / ((1.0 + c[i]) * K)) + \
                    (cos(K) * (ac - bc))) / m_x[i]
    return u_k


def miscenter(p_off, r_off, m_x, r_x, k_x, c=None):
    
    u_k = np.zeros((k_x.size,m_x.size))
    if c is None:
        c = Con(z, m_x, f)
    else:
        c = c * np.ones(m_x.size)
    for i in xrange(m_x.size):
        r_s = NFW_RS(c[i], r_x[i])
        u_k[:,i] = (1.0 - p_off + p_off*np.exp(-0.5*(k_x**2.0)*(r_s*r_off)**2.0))
    return u_k


def Con(z, M, f):

    #duffy rho_crit
    #c = 6.71 * (M / (2.0 * 10.0 ** 12.0)) ** -0.091 * (1 + z) ** -0.44

    #duffy rho_mean
    c = f * 10.14 * (M / (2.0 * 10.0 ** 12.0)) ** -0.081 * (1.0 + z) ** -1.01

    #maccio08
    #c = 10**0.830 / (M*0.3/(1e12))**0.098

    #zehavi
    #c = ((M / 1.5e13) ** -0.13) * 9.0 / (1 + z)

    #bullock_rescaled
    #c = (M / 10 ** 12.47) ** (-0.13) * 11 / (1 + z)

    #c = c0 * (M/M0) ** b

    #c = f * np.ones(M.shape)
    return c


def delta_NFW(z, rho_mean, f, M, r):

	c = Con(z, M, f)
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


def av_delta_NFW(mass_func, z, rho_mean, f, hod, M, r):

	integ = np.ones((len(M), len(r)))
	average = np.ones(len(r))
	prob = hod*mass_func

	for i in range(len(M)):

		integ[i,:] = delta_NFW(z, rho_mean, f, M[i], r)

	for i in range(len(r)):

		average[i] = Integrate(prob*integ[:,i], M)

	av = average/Integrate(prob, M) # Needs to be normalized!

	return av


"""
# Some bias functions
"""

def Bias(hmf, r_x):
    """
    PS bias - analytic
        
    """
    bias = 1.0+(hmf.nu-1.0)/(hmf.growth*hmf.delta_c)
    #print ("Bias OK.")
    return bias


def Bias_Tinker10(hmf, r_x):
    """
    Tinker 2010 bias - empirical
        
    """
    nu = hmf.nu**0.5
    y = np.log10(hmf.delta_halo)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4
    #print y, A, a, B, b, C, c
    return 1 - A * nu**a / (nu**a + hmf.delta_c**a) + B * nu**b + C * nu**c





"""
# Spectrum 1-halo components for dark matter.
"""

def MM_analy(mass_func, uk, rho_dm, m_x):
    return trapz(m_x * mass_func.dndlnm * uk**2.0,
             m_x, axis=1) / (rho_dm**2.0)


def GM_cen_analy(mass_func, uk, rho_dm, population, ngal, m_x):
    return trapz(mass_func.dndlnm * population * uk,
                 m_x, axis=1) / (rho_dm*ngal)


def GM_sat_analy(mass_func, uk_m, uk_s, rho_dm, population, ngal, m_x):
    return trapz(mass_func.dndlnm * population * uk_m * uk_s,
                 m_x, axis=1) / (rho_dm*ngal)


def GG_cen_analy(mass_func, ncen, ngal, m_x):
    return ncen / (ngal**2.0)


def GG_sat_analy(mass_func, uk_s, population_sat, ngal, beta, m_x):
    return trapz(beta * mass_func.dndm * population_sat**2.0 * uk_s * uk_s,
                 m_x, axis=1) / (ngal**2.0)


def GG_cen_sat_analy(mass_func, uk_s, population_cen, population_sat, ngal, m_x):
    return trapz(mass_func.dndm * population_sat * population_cen * uk_s,
                 m_x, axis=1) / (ngal**2.0)


def DM_mm_spectrum(mass_func, z, rho_dm, rho_mean, n, k_x, r_x, m_x, T):

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


def GM_cen_spectrum(mass_func, z, rho_dm, rho_mean, n, population, \
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


def GM_sat_spectrum(mass_func, z, rho_dm, rho_mean, n, population, \
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

