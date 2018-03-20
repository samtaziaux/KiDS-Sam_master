#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  halo.py
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

# Halo model code
# Andrej Dvornik, 2014/2015

import multiprocessing as multi
import numpy as np
import mpmath as mp
import matplotlib.pyplot as pl
import scipy
import sys
from numpy import arange, array, exp, linspace, logspace, ones
from scipy import special as sp
from scipy.integrate import simps, trapz, quad
from scipy.interpolate import interp1d, UnivariateSpline
from itertools import count
from time import time
from astropy.cosmology import LambdaCDM

if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
    xrange = range

from hmf import MassFunction
from hmf import fitting_functions as ff
from hmf import transfer_models as tf

import baryons as baryons
import longdouble_utils as ld
from tools import (
                    Integrate, Integrate1, extrap1d, extrap2d, fill_nan, gas_concentration,
                    star_concentration, virial_mass, virial_radius)
from lens import (
                   power_to_corr, power_to_corr_multi, sigma, d_sigma, power_to_corr_ogata,
                   wp, wp_beta_correction)
from dark_matter import (
                          NFW, NFW_Dc, NFW_f, Con, DM_mm_spectrum, GM_cen_spectrum, GM_sat_spectrum,
                          delta_NFW, MM_analy, GM_cen_analy, GM_sat_analy, GG_cen_analy,
                          GG_sat_analy, GG_cen_sat_analy, miscenter, Bias, Bias_Tinker10)
from cmf import *

import pylab


"""
#-------- Declaring functions ----------
"""


"""
# --------------- Actual halo functions and stuff ------------------
"""


"""
# Mass function from HMFcalc.
"""

def memoize(function):
    memo = {}
    def wrapper(*args, **kwargs):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args, **kwargs)
            memo[args] = rv
        return rv
    return wrapper
"""
#@memoize
def Mass_Function(M_min, M_max, step, name, **cosmology_params):
    return MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=step,
                        mf_fit=name, delta_h=200.0, delta_wrt='mean',
                        cut_fit=False, z2=None, nz=None, delta_c=1.686,
                        **cosmology_params)
"""

"""
# Components of density profile from Mohammed and Seljak 2014
"""

def T_n(n, rho_mean, z, M, R, h_mass, profile, f, omegab, omegac, slope,
        r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, Ac2s):

    np.seterr(divide='ignore', over='ignore', under='ignore',
              invalid='ignore')

    """
    Takes some global variables! Be carefull if you remove or split some
    stuff to different container!
    """
    n = np.float64(n)

    if len(M.shape) == 0:
        T = np.ones(1)
        M = array([M])
        R = array([R])
    else:
        T = np.ones(M.size, dtype=np.longdouble)

    if profile == 'dm':
        for i in xrange(M.size):
            i_range = np.linspace(0,R[i],100)
            Ti = (mp.mpf(4.0 * np.pi) / \
                  (M[i] * (mp.factorial(2.0*n + 1.0)))) * \
                 mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * \
                        (NFW(rho_mean, Con(z, M[i], f), i_range)), i_range))
            T[i] = ld.string2longdouble(str(Ti))
    elif profile == 'gas':
        for i in xrange(M.size):
            i_range = np.linspace(0,R[i],100)
            Ti = (mp.mpf(4.0 * np.pi)/(M[i] * \
                  (mp.factorial(2.0*n + 1.0)))) * \
                 mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * \
                        (baryons.u_g(np.float64(i_range), slope, r_char[i],
                                     omegab, omegac, M[i])), i_range))
            T[i] = ld.string2longdouble(str(Ti))
    elif profile == 'stars':
        for i in xrange(M.size):
            i_range = np.linspace(0,R[i],100)
            Ti = (mp.mpf(4.0 * np.pi)/(M[i] * \
                  (mp.factorial(2.0*n + 1.0)))) * \
                 mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * \
                        (baryons.u_s(np.float64(i_range), slope, r_char[i],
                                     h_mass, M[i], sigma, alpha, A, M_1,
                                     gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)),
                                   i_range))
            T[i] = ld.string2longdouble(str(Ti))
    return T


"""
# Integrals of mass functions with density profiles and population functions.
"""

def f_k(k_x):
    F = sp.erf(k_x/0.1) #0.05!
    return F

def multi_proc_T(a,b, que, n, rho_mean, z, m_x, r_x, h_mass, profile, f,
                 omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1,
                 gamma_2, b_0, b_1, b_2, Ac2s):
    outdict = {}
    r = arange(a, b, 1)
    T = np.ones(r.size, len(m_x), dtype=np.longdouble)
    for i in xrange(r.size):
        T[i,:] = T_n(r[i], rho_mean, z, m_x, r_x, h_mass, profile, f, omegab,
                        omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1,
                        gamma_2, b_0, b_1, b_2, Ac2s)
    # Write in dictionary, so the result can be read outside of function.
    outdict = np.column_stack((r, T))
    que.put(outdict)
    return

def T_table_multi(n, rho_mean, z, m_x, r_x, h_mass, profile, f,
                  omegab, omegac, slope, r_char, sigma, alpha, A, M_1,
                  gamma_1, gamma_2, b_0, b_1, b_2):
    n = (n+4)/2
    # Match the number of cores!
    nprocs = multi.cpu_count()
    q1 = multi.Queue()
    procs = []
    chunk = int(np.ceil(n/float(nprocs)))

    #widgets = ['Calculating T: ', Percentage(), ' ',
                #Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=nprocs).start()
    for j in xrange(nprocs):
        work = multi.Process(target=multi_proc_T, args=((j*chunk),
                                ((j+1)*chunk), q1, n, rho_mean, z, m_x, r_x,
                                h_mass, profile, f, omegab, omegac, slope,
                                r_char, sigma, alpha, A, M_1,
                                gamma_1, gamma_2, b_0, b_1, b_2))
        procs.append(work)
        work.start()
    result = array([]).reshape(0, len(m_x)+1)
    for j in xrange(nprocs):
        result = np.vstack([result, array(q1.get())])
    #pbar.finish()
    result = result[np.argsort(result[:, 0])]
    return np.delete(result, 0, 1)


def T_table(n, rho_mean, z, m_x, r_x, h_mass, profile, f, omegab, omegac,
            slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2,
            b_0, b_1, b_2, Ac2s):

    """
    Calculates all the T integrals and saves them into a array, so that the
    calling of them is fast for all other purposes.

    """
    n = n+2
    T = np.ones((n/2,m_x.size))
    #widgets = ['Calculating T: ', Percentage(), ' ',
                #Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=n/2).start()
    for i in xrange(n/2):
        T[i] = T_n(i, rho_mean, z, m_x, r_x, h_mass, profile, f, omegab,
                   omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1,
                   gamma_2, b_0, b_1, b_2, Ac2s)
        #pbar.update(i+1)
    #pbar.finish()
    return T


def n_gal(mass_func, population, m_x):
    """
    Calculates average number of galaxies!

    """
    #print mass_func.dndm.shape, population.shape, m_x.shape
    return trapz(mass_func.dndm * population, m_x)


def eff_mass(z, mass_func, population, m_x):
    #integ1 = mass_func.dndlnm*population
    #integ2 = mass_func.dndm*population
    #mass = Integrate(integ1, m_x)/Integrate(integ2, m_x)
    #return mass
    return trapz(mass_func.dndlnm * population, m_x) / \
           trapz(mass_func.dndm * population, m_x)



"""
# Two halo term for matter-galaxy specta! For matter-matter it is only P_lin!
"""

def TwoHalo(mass_func, norm, population, k_x, r_x, m_x):
    """
    This is ok!

    """
    #P2 = (exp(mass_func.power)/norm) * \
         #(Integrate((mass_func.dndlnm * population * \
                     #Bias_Tinker10(mass_func,r_x)/m_x), m_x))
    ##print ("Two halo term calculated.")
    #return P2
    
    b_g = trapz(mass_func.dndlnm * population * \
                Bias_Tinker10(mass_func, r_x) / m_x, m_x) / norm
                
    return (mass_func.power * b_g), b_g


def model(theta, R, h=0.7, Om=0.315, Ol=0.685,
          expansion=100, expansion_stars=160, n_bins=10000,
          lnk_min=-13., lnk_max=17.):
    np.seterr(divide='ignore', over='ignore', under='ignore',
              invalid='ignore')
    # making them local doesn't seem to make any difference
    # but keeping for now
    _array = array
    _izip = izip
    _logspace = logspace
    _linspace = linspace

    # Setting parameters from config file
    # this is the way theta will look in v1.2.0
    #z, f, sigma_c, A, M_1, gamma_1, gamma_2, \
        #fc_nsat, alpha_s, b_0, b_1, b_2, Ac2s, \
        #alpha_star, beta_gas, r_t0, r_c0, \
        #M_min, M_max, M_step, M_bin_min, M_bin_max, \
        #centrals, satellites, taylor_procedure, include_baryons, \
        #smth1, smth2 = theta
    sigma_8, H0, omegam, omegab, omegav, n, \
        z, f, sigma_c, A, M_1, gamma_1, gamma_2, \
        fc_nsat, alpha_s, b_0, b_1, b_2, \
        alpha_star, beta_gas, r_t0, r_c0, p_off, r_off, bias, \
        M_bin_min, M_bin_max, \
        Mstar, \
        centrals, satellites, miscentering, taylor_procedure, include_baryons, \
        simple_hod, smth1, smth2 = theta
    # hard-coded in the current version
    Ac2s = 0.56
    M_min = 5.
    M_max = 16.
    M_step = 200
    #centrals = 1
    #satellites = 1

    #to = time()
    # HMF set up parameters
    k_step = (lnk_max-lnk_min) / n_bins
    k_range = arange(lnk_min, lnk_max, k_step)
    k_range_lin = exp(k_range)
    #mass_range = _logspace(M_min, M_max, int((M_max-M_min)/M_step))
    mass_range = 10**_linspace(M_min, M_max, M_step)
    M_step = (M_max - M_min)/M_step

    if not np.iterable(M_bin_min):
        M_bin_min = np.array([M_bin_min])
        M_bin_max = np.array([M_bin_max])
    if not np.iterable(z):
        z = np.array([z]*M_bin_min.size)
    if not np.iterable(f):
        f = np.array([f]*M_bin_min.size)
    if not np.iterable(fc_nsat):
        fc_nsat = np.array([fc_nsat]*M_bin_min.size)
    if not np.iterable(Mstar):
        Mstar = np.array([Mstar]*M_bin_min.size)


    concentration = np.array([Con(np.float64(z_i), mass_range, np.float64(f_i))\
                              for z_i, f_i in _izip(z,f)])
    
    n_bins_obs = M_bin_min.size
    bias = np.array([bias]*k_range_lin.size).T
    
    #hod_mass = np.array([np.logspace(Mi, Mx, 200, endpoint=False, dtype=np.longdouble)
    #                   for Mi, Mx in _izip(M_bin_min, M_bin_max)])

    hod_mass = 10.0**np.array([np.linspace(Mi, Mx, 200, dtype=np.longdouble)
                           for Mi, Mx in _izip(M_bin_min, M_bin_max)])
    
    transfer_params = _array([])
    for z_i in z:
        transfer_params = np.append(transfer_params, {'sigma_8': sigma_8,
                                    'n': n,
                                    'lnk_min': lnk_min ,'lnk_max': lnk_max,
                                    'dlnk': k_step,
                                    'z':np.float64(z_i)})
    
    # Calculation
    # Tinker10 should also be read from theta!
    #to = time()
    hmf = _array([])
    h = H0/100.0
    cosmo_model = LambdaCDM(H0=H0, Ob0=omegab, Om0=omegam, Ode0=omegav, Tcmb0=2.725)
    for i in transfer_params:
        hmf_temp = MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=M_step,
                                hmf_model=ff.Tinker10, delta_h=200.0, delta_wrt='mean',
                                delta_c=1.686,
                                **i)
        hmf_temp.update(cosmo_model=cosmo_model)
        hmf = np.append(hmf, hmf_temp)

    mass_func = np.zeros((z.size, mass_range.size))
    rho_mean = np.zeros(z.shape)
    rho_crit = np.zeros(z.shape)
    rho_dm = np.zeros(z.shape)
    
    omegab = hmf[0].cosmo.Ob0
    omegac = hmf[0].cosmo.Om0-omegab
    omegav = hmf[0].cosmo.Ode0
    
    for i in xrange(z.size):
        mass_func[i] = hmf[i].dndlnm
        rho_mean[i] = hmf[i].mean_density0
        rho_crit[i] = rho_mean[i] / (omegac+omegab)
        rho_dm[i] = rho_mean[i] * baryons.f_dm(omegab, omegac)

    if include_baryons:
        # these might as well be scalars
        r_t0 = r_t0 * ones(mass_range.size)
        r_c0 = r_c0 * ones(mass_range.size)
        #to = time()
        #rho_dm = baryons.rhoDM(hmf, mass_range, omegab, omegac)
        rho_stars = _array([baryons.rhoSTARS(hmf_i, i, mass_range,
                                             sigma_c, alpha_s, A, M_1,
                                             gamma_1, gamma_2, b_0, b_1, b_2,
                                             Ac2s)
                            for hmf_i, i in _izip(hmf, hod_mass)])
        rho_gas, F = np.transpose([baryons.rhoGAS(hmf_i, rho_crit, omegab,
                                                  omegac, i,
                                                  mass_range, sigma_c,
                                                  alpha_s, A, M_1, gamma_1,
                                                  gamma_2, b_0, b_1, b_2,
                                                  Ac2s)
                                   for hmf_i, i in _izip(hmf, hod_mass)])
        #print 'rho_stars =', time() - to

    #to = time()
    rvir_range_lin = _array([virial_radius(mass_range, rho_mean_i, 200.0)
                             for rho_mean_i in rho_mean])
    rvir_range = np.log10(rvir_range_lin)
    rvir_range_3d = _logspace(-3.2, 4, 200, endpoint=True)
    rvir_range_3d_i = _logspace(-2.5, 1.2, 25, endpoint=True)
    rvir_range_2d_i = R[0][1:]
    #print 'rvir =', time() - to

    # Calculating halo model

    #to = time()
    if centrals:
        if simple_hod:
            if not np.iterable(M_1):
                M_1 = _array([M_1]*M_bin_min.size)
            if not np.iterable(sigma_c):
                sigma_c = _array([sigma_c]*M_bin_min.size)
            pop_c = _array([ncm_simple(hmf_i, mass_range, M_1_i, sigma_c_i)
                            for hmf_i, M_1_i, sigma_c_i in
                            _izip(hmf, M_1, sigma_c)])
        else:
            pop_c = _array([ncm(hmf_i, i, mass_range, sigma_c, alpha_s, A, M_1,
                            gamma_1, gamma_2, b_0, b_1, b_2)
                        for hmf_i, i in _izip(hmf, hod_mass)])
    else:
        pop_c = np.zeros(hod_mass.shape)
    #print 'centrals =', time() - to
    #to = time()
    if satellites:
        if simple_hod:
            """
            if not np.iterable(M_1):
                M_1 = _array([M_1]*M_bin_min.size)
            if not np.iterable(sigma_c):
                sigma_c = _array([sigma_c]*M_bin_min.size)
            pop_s = _array([nsm_simple(hmf, mass_range, M_1_i,
                                       sigma_c_i, alpha_s_i)
                            for M_1_i, sigma_c_i, alpha_s_i in
                            _izip(_array([M_1]), _array([sigma_c]), \
                            _array([alpha_s]))]) * \
                    _array([ncm_simple(hmf, mass_range, M_1_i, sigma_c_i)
                            for M_1_i, sigma_c_i in
                            _izip(_array([M_1]), _array([sigma_c]))])
            """
            pop_s = np.zeros(hod_mass.shape)
        else:
            pop_s = _array([nsm(hmf_i, i, mass_range, sigma_c, alpha_s, A, M_1,
                            gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)
                        for hmf_i, i in _izip(hmf, hod_mass)])
    else:
        pop_s = np.zeros(hod_mass.shape)
    #print 'satellites =', time() - to
    pop_g = pop_c + pop_s

    #to = time()
    ngal = _array([n_gal(hmf_i, pop_g_i , mass_range)
                   for hmf_i, pop_g_i in _izip(hmf, pop_g)])
    #print 'ngal =', time() - to

    #to = time()
    effective_mass = _array([eff_mass(np.float64(z_i), hmf_i, pop_g_i, mass_range)
                             for z_i, hmf_i, pop_g_i in _izip(z, hmf, pop_g)])
    # Why does this only have pop_c and not pop_g?
    # Do we need it? It's not used anywhere in the code
    effective_mass_bar = _array([eff_mass(np.float64(z_i), hmf_i, pop_g_i, mass_range) * \
                                    (1.0 - baryons.f_dm(omegab, omegac))
                                 for z_i, hmf_i, pop_g_i in _izip(z, hmf, pop_g)])
    #effective_mass_bar = _array([effective_mass2 * \
                                   #(baryons.f_stars(i[0], effective_mass2,
                                                    #sigma_c, alpha_s, A, M_1,
                                                    #gamma_1, gamma_2, b_0,
                                                    #b_1, b_2))
                                   #for i in _izip(hod_mass)])
    #print 'eff mass =', time() - to


    """
    # Power spectrum
    """
    #to = time()
    if taylor_procedure:
        T_dm = _array([T_table(expansion, rho_dm_i, np.float64(z_i), mass_range,
                               rvir_range_lin_i, i, 'dm', np.float64(f_i), omegab,
                               omegac, 0, 0, sigma_c, alpha_s, A, M_1,
                               gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)
                       for rvir_range_lin_i, rho_dm_i, z_i, i, f_i in\
                       _izip(rvir_range_lin, rho_dm, z, hod_mass, f)])
        T_tot = _array([T_dm[i][0:1:1,:] for i in xrange(M_bin_min.size)])
    else:
        T_dm = np.ones((hod_mass.size, np.int((expansion+2)/2), mass_range.size))
        T_tot = _array([T_dm[i][0:1:1,:] for i in xrange(M_bin_min.size)])
    #print 'T_tot =', time() - to

    if include_baryons:
        #to = time()
        T_dm = _array([T_table(expansion, rho_dm_i, np.float64(z_i), mass_range,
                               rvir_range_lin_i, i, 'dm', np.float64(f_i), omegab,
                               omegac, 0, 0, sigma_c, alpha_s, A, M_1,
                               gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)
                       for rvir_range_lin_i, rho_dm_i, z_i, i, f_i in \
                       _izip(rvir_range_lin, rho_dm, z, hod_mass, f)])
        T_stars = _array([T_table(expansion_stars, rho_mean_i, np.float64(z_i),
                                  mass_range, rvir_range_lin_i, i,
                                  'stars', np.float64(f_i), omegab, omegac, alpha_star,
                                  r_t0, sigma_c, alpha_s, A, M_1, gamma_1,
                                  gamma_2, b_0, b_1, b_2, Ac2s)
                          for rvir_range_lin_i, rho_mean_i, z_i, i, f_i in \
                          _izip(rvir_range_lin, rho_mean, z, hod_mass, f)])
        T_gas = _array([T_table(expansion, rho_mean_i, np.float64(z_i), mass_range,
                                rvir_range_lin_i, i, 'gas', np.float64(f_i), omegab,
                                omegac, beta_gas, r_c0, sigma_c, alpha_s,
                                A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)
                        for rvir_range_lin_i, rho_mean_i, z_i, i, f_i in \
                        _izip(rvir_range_lin, rho_mean, z, hod_mass, f)])
        T_tot = _array([T_dm[i][0:1:1,:] + T_stars[i][0:1:1,:] + \
                        T_gas[i][0:1:1,:]
                        for i in xrange(M_bin_min.size)])
        #print 'baryons =', time() - to

    # damping of the 1h power spectra at small k
    F_k1 = f_k(k_range_lin)
    # Fourier Transform of the NFW profile
    u_k = _array([NFW_f(np.float64(z_i), rho_dm_i, np.float64(f_i), mass_range,\
                        rvir_range_lin_i, k_range_lin,\
                        c=concentration_i) for rvir_range_lin_i, rho_dm_i, z_i,\
                  f_i, concentration_i in _izip(rvir_range_lin, \
                                                rho_dm, z, f, concentration)])

    # and of the NFW profile of the satellites
    #print fc_nsat
    uk_s = _array([NFW_f(np.float64(z_i), rho_dm_i, np.float64(fc_nsat_i), \
                         mass_range, rvir_range_lin_i, k_range_lin)
                   for rvir_range_lin_i, rho_dm_i, z_i, fc_nsat_i in \
                   _izip(rvir_range_lin, rho_dm, z, fc_nsat)])
    uk_s = uk_s/uk_s[:,0][:,None]
    #uk_s = u_k
    
    # If there is miscentering to be accounted for
    if miscentering:
        if not np.iterable(p_off):
            p_off = _array([p_off]*M_bin_min.size)
        if not np.iterable(r_off):
            r_off = _array([r_off]*M_bin_min.size)
        u_k = _array([NFW_f(np.float64(z_i), rho_dm_i, np.float64(f_i), \
                            mass_range, rvir_range_lin_i, k_range_lin,
                    c=concentration_i) * miscenter(p_off_i, r_off_i, mass_range,
                                                 rvir_range_lin_i, k_range_lin,
                                                 c=concentration_i) for \
                      rvir_range_lin_i, rho_dm_i, z_i, f_i, concentration_i, p_off_i, r_off_i
                      in _izip(rvir_range_lin, rho_dm, z, f, concentration, p_off, r_off)])
    u_k = u_k/u_k[:,0][:,None]

    # Galaxy - dark matter spectra
    #to = time()
    Pg_2h = bias * _array([TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin,
                            rvir_range_lin_i, mass_range)[0]
                    for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i in \
                           _izip(rvir_range_lin, hmf, ngal, pop_g)])

    bias_out = bias.T[0] * _array([TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin,
                                  rvir_range_lin_i, mass_range)[1]
                          for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i in \
                                   _izip(rvir_range_lin, hmf, ngal, pop_g)])
    #print 'TwoHalo =', time() - to
    if taylor_procedure or include_baryons:
        #print 'spectrum'
        if centrals:
            #to = time()
            Pg_c = F_k1 * _array([GM_cen_spectrum(hmf_i, np.float64(z_i), \
                                                  rho_dm_i, rho_mean_i,
                                                  expansion, pop_c_i,
                                                  ngal_i, k_range_lin,
                                                  rvir_range_lin_i,
                                                  mass_range,
                                                  T_dm_i, T_tot_i)
                                  for rvir_range_lin_i, rho_dm_i, rho_mean_i, \
                                  hmf_i, z_i, pop_c_i, ngal_i, T_dm_i, T_tot_i
                                  in _izip(rvir_range_lin, rho_dm, rho_mean, \
                                           hmf, z, pop_c, ngal, T_dm, T_tot)])
            #print 'Pg_c =', time() - to
        else:
            Pg_c = np.zeros((n_bins_obs,n_bins))
        if satellites:
            #to = time()
            Pg_s = F_k1 * _array([GM_sat_spectrum(hmf_i, np.float64(z_i), \
                                                  rho_dm_i, rho_mean_i,
                                                  expansion, pop_s_i,
                                                  ngal_i, k_range_lin,
                                                  rvir_range_lin_i,
                                                  mass_range,
                                                  T_dm_i, T_tot_i)
                                  for rvir_range_lin_i, rho_dm_i, rho_mean_i, \
                                  hmf_i, z_i, pop_s_i, ngal_i, T_dm_i, T_tot_i
                                  in _izip(rvir_range_lin, rho_dm, rho_mean, \
                                           hmf, z, pop_s, ngal, T_dm, T_tot)])
            #print 'Pg_s =', time() - to
        else:
            Pg_s = np.zeros((n_bins_obs,n_bins))
    else:
        #print 'analytic'
        if centrals:
            #to = time()
            Pg_c = F_k1 * _array([GM_cen_analy(hmf_i, u_k_i, rho_dm_i, pop_c_i,
                                               ngal_i, mass_range)
                                  for rho_dm_i, hmf_i, pop_c_i, ngal_i, u_k_i in\
                                  _izip(rho_dm, hmf, pop_c, ngal, u_k)])
            #print 'Pg_c =', time() - to
        else:
            Pg_c = np.zeros((n_bins_obs,n_bins))
        if satellites:
            #to = time()
            Pg_s = F_k1 * _array([GM_sat_analy(hmf_i, u_k_i, uk_s_i, rho_dm_i,
                                               pop_s_i, ngal_i, mass_range)
                                  for rho_dm_i, hmf_i, pop_s_i, ngal_i, u_k_i, uk_s_i in\
                                  _izip(rho_dm, hmf, pop_s, ngal, u_k, uk_s)])
            #print 'Pg_s =', time() - to
        else:
            Pg_s = np.zeros((n_bins_obs,n_bins))

    # Galaxy - stars/gas spectra
    if include_baryons:
        if centrals:
            #to = time()
            Ps_c = F_k1 * _array([baryons.GS_cen_spectrum(
                                        hmf_i, np.float64(z_i), rho_stars_i,\
                                                          rho_mean_i,
                                        expansion_stars, pop_c_i, ngal_i,
                                        k_range_lin, rvir_range_lin_i,
                                        mass_range, T_stars_i, T_tot_i)
                                    for rvir_range_lin_i, rho_mean_i, hmf_i, \
                                  z_i, rho_stars_i, pop_c_i, ngal_i, \
                                        T_stars_i, T_tot_i
                                    in _izip(rvir_range_lin, rho_mean, hmf, z,\
                                             rho_stars, pop_c, ngal, T_stars,
                                            T_tot)])
            #print 'Ps_c =', time() - to
            #to = time()
            Pgas_c = F_k1 * _array([baryons.GGas_cen_spectrum(
                                        hmf_i, np.float64(z_i), F_i, rho_gas_i,\
                                                              rho_mean_i,
                                        expansion, pop_c_i, ngal_i,
                                        k_range_lin, rvir_range_lin_i,
                                        mass_range, T_gas_i, T_tot_i)
                                    for rvir_range_lin_i, rho_mean_i, hmf_i,\
                                    z_i, F_i, rho_gas_i, pop_c_i, ngal_i, \
                                        T_gas_i, T_tot_i
                                    in _izip(rvir_range_lin, rho_mean, hmf, z,\
                                             F, rho_gas, pop_c, ngal, T_gas,
                                            T_tot)])
            #print 'Pgas_c =', time() - to
        else:
            # In this case Pg_c is an array of zeros with the right shape
            Ps_c = Pg_c
            Pgas_c = Pg_c
        if satellites:
            #to = time()
            Ps_s = F_k1 * _array([baryons.GS_sat_spectrum(
                                        hmf_i, np.float64(z_i), rho_stars_i,\
                                                          rho_mean_i,
                                        expansion, pop_s_i, ngal_i,
                                        k_range_lin, rvir_range_lin_i,
                                        mass_range, T_dm_i, T_stars_i,
                                        T_tot_i)
                                    for rvir_range_lin_i, rho_mean_i, hmf_i,\
                                  z_i, rho_stars_i, pop_s_i, ngal_i, \
                                        T_dm_i, T_stars_i, T_tot_i
                                    in _izip(rvir_range_lin, rho_mean, hmf,\
                                             z, rho_stars, pop_s, ngal, T_dm,
                                            T_stars, T_tot)])
            #print 'Ps_s =', time() - to
            #to = time()
            Pgas_s = F_k1 * _array([baryons.GGas_sat_spectrum(
                                        hmf_i, np.float64(z_i), F_i, rho_gas_i,rho_mean_i,
                                        expansion, pop_s_i, ngal_i,
                                        k_range_lin, rvir_range_lin_i,
                                        mass_range, T_dm_i, T_gas_i, T_tot_i)
                                    for rvir_range_lin_i, rho_mean_i, hmf_i,\
                                    z_i, F_i, rho_gas_i, pop_s_i, ngal_i, \
                                        T_dm_i, T_gas_i, T_tot_i
                                    in _izip(rvir_range_lin, rho_mean, hmf,\
                                             z, F, rho_gas, pop_s, ngal, T_dm,
                                            T_gas, T_tot)])
            #print 'Pgas_s =', time() - to
        else:
            # In this case Pg_s is an array of zeros with the right shape
            Ps_s = Pg_s
            Pgas_s = Pg_s

    # Combined (all) by type
    #Pg_k_s = _array([(rho_stars_i * (Ps_c_i + Ps_s_i)) / (rho_mean)
                       #for rho_stars_i, Ps_c_i, Ps_s_i
                       #in _izip(rho_stars, Ps_c, Ps_s)])
    #Pg_k_g = _array([(rho_gas_i * (Pgas_c_i + Pgas_s_i)) / (rho_mean)
                       #for rho_gas_i, Pgas_c_i, Pgas_s_i
                       #in _izip(rho_gas, Pgas_c, Pgas_s)])

    #to = time()
    if include_baryons:
        # all components
        Pg_k = _array([(rho_dm_i * (Pg_c_i + Pg_s_i) + \
                           Pg_2h_i * rho_mean_i + \
                           rho_stars_i * (Ps_c_i + Ps_s_i) + \
                           rho_gas_i * (Pgas_c_i + Pgas_s_i))  / rho_mean_i
                       for Pg_c_i, Pg_s_i, Pg_2h_i, rho_stars_i, Ps_c_i, \
                           Ps_s_i, rho_gas_i, Pgas_c_i, Pgas_s_i, rho_mean_i, rho_dm_i
                       in _izip(Pg_c, Pg_s, Pg_2h, rho_stars, Ps_c, Ps_s,
                                rho_gas, Pgas_c, Pgas_s, rho_mean, rho_dm)])
    else:
        Pg_k = _array([(rho_dm_i/rho_mean_i) * (Pg_c_i + Pg_s_i) + Pg_2h_i
                       for Pg_c_i, Pg_s_i, Pg_2h_i, rho_dm_i, rho_mean_i
                       in _izip(Pg_c, Pg_s, Pg_2h, rho_dm, rho_mean)])
    #print 'Pg_k =', time() - to

    # Normalized sattelites and centrals for sigma and d_sigma

    #Pg_c2 = (rho_dm/rho_mean)*_array([Pg_c_i for Pg_c_i in _izip(Pg_c)])
    #Pg_s2 = (rho_dm/rho_mean)*_array([Pg_s_i for Pg_s_i in _izip(Pg_s)])

    #Ps_c2 = _array([(rho_stars_i/rho_mean) * Ps_c_i
                      #for rho_stars_i, Ps_c_i in _izip(rho_stars, Ps_c)])
    #Ps_s2 = _array([(rho_stars_i/rho_mean) * Ps_s_i
                      #for rho_stars_i, Ps_s_i in_izip(rho_stars, Ps_s)])

    #Pgas_c2 =  _array([(rho_gas_i/rho_mean) * Pgas_c_i
                         #for rho_gas_i, Pgas_c_i in _izip(rho_gas, Pgas_c)])
    #Pgas_s2 =  _array([(rho_gas_i/rho_mean) * Pgas_s_i
                         #for rho_gas_i, Pgas_s_i in _izip(rho_gas, Pgas_s)])

    #to = time()
    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
               for Pg_k_i in _izip(Pg_k)]
    #print 'P_inter =', time() - to

    # For plotting parts - Andrej!
    """
    P_inter_c = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
           for Pg_k_i in _izip((rho_dm/rho_mean) *Pg_c)]

    P_inter_2 = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
           for Pg_k_i in _izip(Pg_2h)]
    """
           

    """
    # Correlation functions
    """

    #to = time()
    xi2 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2[i] = power_to_corr_ogata(P_inter[i], rvir_range_3d)
        #xi2[xi2 <= 0.0] = np.nan
        #xi2[i,:] = fill_nan(xi2[i,:])
    #print 'xi2 =', time() - to
    #for i in xrange(len(xi2)):
        #pylab.plot(rvir_range_3d, xi2[i], '-', label=i)
    ##pylab.legend()
    #pylab.xscale('log')
    #pylab.yscale('log')
    #pylab.show()

    # For plotting parts - Andrej!
    """
    xi3 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi3[i] = power_to_corr_ogata(P_inter_c[i], rvir_range_3d)

    xi4 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi4[i] = power_to_corr_ogata(P_inter_2[i], rvir_range_3d)
    """


    """
    # Projected surface density
    """

    #to = time()
    sur_den2 = _array([sigma(xi2_i, rho_mean_i, rvir_range_3d, rvir_range_3d_i)
                       for xi2_i, rho_mean_i in _izip(xi2, rho_mean)])
    for i in xrange(M_bin_min.size):
        #sur_den2[i][sur_den2[i] <= 0.0] = np.nan
        #sur_den2[i][sur_den2[i] >= 1e20] = np.nan
        sur_den2[i][(sur_den2[i] <= 0.0) | (sur_den2[i] >= 1e20)] = np.nan
        sur_den2[i] = fill_nan(sur_den2[i])
    #print 'sur_den2 =', time() - to

    # For plotting parts - Andrej!
    """
    sur_den3 = _array([sigma(xi2_i, rho_mean_i, rvir_range_3d, rvir_range_3d_i)
                   for xi2_i, rho_mean_i in _izip(xi3, rho_mean)])
    sur_den4 = _array([sigma(xi2_i, rho_mean_i, rvir_range_3d, rvir_range_3d_i)
                   for xi2_i, rho_mean_i in _izip(xi4, rho_mean)])

    for i in xrange(M_bin_min.size):
        sur_den3[i][(sur_den3[i] <= 0.0) | (sur_den3[i] >= 1e20)] = np.nan
        sur_den3[i] = fill_nan(sur_den3[i])
        sur_den4[i][(sur_den4[i] <= 0.0) | (sur_den4[i] >= 1e20)] = np.nan
        sur_den4[i] = fill_nan(sur_den4[i])
    """

    """
    # Excess surface density
    """
    #mass_range_dm = mass_range*baryons.f_dm(omegab, omegac)
    #from dark_matter import av_delta_NFW
    #NFW_d_sigma_av = (rho_dm/rho_mean) * \
                      #_array([av_delta_NFW(hmf.dndm, z, rho_mean, pop_c_i,
                                             #mass_range,
                                             #rvir_range_2d_i) / 1e12
                                #for pop_c_i in _izip(pop_c)])

    #to = time()
    d_sur_den2 = _array([np.nan_to_num(d_sigma(sur_den2_i,
                                                 rvir_range_3d_i,
                                                 rvir_range_2d_i))
                           for sur_den2_i in _izip(sur_den2)]) / 1e12

    # For plotting parts - Andrej!
    """
    d_sur_den3 = _array([np.nan_to_num(d_sigma(sur_den2_i,
                                           rvir_range_3d_i,
                                           rvir_range_2d_i))
                     for sur_den2_i in _izip(sur_den3)]) / 1e12

    d_sur_den4 = _array([np.nan_to_num(d_sigma(sur_den2_i,
                                           rvir_range_3d_i,
                                           rvir_range_2d_i))
                     for sur_den2_i in _izip(sur_den4)]) / 1e12
                     
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    nu = spline(hmf[0].nu,hmf[0].M,k=5)
                     
    """
    
    #for i in xrange(len(M_bin_min)):
        #d_sur_den2[i][d_sur_den2[i] <= 0.0] = np.nan
        #d_sur_den2[i][d_sur_den2[i] >= 10.0**20.0] = np.nan
        #d_sur_den2[i] = fill_nan(d_sur_den2[i])
    #print 'd_sur_den2 =', time() - to

    #to = time()
    out_esd_tot = _array([UnivariateSpline(rvir_range_2d_i,
                                           np.nan_to_num(d_sur_den2_i), s=0)
                          for d_sur_den2_i in _izip(d_sur_den2)])
    #print 'out_esd_tot =', time() - to
    #to = time()
    out_esd_tot_inter = np.zeros((M_bin_min.size, rvir_range_2d_i.size))
    for i in xrange(M_bin_min.size):
        out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)

    if include_baryons:
        pointmass = _array([Mi/(np.pi*rvir_range_2d_i**2.0)/1e12 \
            for Mi in izip(effective_mass_bar)])
    else:
        pointmass = _array([(10.0**Mi[0])/(np.pi*rvir_range_2d_i**2.0)/1e12 \
            for Mi in izip(Mstar)])

    out_esd_tot_inter = out_esd_tot_inter + pointmass
    

    # Add other outputs as needed. Total ESD should always be first!
    return [out_esd_tot_inter, np.log10(effective_mass)]
    #return out_esd_tot_inter, d_sur_den3, d_sur_den4, pointmass, nu(1)

if __name__ == '__main__':
    print(0)
