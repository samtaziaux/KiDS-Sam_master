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

# Halo model code
# Andrej Dvornik, 2014/2015

import time
import multiprocessing as multi
import numpy as np
import mpmath as mp
import longdouble_utils as ld
import matplotlib.pyplot as pl
import scipy
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d, UnivariateSpline
import scipy.special as sp
from itertools import count, izip
from hmf import MassFunction

import baryons
from tools import Integrate, Integrate1, extrap1d, extrap2d, fill_nan, \
                  gas_concentration, star_concentration, virial_mass, \
                  virial_radius
from lens import power_to_corr, power_to_corr_multi, sigma, d_sigma, \
                 power_to_corr_ogata
from dark_matter import NFW, NFW_Dc, NFW_f, Con, DM_mm_spectrum, \
                        GM_cen_spectrum, GM_sat_spectrum, delta_NFW, \
                        GM_cen_analy, GM_sat_analy
from cmf import *


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

@memoize
def Mass_Function(M_min, M_max, step, name, **cosmology_params):
    return MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=step,
                        mf_fit=name, delta_h=200.0, delta_wrt='mean',
                        cut_fit=False, z2=None, nz=None, delta_c=1.686,
                        **cosmology_params)
    return m


"""
# Components of density profile from Mohammed and Seljak 2014
"""

def T_n(n, rho_mean, z, M, R, h_mass, profile, f, omegab, omegac, slope,
        r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):

    np.seterr(divide='ignore', over='ignore', under='ignore',
              invalid='ignore')

    """
    Takes some global variables! Be carefull if you remove or split some
    stuff to different container!
    """
    n = np.float64(n)

    if len(M.shape) == 0:
        T = np.ones(1)
        M = np.array([M])
        R = np.array([R])
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
                                     gamma_1, gamma_2, b_0, b_1, b_2)),
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
                 gamma_2, b_0, b_1, b_2):
    outdict = {}
    r = np.arange(a, b, 1)
    T = np.ones(r.size, len(m_x), dtype=np.longdouble)
    for i in xrange(r.size):
        T[i,:] = T_n(r[i], rho_mean, z, m_x, r_x, h_mass, profile, f, omegab,
                        omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1,
                        gamma_2, b_0, b_1, b_2)
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
                                r_char, sigma, alpha, A, M_1, gamma_1, gamma_2,
                                b_0, b_1, b_2))
        procs.append(work)
        work.start()
    result = np.array([]).reshape(0, len(m_x)+1)
    for j in xrange(nprocs):
        result = np.vstack([result, np.array(q1.get())])
    #pbar.finish()
    result = result[np.argsort(result[:, 0])]
    return np.delete(result, 0, 1)


def T_table(n, rho_mean, z, m_x, r_x, h_mass, profile, f, omegab, omegac,
            slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2,
            b_0, b_1, b_2):

    """
    Calculates all the T integrals and saves them into a array, so that the
    calling of them is fast for all other purposes.

    """
    n = n+2
    T = np.ones((n/2, len(m_x)))
    #widgets = ['Calculating T: ', Percentage(), ' ',
                #Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=n/2).start()
    for i in xrange(0, n/2, 1):
        T[i,:] = T_n(i, rho_mean, z, m_x, r_x, h_mass, profile, f, omegab,
                     omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1,
                     gamma_2, b_0, b_1, b_2)
        #pbar.update(i+1)
    #pbar.finish()
    return T


def n_gal(z, mass_func, population, m_x, r_x):
    """
    Calculates average number of galaxies!

    """
    integrand = mass_func.dndm*population
    n =  Integrate(integrand, m_x)
    return n


def eff_mass(z, mass_func, population, m_x):
    integ1 = mass_func.dndlnm*population
    integ2 = mass_func.dndm*population
    mass = Integrate(integ1, m_x)/Integrate(integ2, m_x)
    return mass


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
    # print y, A, a, B, b, C, c
    return 1 - A * nu**a / (nu**a + hmf.delta_c**a) + B * nu**b + C * nu**c


"""
# Two halo term for matter-galaxy specta! For matter-matter it is only P_lin!
"""

def TwoHalo(mass_func, norm, population, k_x, r_x, m_x):
    """
    This is ok!

    """
    P2 = (np.exp(mass_func.power)/norm) * \
         (Integrate((mass_func.dndlnm * population * \
                     Bias_Tinker10(mass_func,r_x)/m_x), m_x))
    #print ("Two halo term calculated.")
    return P2

def model(theta, R, h=0.7, Om=0.315, Ol=0.685,
          expansion=100, expansion_stars=160, n_bins=10000,
          M_min=5., M_max=16., M_bins=100, lnk_min=-13., lnk_max=17.):

    np.seterr(divide='ignore', over='ignore', under='ignore',
              invalid='ignore')

    # HMF set up parameters - for now fixed and not setable from config file.
    step = (M_max - M_min) / M_bins
    k_step = (lnk_max-lnk_min)/n_bins
    k_range = np.arange(lnk_min, lnk_max, k_step)
    k_range_lin = np.exp(k_range)
    mass_range = np.logspace(M_min, M_max, M_bins)

    # Setting parameters from config file
    z, f, sigma_c, A, M_1, gamma_1, gamma_2, alpha_s, b_0, b_1, b_2, \
        alpha_star, beta_gas, r_t0, r_c0, M_bin_min, M_bin_max, \
        taylor_procedure, include_baryons, smth1, smth2 = theta

    hod_mass = np.array([np.logspace(Mi, Mx, 100, endpoint=False,
                                     dtype=np.longdouble)
                         for Mi, Mx in izip(M_bin_min, M_bin_max)])
    r_t0 = r_t0*np.ones(100)
    r_c0 = r_c0*np.ones(100)

    cosmology_params = {"sigma_8": 0.80, "H0": 70.0,"omegab_h2": 0.022,
                        "omegam": 0.3, "omegav": 0.7, "n": 0.96,
                        "lnk_min": lnk_min ,"lnk_max": lnk_max,
                        "dlnk": k_step, "transfer_fit": "BBKS", "z": z,
                        "force_flat": True}
    # Calculation
    # Tinker10 should also be read from theta!
    hmf = Mass_Function(M_min, M_max, step, "Tinker10", **cosmology_params)

    omegab = hmf.omegab
    omegac = hmf.omegac
    omegav = hmf.omegav
    h = hmf.h
    mass_func = hmf.dndlnm
    rho_mean = hmf.mean_dens_z
    rho_crit = rho_mean / (omegac+omegab)
    rho_dm = rho_mean * baryons.f_dm(omegab, omegac)

    if include_baryons:
        #rho_dm = baryons.rhoDM(hmf, mass_range, omegab, omegac)
        rho_stars = np.array([baryons.rhoSTARS(hmf, i[0], mass_range,
                                               sigma_c, alpha_s, A, M_1,
                                               gamma_1, gamma_2, b_0, b_1, b_2)
                              for i in izip(hod_mass)])
        #rho_gas = np.array([baryons.rhoGAS(hmf, rho_crit, omegab, omegac,
                                           #i[0], mass_range, sigma_c,
                                           #alpha_s, A, M_1, gamma_1, gamma_2,
                                           #b_0, b_1, b_2)
                            #for i in izip(hod_mass)])[:,0]
        #F = np.array([baryons.rhoGAS(hmf, rho_crit, omegab, omegac, i[0],
                                     #mass_range, sigma_c, alpha_s, A, M_1,
                                     #gamma_1, gamma_2, b_0, b_1, b_2)
                      #for i in izip(hod_mass)])[:,1]
        # this should be the same as the two calls above
        rho_gas, F = np.transpose([baryons.rhoGAS(hmf, rho_crit, omegab,
                                                  omegac, i[0],
                                                  mass_range, sigma_c,
                                                  alpha_s, A, M_1, gamma_1,
                                                  gamma_2, b_0, b_1, b_2)
                                   for i in izip(hod_mass)])

    radius_range_lin = virial_radius(mass_range, rho_mean, 200.0)
    radius_range = np.log10(radius_range_lin)
    radius_range_3d = np.logspace(-3.2, 4, 200, endpoint=True)
    radius_range_3d_i = np.logspace(-2.5, 1.2, 25, endpoint=True)
    radius_range_2d_i = R[0][1:]

    # Calculating halo model

    pop_c = np.array([ncm(hmf, i[0], mass_range, sigma_c, alpha_s, A, M_1,
                          gamma_1, gamma_2, b_0, b_1, b_2)
                      for i in izip(hod_mass)])
    pop_s = np.array([nsm(hmf, i[0], mass_range, sigma_c, alpha_s, A, M_1,
                          gamma_1, gamma_2, b_0, b_1, b_2)
                      for i in izip(hod_mass)])
    #pop_g = np.array([ngm(hmf, i[0], mass_range, sigma_c, alpha_s, A, M_1,
                          #gamma_1, gamma_2, b_0, b_1, b_2)
                      #for i in izip(hod_mass)])
    pop_g = pop_c + pop_s

    ngal = np.array([n_gal(z, hmf, pop_g_i , mass_range, radius_range_lin)
                     for pop_g_i in izip(pop_g)])

    effective_mass = np.array([eff_mass(z, hmf, pop_g_i, mass_range)
                               for pop_g_i in izip(pop_g)])
    effective_mass_bar = np.array([eff_mass(z, hmf, pop_c_i,
                                            mass_range) * \
                                   (1.0 - baryons.f_dm(omegab, omegac))
                                   for pop_c_i in izip(pop_c)])
    #effective_mass_bar = np.array([effective_mass2 * \
                                   #(baryons.f_stars(i[0], effective_mass2,
                                                    #sigma_c, alpha_s, A, M_1,
                                                    #gamma_1, gamma_2, b_0,
                                                    #b_1, b_2))
                                   #for i in izip(hod_mass)])


    """
    # Power spectrum
    """
    if taylor_procedure:
        T_dm = np.array([T_table(expansion, rho_dm, z, mass_range,
                                 radius_range_lin, i[0], "dm", f, omegab,
                                 omegac, 0, 0, sigma_c, alpha_s, A, M_1,
                                 gamma_1, gamma_2, b_0, b_1, b_2)
                         for i in izip(hod_mass)])
        T_tot = np.array([T_dm[i][0:1:1,:] for i in xrange(M_bin_min.size)])
    else:
        T_dm = np.ones((hod_mass.size, (expansion+2)/2, mass_range.size))
        T_tot = np.array([T_dm[i][0:1:1,:] for i in xrange(M_bin_min.size)])

    if include_baryons:
        T_dm = np.array([T_table(expansion, rho_dm, z, mass_range,
                                 radius_range_lin, i[0], "dm", f, omegab,
                                 omegac, 0, 0, sigma_c, alpha_s, A, M_1,
                                 gamma_1, gamma_2, b_0, b_1, b_2)
                         for i in izip(hod_mass)])
        T_stars = np.array([T_table(expansion_stars, rho_mean, z,
                                    mass_range, radius_range_lin, i[0],
                                    "stars", f, omegab, omegac, alpha_star,
                                    r_t0, sigma_c, alpha_s, A, M_1, gamma_1,
                                    gamma_2, b_0, b_1, b_2)
                            for i in izip(hod_mass)])
        T_gas = np.array([T_table(expansion, rho_mean, z, mass_range,
                                  radius_range_lin, i[0], "gas", f, omegab,
                                  omegac, beta_gas, r_c0, sigma_c, alpha_s,
                                  A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)
                          for i in izip(hod_mass)])
        T_tot = np.array([T_dm[i][0:1:1,:] + T_stars[i][0:1:1,:] + \
                          T_gas[i][0:1:1,:]
                          for i in xrange(M_bin_min.size)])

    # damping of the 1h power spectra at small k
    F_k1 = f_k(k_range_lin)

    # Galaxy - dark matter spectra
    Pg_2h = np.array([TwoHalo(hmf, ngal_i, pop_g_i, k_range_lin,
                              radius_range_lin, mass_range)
                      for ngal_i, pop_g_i in izip(ngal, pop_g)])
    if taylor_procedure or include_baryons:
        Pg_c = F_k1 * np.array([GM_cen_spectrum(hmf, z, rho_dm, rho_mean,
                                                expansion, pop_c_i, ngal_i,
                                                k_range_lin, radius_range_lin,
                                                mass_range,
                                                T_dm_i, T_tot_i)
                                for pop_c_i, ngal_i, T_dm_i, T_tot_i
                                in izip(pop_c, ngal, T_dm, T_tot)])
        Pg_s = F_k1 * np.array([GM_sat_spectrum(hmf, z, rho_dm, rho_mean,
                                                expansion, pop_s_i, ngal_i,
                                                k_range_lin, radius_range_lin,
                                                mass_range,
                                                T_dm_i, T_tot_i)
                                for pop_s_i, ngal_i, T_dm_i, T_tot_i
                                in izip(pop_s, ngal, T_dm, T_tot)])
    else:
        Pg_c = F_k1 * np.array([GM_cen_analy(hmf, z, rho_dm, rho_mean, f,
                                             expansion, pop_c_i, ngal_i,
                                             k_range_lin, radius_range_lin,
                                             mass_range, T_dm_i, T_tot_i)
                                for pop_c_i, ngal_i, T_dm_i, T_tot_i
                                in izip(pop_c, ngal, T_dm, T_tot)])
        Pg_s = F_k1 * np.array([GM_sat_analy(hmf, z, rho_dm, rho_mean, f,
                                             expansion, pop_s_i, ngal_i,
                                             k_range_lin, radius_range_lin,
                                             mass_range, T_dm_i, T_tot_i)
                                for pop_s_i, ngal_i, T_dm_i, T_tot_i
                                in izip(pop_s, ngal, T_dm, T_tot)])

    # Galaxy - stars/gas spectra
    if include_baryons:
        Ps_c = F_k1 * np.array([baryons.GS_cen_spectrum(hmf, z, rho_stars_i,
                                                        rho_mean,
                                                        expansion_stars,
                                                        pop_c_i, ngal_i,
                                                        k_range_lin,
                                                        radius_range_lin,
                                                        mass_range,
                                                        T_stars_i, T_tot_i)
                                 for rho_stars_i, pop_c_i, ngal_i, \
                                     T_stars_i, T_tot_i
                                 in izip(rho_stars, pop_c, ngal, T_stars,
                                         T_tot)])
        Ps_s = F_k1 * np.array([baryons.GS_sat_spectrum(hmf, z, rho_stars_i,
                                                        rho_mean, expansion,
                                                        pop_s_i, ngal_i,
                                                        k_range_lin,
                                                        radius_range_lin,
                                                        mass_range,
                                                        T_dm_i, T_stars_i,
                                                        T_tot_i)
                                for rho_stars_i, pop_s_i, ngal_i, \
                                    T_dm_i, T_stars_i, T_tot_i
                                in izip(rho_stars, pop_s, ngal, T_dm,
                                        T_stars, T_tot)])
        Pgas_c = F_k1 * np.array([baryons.GGas_cen_spectrum(hmf, z, F_i,
                                                            rho_gas_i,
                                                            rho_mean,
                                                            expansion,
                                                            pop_c_i, ngal_i,
                                                            k_range_lin,
                                                            radius_range_lin,
                                                            mass_range,
                                                            T_gas_i, T_tot_i)
                                  for F_i, rho_gas_i, pop_c_i, ngal_i, \
                                      T_gas_i, T_tot_i
                                  in izip(F, rho_gas, pop_c, ngal, T_gas,
                                          T_tot)])
        Pgas_s = F_k1 * np.array([baryons.GGas_sat_spectrum(hmf, z, F_i,
                                                            rho_gas_i,
                                                            rho_mean,
                                                            expansion,
                                                            pop_s_i, ngal_i,
                                                            k_range_lin,
                                                            radius_range_lin,
                                                            mass_range,
                                                            T_dm_i, T_gas_i,
                                                            T_tot_i)
                                   for F_i, rho_gas_i, pop_s_i, ngal_i, \
                                       T_dm_i, T_gas_i, T_tot_i
                                   in izip(F, rho_gas, pop_s, ngal, T_dm,
                                           T_gas, T_tot)])

    # Combined (all) by type

    #Pg_k_s = np.array([(rho_stars_i * (Ps_c_i + Ps_s_i)) / (rho_mean)
                       #for rho_stars_i, Ps_c_i, Ps_s_i
                       #in izip(rho_stars, Ps_c, Ps_s)])
    #Pg_k_g = np.array([(rho_gas_i * (Pgas_c_i + Pgas_s_i)) / (rho_mean)
                       #for rho_gas_i, Pgas_c_i, Pgas_s_i
                       #in izip(rho_gas, Pgas_c, Pgas_s)])

    if include_baryons:
        # all components
        Pg_k = np.array([(rho_dm * (Pg_c_i + Pg_s_i + \
                                    Pg_2h_i*(rho_mean/rho_dm)) + \
                          rho_stars_i * (Ps_c_i + Ps_s_i) + \
                          rho_gas_i * (Pgas_c_i + Pgas_s_i)) / (rho_mean)
                          for Pg_c_i, Pg_s_i, Pg_2h_i, rho_stars_i, Ps_c_i, \
                              Ps_s_i, rho_gas_i, Pgas_c_i, Pgas_s_i
                          in izip(Pg_c, Pg_s, Pg_2h, rho_stars, Ps_c, Ps_s,
                                  rho_gas, Pgas_c, Pgas_s)])
    else:
        Pg_k = np.array([(rho_dm*(Pg_c_i + Pg_s_i + Pg_2h_i * \
                                  (rho_mean/rho_dm))) / (rho_mean)
                         for Pg_c_i, Pg_s_i, Pg_2h_i
                         in izip(Pg_c, Pg_s, Pg_2h)])

    # Normalized sattelites and centrals for sigma and d_sigma

    #Pg_c2 = (rho_dm/rho_mean)*np.array([Pg_c_i for Pg_c_i in izip(Pg_c)])
    #Pg_s2 = (rho_dm/rho_mean)*np.array([Pg_s_i for Pg_s_i in izip(Pg_s)])

    #Ps_c2 = np.array([(rho_stars_i/rho_mean) * Ps_c_i
                      #for rho_stars_i, Ps_c_i in izip(rho_stars, Ps_c)])
    #Ps_s2 = np.array([(rho_stars_i/rho_mean) * Ps_s_i
                      #for rho_stars_i, Ps_s_i inizip(rho_stars, Ps_s)])

    #Pgas_c2 =  np.array([(rho_gas_i/rho_mean) * Pgas_c_i
                         #for rho_gas_i, Pgas_c_i in izip(rho_gas, Pgas_c)])
    #Pgas_s2 =  np.array([(rho_gas_i/rho_mean) * Pgas_s_i
                         #for rho_gas_i, Pgas_s_i in izip(rho_gas, Pgas_s)])

    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
               for Pg_k_i in izip(Pg_k)]

    """
    # Correlation functions
    """

    xi2 = np.zeros((M_bin_min.size, radius_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2[i] = power_to_corr_ogata(P_inter[i], radius_range_3d)
        #xi2[xi2 <= 0.0] = np.nan
        #xi2[i,:] = fill_nan(xi2[i,:])

    """
    # Projected surface density
    """

    sur_den2 = np.array([sigma(xi2_i, rho_mean, radius_range_3d,
                               radius_range_3d_i)
                         for xi2_i in izip(xi2)])
    for i in xrange(M_bin_min.size):
        sur_den2[i][sur_den2[i] <= 0.0] = np.nan
        sur_den2[i][sur_den2[i] >= 10.0**20.0] = np.nan
        sur_den2[i] = fill_nan(sur_den2[i])

    """
    # Excess surface density
    """
    #mass_range_dm = mass_range*baryons.f_dm(omegab, omegac)
    #from dark_matter import av_delta_NFW
    #NFW_d_sigma_av = (rho_dm/rho_mean) * \
                      #np.array([av_delta_NFW(hmf.dndm, z, rho_mean, pop_c_i,
                                             #mass_range,
                                             #radius_range_2d_i) / 1e12
                                #for pop_c_i in izip(pop_c)])

    d_sur_den2 = np.array([np.nan_to_num(d_sigma(sur_den2_i,
                                                 radius_range_3d_i,
                                                 radius_range_2d_i))
                           for sur_den2_i in izip(sur_den2)]) / 1e12
    #for i in xrange(len(M_bin_min)):
        #d_sur_den2[i][d_sur_den2[i] <= 0.0] = np.nan
        #d_sur_den2[i][d_sur_den2[i] >= 10.0**20.0] = np.nan
        #d_sur_den2[i] = fill_nan(d_sur_den2[i])

    out_esd_tot =np.array([UnivariateSpline(radius_range_2d_i,
                                            np.nan_to_num(d_sur_den2_i), s=0)
                           for d_sur_den2_i in izip(d_sur_den2)])
    out_esd_tot_inter = np.zeros((M_bin_min.size, radius_range_2d_i.size))
    for i in xrange(M_bin_min.size):
        out_esd_tot_inter[i] = out_esd_tot[i](radius_range_2d_i)

    #print np.nan_to_num(out_esd_tot_inter)
    #print effective_mass
    #print z, f, sigma_c, A, M_1, gamma_1, gamma_2, alpha_s, b_0, b_1, b_2

    #end = time.time()
    #print end-start

    # Add other outputs as needed. Total ESD should always be first!
    return [out_esd_tot_inter, effective_mass.T[0], 0]


if __name__ == '__main__':
    print 0
