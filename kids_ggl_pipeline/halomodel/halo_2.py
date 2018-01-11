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
if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
from time import time
from astropy.cosmology import LambdaCDM

from hmf import MassFunction

from . import baryons, longdouble_utils as ld
from .tools import (
    Integrate, Integrate1, extrap1d, extrap2d, fill_nan, gas_concentration,
    star_concentration, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, power_to_corr_ogata,
    wp, wp_beta_correction)
from .dark_matter import (
    NFW, NFW_Dc, NFW_f, Con, DM_mm_spectrum, GM_cen_spectrum, GM_sat_spectrum,
    delta_NFW, MM_analy, GM_cen_analy, GM_sat_analy, GG_cen_analy,
    GG_sat_analy, GG_cen_sat_analy, miscenter)
from .cmf import *

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
# Integrals of mass functions with density profiles and population functions.
"""

def f_k(k_x):
    F = sp.erf(k_x/0.1) #0.05!
    return F


def n_gal(mass_func, population, m_x):
    """
    Calculates average number of galaxies!
        
    """
    #print mass_func.dndm.shape, population.shape, m_x.shape
    #c = UnivariateSpline(m_x, (mass_func.dndm * population), s=0, ext=0)
    #integ = lambda x: c(x)
    #return quad(integ, m_x[0], m_x[-1], full_output=1)[0]
    return trapz(mass_func.dndm * population, m_x)


def eff_mass(z, mass_func, population, m_x):
    #integ1 = mass_func.dndlnm*population
    #integ2 = mass_func.dndm*population
    #mass = Integrate(integ1, m_x)/Integrate(integ2, m_x)
    #return mass
    return trapz(mass_func.dndlnm * population, m_x) / \
        trapz(mass_func.dndm * population, m_x)


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
    A = 1.0 + 0.24 * y * exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * exp(-(4 / y) ** 4)
    c = 2.4
    #print y, A, a, B, b, C, c
    return 1 - A * nu**a / (nu**a + hmf.delta_c**a) + B * nu**b + C * nu**c


    """
    # Two halo term for matter-galaxy and galaxy-galaxy specta!
    # For matter-matter it is only P_lin!
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


def TwoHalo_gg(mass_func, norm, population, k_x, r_x, m_x):
    
    b_g = trapz(mass_func.dndlnm * population * \
                Bias_Tinker10(mass_func, r_x) / m_x, m_x) / norm
                
    return (mass_func.power * b_g**2.0), b_g**2.0


def gamma(theta, R, lnk_min=-13., lnk_max=17., n_bins=10000):
    np.seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')
    # making them local doesn't seem to make any difference
    # but keeping for now
    _array = array
    _izip = izip
    _logspace = logspace
    _linspace = linspace
              
    sigma_8, H0, omegam, omegab, omegav, n, \
    z, f, sigma_c, A, M_1, gamma_1, gamma_2, \
    fc_nsat, alpha_s, b_0, b_1, b_2, \
    alpha_star, beta_gas, r_t0, r_c0, p_off, r_off, bias, beta, \
    M_bin_min, M_bin_max, \
    Mstar, \
    centrals, satellites, miscentering, taylor_procedure, include_baryons, \
    simple_hod, smth1, smth2 = theta
    # hard-coded in the current version
    Ac2s = 0.56
    M_min = 9. #5.
    M_max = 16. #16.
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
    if not np.iterable(beta):
        beta = np.array([beta]*M_bin_min.size)

    # repurposing the p_off and r_off to use for different sizes of x-axis for clustering and lensing
    
    exclude_bins_esd = p_off
    exclude_bins_wp = r_off

    concentration = np.array([Con(np.float64(z_i), mass_range, np.float64(f_i)) for z_i, f_i in _izip(z,f)])

    concentration_sat = np.array([Con(np.float64(z_i), mass_range, np.float64(f_i*fc_nsat_i)) for z_i, f_i,fc_nsat_i in _izip(z,f,fc_nsat)])
    
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
                                    'dlnk': k_step, 'transfer_model': 'EH',
                                    'z':np.float64(z_i)})

    # Calculation
    # Tinker10 should also be read from theta!
    #to = time()
    hmf = _array([])
    h = H0/100.0
    cosmo_model = LambdaCDM(H0=H0, Ob0=omegab, Om0=omegam, Ode0=omegav, Tcmb0=2.725)
    for i in transfer_params:
        hmf_temp = MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=M_step,
                                hmf_model='Tinker10', delta_h=200.0, delta_wrt='mean',
                                delta_c=1.686,
                                **i)
        hmf_temp.update(cosmo_model=cosmo_model)
        hmf = np.append(hmf, hmf_temp)

    mass_func = np.zeros((z.size, mass_range.size))
    rho_mean = np.zeros(z.shape)
    rho_crit = np.zeros(z.shape)
    #rho_dm = np.zeros(z.shape)
    
    omegab = hmf[0].cosmo.Ob0
    omegac = hmf[0].cosmo.Om0-omegab
    omegav = hmf[0].cosmo.Ode0
    
    for i in xrange(z.size):
        mass_func[i] = hmf[i].dndlnm
        rho_mean[i] = hmf[i].mean_density0
        rho_crit[i] = rho_mean[i] / (omegac+omegab)
        #rho_dm[i] = rho_mean[i]# * baryons.f_dm(omegab, omegac)


    rvir_range_lin = _array([virial_radius(mass_range, rho_mean_i, 200.0)
                         for rho_mean_i in rho_mean])
    rvir_range = np.log10(rvir_range_lin)
    rvir_range_3d = _logspace(-3.2, 4, 200, endpoint=True)
    rvir_range_3d_i = _logspace(-2.5, 1.2, 25, endpoint=True)
    rvir_range_2d_i = R[0][1:]
    rvir_range_2d_i = np.unique(rvir_range_2d_i)

    # Calculating halo model
    
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
    pop_g = pop_c + pop_s
    
    ngal = _array([n_gal(hmf_i, pop_g_i , mass_range)
                   for hmf_i, pop_g_i in _izip(hmf, pop_g)])
        
    effective_mass = _array([eff_mass(np.float64(z_i), hmf_i, pop_g_i, mass_range)
                        for z_i, hmf_i, pop_g_i in _izip(z, hmf, pop_g)])
    # Why does this only have pop_c and not pop_g?
    # Do we need it? It's not used anywhere in the code
    effective_mass_bar = _array([eff_mass(np.float64(z_i), hmf_i, pop_g_i, mass_range) * \
                        (1.0 - baryons.f_dm(omegab, omegac))
                        for z_i, hmf_i, pop_g_i in _izip(z, hmf, pop_g)])
    """
    import matplotlib.pyplot as pl
    pl.plot(mass_range, pop_c[0], color='black', ls=':')
    pl.plot(mass_range, pop_s[0], color='black', ls='-.')
    pl.plot(mass_range, pop_g[0], color='black', ls='-')
                       
    pl.plot(mass_range, pop_c[1], color='orange', ls=':')
    pl.plot(mass_range, pop_s[1], color='orange', ls='-.')
    pl.plot(mass_range, pop_g[1], color='orange', ls='-')
    
    pl.plot(mass_range, pop_c[2], color='red', ls=':')
    pl.plot(mass_range, pop_s[2], color='red', ls='-.')
    pl.plot(mass_range, pop_g[2], color='red', ls='-')
    
    pl.xscale('log')
    pl.yscale('log')
    pl.ylim([0.1, 100])
    pl.xlim([1e10, 1e14])
    pl.show()
                       
    #from scipy.interpolate import InterpolatedUnivariateSpline as spline
    #nu = spline(hmf[0].nu,hmf[0].M,k=5)
                       
    #print nu(1)
                       
    #pl.plot(hmf[0].nu, hmf[0].M)
    #pl.yscale('log')
    #pl.show()
    #quit()
    """

    """
    # Power spectrum
    """
                   
                   
    # damping of the 1h power spectra at small k
    F_k1 = f_k(k_range_lin)
    # Fourier Transform of the NFW profile
    u_k = _array([NFW_f(np.float64(z_i), rho_mean_i, np.float64(f_i), mass_range,\
                rvir_range_lin_i, k_range_lin,\
                c=concentration_i) for rvir_range_lin_i, rho_mean_i, z_i,\
                f_i, concentration_i in _izip(rvir_range_lin, \
                rho_mean, z, f, concentration)])
                   
    # and of the NFW profile of the satellites
    """
    uk_s = _array([NFW_f(np.float64(z_i), rho_mean_i, np.float64(fc_nsat_i), \
                mass_range, rvir_range_lin_i, k_range_lin)
                for rvir_range_lin_i, rho_mean_i, z_i, fc_nsat_i in \
                _izip(rvir_range_lin, rho_mean, z, fc_nsat)])
    """
    uk_s = _array([NFW_f(np.float64(z_i), rho_mean_i, np.float64(f_i), mass_range,\
                rvir_range_lin_i, k_range_lin,\
                c=concentration_i) for rvir_range_lin_i, rho_mean_i, z_i,\
                f_i, concentration_i in _izip(rvir_range_lin, \
                rho_mean, z, f, concentration_sat)])
    #uk_s = u_k
    uk_s = uk_s/uk_s[:,0][:,None]
                   
    # If there is miscentering to be accounted for
    if miscentering:
        if not np.iterable(p_off):
            p_off = _array([p_off]*M_bin_min.size)
        if not np.iterable(r_off):
            r_off = _array([r_off]*M_bin_min.size)
        u_k = _array([NFW_f(np.float64(z_i), rho_mean_i, np.float64(f_i), \
                            mass_range, rvir_range_lin_i, k_range_lin,
                            c=concentration_i) * miscenter(p_off_i, r_off_i, mass_range,
                            rvir_range_lin_i, k_range_lin,
                            c=concentration_i) for \
                            rvir_range_lin_i, rho_mean_i, z_i, f_i, concentration_i, p_off_i, r_off_i
                            in _izip(rvir_range_lin, rho_mean, z, f, concentration, p_off, r_off)])
    u_k = u_k/u_k[:,0][:,None]

    # Galaxy - dark matter spectra (for lensing)
    Pg_2h = bias * _array([TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin,
                    rvir_range_lin_i, mass_range)[0]
                    for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i in \
                    _izip(rvir_range_lin, hmf, ngal, pop_g)])
    
    bias_out = bias.T[0] * _array([TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin,
                    rvir_range_lin_i, mass_range)[1]
                    for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i in \
                    _izip(rvir_range_lin, hmf, ngal, pop_g)])
        
        
    if centrals:
        Pg_c = F_k1 * _array([GM_cen_analy(hmf_i, u_k_i, rho_mean_i, pop_c_i,
                    ngal_i, mass_range)
                    for rho_mean_i, hmf_i, pop_c_i, ngal_i, u_k_i in\
                    _izip(rho_mean, hmf, pop_c, ngal, u_k)])
    else:
        Pg_c = np.zeros((n_bins_obs,n_bins))
    if satellites:
        Pg_s = F_k1 * _array([GM_sat_analy(hmf_i, u_k_i, uk_s_i, rho_mean_i,
                    pop_s_i, ngal_i, mass_range)
                    for rho_mean_i, hmf_i, pop_s_i, ngal_i, u_k_i, uk_s_i in\
                    _izip(rho_mean, hmf, pop_s, ngal, u_k, uk_s)])
    else:
        Pg_s = np.zeros((n_bins_obs,n_bins))



    Pg_k = _array([(Pg_c_i + Pg_s_i) + Pg_2h_i
                   for Pg_c_i, Pg_s_i, Pg_2h_i
                   in _izip(Pg_c, Pg_s, Pg_2h)])
    
    
    
    # Galaxy - galaxy spectra (for clustering)
    Pgg_2h = bias * _array([TwoHalo_gg(hmf_i, ngal_i, pop_g_i, k_range_lin,
                    rvir_range_lin_i, mass_range)[0]
                    for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i in \
                    _izip(rvir_range_lin, hmf, ngal, pop_g)])

    ncen = _array([n_gal(hmf_i, pop_c_i, mass_range)
                    for hmf_i, pop_c_i in _izip(hmf, pop_c)])
    nsat = _array([n_gal(hmf_i, pop_s_i, mass_range)
                   for hmf_i, pop_s_i in _izip(hmf, pop_s)])

    """
    Pgg_c = F_k1 * _array([GG_cen_analy(hmf_i, ncen_i*np.ones(k_range_lin.shape),
                    ngal_i*np.ones(k_range_lin.shape), mass_range)
                    for hmf_i, ncen_i, ngal_i in\
                    _izip(hmf, ncen, ngal)])
    """
    Pgg_c = np.zeros((n_bins_obs,n_bins))
    #beta = np.ones(M_bin_min.size)
    Pgg_s = F_k1 * _array([GG_sat_analy(hmf_i, u_k_i, pop_s_i, ngal_i, beta_i, mass_range)
                    for hmf_i, u_k_i, pop_s_i, ngal_i, beta_i in\
                    _izip(hmf, uk_s, pop_s, ngal, beta)])
                            
    Pgg_cs = F_k1 * _array([GG_cen_sat_analy(hmf_i, u_k_i, pop_c_i,
                    pop_s_i, ngal_i, mass_range)
                    for hmf_i, pop_c_i, pop_s_i, ngal_i, u_k_i in\
                    _izip(hmf, pop_c, pop_s, ngal, uk_s)])
                            
    Pgg_k = _array([(Pgg_c_i + (2.0 * Pgg_cs_i) + Pgg_s_i) + Pgg_2h_i
                    for Pgg_c_i, Pgg_cs_i, Pgg_s_i, Pgg_2h_i
                    in _izip(Pgg_c, Pgg_cs, Pgg_s, Pgg_2h)])
                            
    #"""
    # Matter - matter spectra
    Pmm_1h = F_k1 * _array([MM_analy(hmf_i, u_k_i, rho_mean_i, mass_range)
                    for hmf_i, u_k_i, rho_mean_i, beta_i in\
                    _izip(hmf, u_k, rho_mean, beta)])
                            
    Pmm = _array([Pmm_1h_i + hmf_i.power
                    for Pmm_1h_i, hmf_i
                    in _izip(Pmm_1h, hmf)])

    #"""
    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                    for Pg_k_i in _izip(Pg_k)]
        
    P_inter_2 = [UnivariateSpline(k_range, np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in _izip(Pgg_k)]
    #"""
    P_inter_3 = [UnivariateSpline(k_range, np.log(Pmm_i), s=0, ext=0)
                    for Pmm_i in _izip(Pmm)]
    #"""

                            
    """
    # Correlation functions
    """
                            
    xi2 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2[i] = power_to_corr_ogata(P_inter[i], rvir_range_3d)

    xi2_2 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2_2[i] = power_to_corr_ogata(P_inter_2[i], rvir_range_3d)
    #"""
    xi2_3 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2_3[i] = power_to_corr_ogata(P_inter_3[i], rvir_range_3d)
    #"""


    """
    # Projected surface density
    """
        
    sur_den2 = _array([sigma(xi2_i, rho_mean_i, rvir_range_3d, rvir_range_3d_i)
                        for xi2_i, rho_mean_i in _izip(xi2, rho_mean)])
    for i in xrange(M_bin_min.size):
        sur_den2[i][(sur_den2[i] <= 0.0) | (sur_den2[i] >= 1e20)] = np.nan
        sur_den2[i] = fill_nan(sur_den2[i])

    sur_den2_2 = _array([rho_mean_i * wp(xi2_2_i, rvir_range_3d, rvir_range_3d_i)
                         for xi2_2_i, rho_mean_i in _izip(xi2_2, rho_mean)])

    w_p = np.zeros((M_bin_min.size,rvir_range_3d_i.size))
    for i in xrange(M_bin_min.size):
        sur_den2_2[i][(sur_den2_2[i] <= 0.0) | (sur_den2_2[i] >= 1e20)] = np.nan
        sur_den2_2[i] = fill_nan(sur_den2_2[i])
        w_p[i] = sur_den2_2[i]/rho_mean[i]

    #"""
    sur_den2_3 = _array([sigma(xi2_3_i, rho_mean_i, rvir_range_3d, rvir_range_3d_i)
                         for xi2_3_i, rho_mean_i in _izip(xi2_3, rho_mean)])
    for i in xrange(M_bin_min.size):
        sur_den2_3[i][(sur_den2_3[i] <= 0.0) | (sur_den2_3[i] >= 1e20)] = np.nan
        sur_den2_3[i] = fill_nan(sur_den2_3[i])
    #"""

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
    
    d_sur_den2 = _array([np.nan_to_num(d_sigma(sur_den2_i,
                                               rvir_range_3d_i,
                                               rvir_range_2d_i))
                         for sur_den2_i in _izip(sur_den2)]) / 1e12
    #"""
    d_sur_den2_2 = _array([np.nan_to_num(d_sigma(sur_den2_2_i,
                                                rvir_range_3d_i,
                                                rvir_range_2d_i))
                        for sur_den2_2_i in _izip(sur_den2_2)]) / 1e12
                         
    
    d_sur_den2_3 = _array([np.nan_to_num(d_sigma(sur_den2_3_i,
                                                 rvir_range_3d_i,
                                                 rvir_range_2d_i))
                        for sur_den2_3_i in _izip(sur_den2_3)]) / 1e12
    #"""

                         
                         
                         
    out_esd_tot = _array([UnivariateSpline(rvir_range_2d_i,
                        np.nan_to_num(d_sur_den2_i), s=0)
                        for d_sur_den2_i in _izip(d_sur_den2)])
                         
    out_esd_tot_inter = np.zeros((M_bin_min.size, rvir_range_2d_i.size))
    
        
    #"""
    out_esd_tot_2 = _array([UnivariateSpline(rvir_range_2d_i,
                            np.nan_to_num(d_sur_den2_2_i), s=0)
                            for d_sur_den2_2_i in _izip(d_sur_den2_2)])
                         
    out_esd_tot_inter_2 = np.zeros((M_bin_min.size, rvir_range_2d_i.size))
                         
    
    out_esd_tot_3 = _array([UnivariateSpline(rvir_range_2d_i,
                            np.nan_to_num(d_sur_den2_3_i), s=0)
                            for d_sur_den2_3_i in _izip(d_sur_den2_3)])
                         
    out_esd_tot_inter_3 = np.zeros((M_bin_min.size, rvir_range_2d_i.size))
    #"""

    for i in xrange(M_bin_min.size):
        out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)
        out_esd_tot_inter_2[i] = out_esd_tot_2[i](rvir_range_2d_i)
        out_esd_tot_inter_3[i] = out_esd_tot_3[i](rvir_range_2d_i)

    if include_baryons:
        pointmass = _array([Mi/(np.pi*rvir_range_2d_i**2.0)/1e12 \
                        for Mi in izip(effective_mass_bar)])
    else:
        pointmass = _array([(10.0**Mi[0])/(np.pi*rvir_range_2d_i**2.0)/1e12 \
                        for Mi in izip(Mstar)])
                        
    out_esd_tot_inter = out_esd_tot_inter + pointmass
    
    print(sigma_c, A, M_1, gamma_1, gamma_2)
    #print z, f, sigma_c, A, M_1, gamma_1, gamma_2, alpha_s, b_0, b_1, b_2
    
    # Add other outputs as needed. Total ESD should always be first!

    sur_den2_2_out = _array([UnivariateSpline(rvir_range_3d_i,
                            np.nan_to_num(wp_i), s=0)
                             for wp_i in _izip(w_p)])
        
    sur_den2_2_out_inter = np.zeros((M_bin_min.size, rvir_range_2d_i.size))
    for i in xrange(M_bin_min.size):
        sur_den2_2_out_inter[i] = sur_den2_2_out[i](rvir_range_2d_i)

    # This is for w_p and esd, separated in bins
    """
    out_esd_tot_inter = _array([out_esd_tot_inter_i * (1.0+z_i)**2.0 for out_esd_tot_inter_i, z_i in izip(out_esd_tot_inter, z)])
    wp_out = np.vstack((out_esd_tot_inter, sur_den2_2_out_inter))
    #wp_out = wp_out.flatten()
    #print(wp_out)
    """

    # This is for w_p and esd, used in one bin
    """
    out_esd_tot_inter = _array([out_esd_tot_inter_i * (1.0+z_i)**2.0 for out_esd_tot_inter_i, z_i in izip(out_esd_tot_inter, z)])
    if exclude_bins_esd != 0:
        out_esd_tot_inter = np.delete(out_esd_tot_inter, exclude_bins_esd-1, axis=1)
    if exclude_bins_wp != 0:
        sur_den2_2_out_inter = np.delete(sur_den2_2_out_inter, exclude_bins_wp-1, axis=1)

    wp_out = np.hstack((out_esd_tot_inter.flatten(), sur_den2_2_out_inter.flatten()))
    #"""

    return np.array([k_range, Pg_k, Pgg_k, Pmm, rvir_range_3d, xi2, xi2_2, xi2_3, rvir_range_3d_i, sur_den2, sur_den2_2_out_inter, sur_den2_3, rvir_range_2d_i, out_esd_tot_inter, out_esd_tot_inter_2, out_esd_tot_inter_3, effective_mass, rho_mean])
    #return [wp_out]
    #return [k_range, Pg_c, Pg_s, Pg_2h, Pg_k, Pgg_s, Pgg_cs, Pgg_2h, Pgg_k, Pmm_1h, [hmf[0].power, hmf[1].power, hmf[2].power], Pmm]

if __name__ == '__main__':
    print(0)
