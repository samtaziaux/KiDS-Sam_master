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

from __future__ import print_function
import multiprocessing as multi
import numpy as np
import mpmath as mp
import longdouble_utils as ld
import matplotlib.pyplot as pl
import scipy
from numpy import arange, array, exp, linspace, logspace, ones
from scipy import special as sp
from scipy.integrate import simps, trapz, quad
from scipy.interpolate import interp1d, interp2d, UnivariateSpline, \
    SmoothBivariateSpline, RectBivariateSpline, interpn, RegularGridInterpolator
from itertools import count, izip
import itertools
from time import time
from astropy.cosmology import LambdaCDM

from hmf import MassFunction

import baryons
from tools import Integrate, Integrate1, extrap1d, extrap2d, fill_nan, \
    gas_concentration, star_concentration, virial_mass, \
        virial_radius
from lens import power_to_corr, power_to_corr_multi, sigma, d_sigma, \
    power_to_corr_ogata, wp, wp_beta_correction
from dark_matter import NFW, NFW_Dc, NFW_f, Con, DM_mm_spectrum, \
    GM_cen_spectrum, GM_sat_spectrum, delta_NFW, \
        MM_analy, GM_cen_analy, GM_sat_analy, GG_cen_analy, \
            GG_sat_analy, GG_cen_sat_analy, miscenter
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

#@memoize
"""
def Mass_Function(M_min, M_max, step, name, **cosmology_params):
    return MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=step,
                        hmf_model=name, delta_h=200.0, delta_wrt='mean',
                        delta_c=1.686,
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


def sigma_crit_kids(hmf, z_in, z_epsilon, srclim, spec_cat_path):
    
    from kids_ggl_pipeline.esd_production.shearcode_modules import calc_Sigmacrit
    import astropy.io.fits as fits
    
    zsrcbins = np.arange(0.025,3.5,0.05)
    
    Dcsbins = np.array((hmf[0].cosmo.comoving_distance(zsrcbins).to('pc')).value)
    Dc_epsilon = (hmf[0].cosmo.comoving_distance(z_epsilon).to('pc')).value

    spec_cat = fits.open(spec_cat_path, memmap=True)[1].data
    Z_S = spec_cat['z_spec']
    spec_weight = spec_cat['spec_weight']
    manmask = spec_cat['MASK']
    srcmask = (manmask==0)

    sigma_selection = {}
    # 10 lens redshifts for calculation of Sigma_crit
    lens_redshifts = np.linspace(0.0, 0.5, 10, endpoint=True)
    lens_comoving = np.array((hmf[0].cosmo.comoving_distance(lens_redshifts).to('pc')).value)
            
            
    lens_angular = lens_comoving/(1.0+lens_redshifts)
    k = np.zeros_like(lens_redshifts)
            
    for i in xrange(lens_redshifts.size):
        srcmask *= (lens_redshifts[i]+z_epsilon <= spec_cat['Z_B']) & (spec_cat['Z_B'] < srclim)
        srcNZ_k, spec_weight_k = Z_S[srcmask], spec_weight[srcmask]
                    
        srcPZ_k, bins_k = np.histogram(srcNZ_k, range=[0.025, 3.5], bins=70, weights=spec_weight_k, density=1)
        srcPZ_k = srcPZ_k/srcPZ_k.sum()
        k[i], kmask = calc_Sigmacrit(np.array([lens_comoving[i]]), np.array([lens_angular[i]]), Dcsbins, srcPZ_k, 3, Dc_epsilon)
            
    k_interpolated = interp1d(lens_redshifts, k, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0))

    return 1.0/k_interpolated(z_in)


def survey_variance(mass_func, W_p, k_range, volume):
    
    # Seems to be about right! To be checked again!.
    
    P_lin = mass_func.power
    
    #integ1 = W_p(np.exp(k_range)) * np.exp(k_range)**2.0
    #v_w = 4.0*np.pi * simps(integ1, np.exp(k_range))
    v_w = volume
    
    integ2 = W_p(np.exp(k_range))**2.0 * np.exp(k_range)**2.0 * P_lin
    sigma = (1.0 / (2.0*np.pi**2.0 * v_w**2.0)) * simps(integ2, np.exp(k_range))
    
    return sigma


# Routines for T_xxxx - connected (non-Gaussian) part of the covariance
# This is following Pielorz et al. 2010 (see also Benjamin's implementation)
def pt_kernel_alpha(k1, k2, mu):
    return 1.0 + ((k2/k1) * mu)


def pt_kernel_beta(k1, k2, mu):
    return (mu/2.0) * ((k1/k2) + (k2/k1) + 2.0*mu)


def pt_kernel_f2(k1, k2, mu):
    return 5.0/7.0 + ((2.0/7.0) * mu*mu) + (0.5 * mu * (k1/k2 + k2/k1))


def pt_kernel_g2(k1, k2, mu):
    return 3.0/7.0 + ((4.0/7.0) * mu*mu) + (0.5 * mu * (k1/k2 + k2/k1))


def pt_kernel_f3(k1, k2, mu, trispec_matter_mulim, trispec_matter_klim):
    
    if np.fabs(k1-k2) < trispec_matter_klim:
        k_m = np.zeros(mu.shape) #avoid nan in sqrt
        mu_1m = np.zeros(mu.shape)   #undefined
        alpha_m = np.ones(mu.shape)
        beta_m = np.zeros(mu.shape)  # undefined
    
    else:
    
        k_m = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu) # |k_-|
        mu_1m = (k2/k_m)*mu - (k1/k_m) # (k1*k_-)/[k1 k_-]
        alpha_m = pt_kernel_alpha(k_m, k1, mu_1m)
        beta_m = pt_kernel_beta(k1, k_m, mu_1m)
        
        k_m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        mu_1m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        alpha_m[((1.0-mu) < trispec_matter_mulim)] = 1.0
        beta_m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        

    if np.fabs(k1-k2) < trispec_matter_klim:
        k_p = np.zeros(mu.shape) # avoid nan in sqrt
        mu_1p = np.zeros(mu.shape) # undefined
        alpha_p = np.ones(mu.shape)
        beta_p = np.zeros(mu.shape) # undefined
        
    else:

        k_p = np.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu) # |k_+|
        mu_1p = (k1/k_p) + mu*(k2/k_p) # (k1*k_+)/[k1 k_+]
        alpha_p = pt_kernel_alpha(k_p, k1, (-1.0)*mu_1p)
        beta_p = pt_kernel_beta(k1, k_p, (-1.0)*mu_1p)
        
        k_p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        mu_1p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        alpha_p[((mu+1.0) < trispec_matter_mulim)] = 1.0
        beta_p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        
            
    F2_plus=pt_kernel_f2(k1, k2, mu)
    F2_minus=pt_kernel_f2(k1, k2, (-1.0)*mu)
    G2_plus=pt_kernel_g2(k1, k2, mu)
    G2_minus=pt_kernel_g2(k1, k2, (-1.0)*mu)

    return ((7.0/54.0)*(alpha_m*F2_minus + alpha_p*mu_1p*F2_plus) + (4.0/54.0)*(beta_m*G2_minus + beta_p*G2_plus) + (7.0/54.0)*(alpha_m*G2_minus + alpha_p*G2_plus))


def trispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    
    if np.fabs(k1-k2) < trispec_matter_klim:
        k_m = np.zeros(mu.shape) #avoid nan in sqrt
        mu_1m = np.zeros(mu.shape)   #undefined
        mu_2m = np.zeros(mu.shape)
        p_m = np.zeros(mu.shape)
        F2_1m = np.zeros(mu.shape)  # undefined
        F2_2m = np.zeros(mu.shape)
    
    else:

        k_m = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu) # |k_-|
        mu_1m = (k2/k_m)*mu - (k1/k_m) # (k1*k_-)/[k1 k_-]
        mu_2m = (k2/k_m) - mu*(k1/k_m) # (k2*k_-)/[k2 k_-]
        p_m = np.exp(P_lin_inter(np.log(k_m)))
        F2_1m = pt_kernel_f2(k1, k_m, mu_1m)
        F2_2m = pt_kernel_f2(k2, k_m, (-1.0)*mu_2m)
        
        k_m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        mu_1m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        mu_2m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        p_m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        F2_1m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        F2_2m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        

    if np.fabs(k1-k2) < trispec_matter_klim:
        k_p = np.zeros(mu.shape) #avoid nan in sqrt
        mu_1p = np.zeros(mu.shape)   #undefined
        mu_2p = np.zeros(mu.shape)
        p_p = np.zeros(mu.shape)
        F2_1p = np.zeros(mu.shape)  # undefined
        F2_2p = np.zeros(mu.shape)
    
    else:

        k_p = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu) # |k_+|
        mu_1p = (k1/k_p) + mu*(k2/k_p) # (k1*k_+)/[k1 k_+]
        mu_2p = (k1/k_p)*mu + (k2/k_p) # (k2*k_+)/[k2 k_+]
        p_p = np.exp(P_lin_inter(np.log(k_p)))
        F2_1p = pt_kernel_f2(k1, k_p, mu_1p)
        F2_2p = pt_kernel_f2(k2, k_p, (-1.0)*mu_2p)
        
        k_p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        mu_1p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        mu_2p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        p_p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        F2_1p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        F2_2p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        

    p1 = np.exp(P_lin_inter(np.log(k1)))
    p2 = np.exp(P_lin_inter(np.log(k2)))

    F3_12 = pt_kernel_f3(k1, k2, mu, trispec_matter_mulim, trispec_matter_klim)
    F3_21 = pt_kernel_f3(k2, k1, mu, trispec_matter_mulim, trispec_matter_klim)

    term1 = 4.0 * p1*p1 * (F2_1p*F2_1p*p_p + F2_1m*F2_1m*p_m)
    term2 = 4.0 * p2*p2 * (F2_2p*F2_2p*p_p + F2_2m*F2_2m*p_m)
    term3 = 8.0 * p1*p2 * (F2_1p*F2_2p*p_p + F2_1m*F2_2m*p_m)
    term4 = 12.0 * (p1*p1*p2*F3_12 + p1*p2*p2*F3_21)
    
    out = term1 + term2 + term3 + term4

    return out


def bispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    
    p1 = np.exp(P_lin_inter(np.log(k1)))
    p2 = np.exp(P_lin_inter(np.log(k2)))
    
    term1 = 2.0 * pt_kernel_f2(k1, k2, mu)*p1*p2
    
    if np.fabs(k1-k2) < trispec_matter_klim:
        k_p = np.zeros(mu.shape)
        term2 = np.zeros(mu.shape)
        term3 = np.zeros(mu.shape)

    else:
        k_p = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu)
        p_p = np.exp(P_lin_inter(np.log(k_p)))
        mu_1p = (k1/k_p) + mu*(k2/k_p) # (k1*k_+)/[k1 k_+]
        mu_2p = (k1/k_p)*mu + (k2/k_p) # (k2*k_+)/[k2 k_+]
        term2 = 2.0*pt_kernel_f2(k1, k_p, (-1.0)*mu_1p)*p1*p_p
        term3 = 2.0*pt_kernel_f2(k2, k_p, (-1.0)*mu_2p)*p2*p_p
        
        k_p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        term2[((mu+1.0) < trispec_matter_mulim)] = 0.0
        term3[((mu+1.0) < trispec_matter_mulim)] = 0.0
        
    return term1 + term2 + term3


# These are integrated over 2PI to get the angular average, for each k1, k2 combination!
def intg_for_trispec_matter_parallel_2h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    if np.fabs(k1-k2) < trispec_matter_klim:
        k_m = np.zeros(mu.shape)
        p_m = np.zeros(mu.shape)
    else:
        k_m = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu)
        p_m = np.exp(P_lin_inter(np.log(k_m)))
        k_m[((1.0-mu) < trispec_matter_mulim)] = 0.0
        p_m[((1.0-mu) < trispec_matter_mulim)] = 0.0

    if np.fabs(k1-k2) < trispec_matter_klim:
        k_p = np.zeros(mu.shape)
        p_p = np.zeros(mu.shape)
    else:
        k_p = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu)
        p_p = np.exp(P_lin_inter(np.log(k_p)))
        k_p[((mu+1.0) < trispec_matter_mulim)] = 0.0
        p_p[((mu+1.0) < trispec_matter_mulim)] = 0.0

    return p_p + p_m


def intg_for_trispec_matter_parallel_3h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    return bispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim) + bispec_parallel_pt(k1, k2, (-1.0)*mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim)


def intg_for_trispec_matter_parallel_4h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    return trispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim)


def trispectra_234h(krange, P_lin_inter, mass_func, uk, bias, rho_mean, m_x, k_x):
    
    trispec_matter_mulim = 0.001
    trispec_matter_klim = 100.0#0.001
    #print(trispec_matter_klim)
    
    trispec_2h = np.zeros((krange.size, krange.size))
    trispec_3h = np.zeros((krange.size, krange.size))
    trispec_4h = np.zeros((krange.size, krange.size))
    
    # Evaluate u(k) on different k grid!
    u_k = np.array([UnivariateSpline(k_x, uk[:,m], s=0, ext=0) for m in xrange(len(m_x))])
    u_k_new = np.array([u_k[m](krange) for m in xrange(len(m_x))])
    
    def Im(i, mass_func, uk, bias, rho_mean, m_x):
        integ = mass_func.dndm * bias * uk[:,i] * m_x
        I = trapz(integ, m_x)/(rho_mean)
        return I
    
    def Imm(i, j, mass_func, uk, bias, rho_mean, m_x):
        integ = mass_func.dndm * bias * uk[:,i] * uk[:,j] * m_x**2.0
        I = trapz(integ, m_x)/(rho_mean**2.0)
        return I
    
    def Immm(i, j, k,  mass_func, uk, bias, rho_mean, m_x):
        integ = mass_func.dndm * bias * uk[:,i] * uk[:,j] * uk[:,k] * m_x**3.0
        I = trapz(integ, m_x)/(rho_mean**3.0)
        return I
    
    x = np.linspace(0.0, 2.0*np.pi, endpoint=True)
    
    for i, k1 in enumerate(krange):
        for j, k2 in enumerate(krange):
            trispec_2h[i,j] = 2.0 * Immm(i, j, j, mass_func, u_k_new, bias, rho_mean, m_x) * Im(i, mass_func, u_k_new, bias, rho_mean, m_x) * np.exp(P_lin_inter(np.log(k1))) + 2.0 * 2.0 * Immm(i, i, j, mass_func, u_k_new, bias, rho_mean, m_x) * Im(j, mass_func, u_k_new, bias, rho_mean, m_x) * np.exp(P_lin_inter(np.log(k2))) + (Imm(i, j, mass_func, u_k_new, bias, rho_mean, m_x)**2.0) * trapz(intg_for_trispec_matter_parallel_2h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)
            trispec_3h[i,j] = 2.0 * Imm(i, j, mass_func, u_k_new, bias, rho_mean, m_x) * Im(i, mass_func, u_k_new, bias, rho_mean, m_x) * Im(j, mass_func, u_k_new, bias, rho_mean, m_x) * trapz(intg_for_trispec_matter_parallel_3h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)
            trispec_4h[i,j] = (Im(i, mass_func, u_k_new, bias, rho_mean, m_x))**2.0 * (Im(j, mass_func, u_k_new, bias, rho_mean, m_x))**2.0 * trapz(intg_for_trispec_matter_parallel_4h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)


            #trispec_2h[i,j] = trapz(intg_for_trispec_matter_parallel_2h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)
            #trispec_3h[i,j] = trapz(intg_for_trispec_matter_parallel_3h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)
            #trispec_4h[i,j] = trapz(intg_for_trispec_matter_parallel_4h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)
        #print(i)
    trispec_2h = np.nan_to_num(trispec_2h)
    trispec_3h = np.nan_to_num(trispec_3h)
    trispec_4h = np.nan_to_num(trispec_4h)

    trispec_tot = trispec_2h + trispec_3h + trispec_4h
    #trispec_tot = trispec_tot + trispec_tot.T - np.diag(trispec_tot.diagonal())
    """
    print(trispec_tot)
    #from scipy import signal
    #diag = np.diagonal(trispec_tot, -1)
    #diag.setflags(write=True)
    #diag.fill(np.nan)
    #diag1 = np.diagonal(trispec_tot, 1)
    #diag1.setflags(write=True)
    #diag1.fill(np.nan)
    for i, k in enumerate(krange):
        trispec_tot[:,i][trispec_tot[:,i] > trispec_tot[i,i]] = np.nan
        #ind = signal.find_peaks_cwt(trispec_tot[:,i], np.arange(1,10))
        #trispec_tot[:,i][ind] = np.nan
        trispec_tot[:,i] = fill_nan(trispec_tot[:,i])
    print(trispec_tot)
    """
    #trispec_tot = trispec_tot + trispec_tot.T - np.diag(trispec_tot.diagonal())

    trispec_tot_interp = RectBivariateSpline(krange, krange, trispec_tot, kx=1, ky=1)
    #trispec_tot_interp = interp2d(krange, krange, trispec_tot)

    return trispec_tot_interp


def trispectra_1h(krange, mass_func, uk, rho_mean, ngal, population_cen, population_sat, m_x, k_x, x):
    
    trispec_1h = np.zeros((krange.size, krange.size))

    u_g_prod = (population_cen + population_sat * uk)
    u_m_prod = m_x * uk
    norm_g = ngal
    norm_m = rho_mean
    
    # Evaluate u(k) on different k grid!
    u_g = np.array([UnivariateSpline(k_x, u_g_prod[:,m], s=0, ext=0) for m in xrange(len(m_x))])
    u_m = np.array([UnivariateSpline(k_x, u_m_prod[:,m], s=0, ext=0) for m in xrange(len(m_x))])
    
    u_m_new = np.array([u_m[m](krange) for m in xrange(len(m_x))])
    u_g_new = np.array([u_g[m](krange) for m in xrange(len(m_x))])
    
    if x == 'gmgm':
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_g_new[:,i] * u_m_new[:,i] / (norm_g*norm_m)
                vec2 = u_g_new[:,j] * u_m_new[:,j] / (norm_g*norm_m)
                integ = mass_func.dndm * vec1 * vec2
                trispec_1h[i,j] = trapz(integ, m_x)
    
    if x == 'gggm':
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_g_new[:,i] * u_g_new[:,i] / (norm_g*norm_g)
                vec2 = u_g_new[:,j] * u_m_new[:,j] / (norm_g*norm_m)
                integ = mass_func.dndm * vec1 * vec2
                trispec_1h[i,j] = trapz(integ, m_x)
    
    if x == 'gggg':
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_g_new[:,i] * u_g_new[:,i] / (norm_g*norm_g)
                vec2 = u_g_new[:,j] * u_g_new[:,j] / (norm_g*norm_g)
                integ = mass_func.dndm * vec1 * vec2
                trispec_1h[i,j] = trapz(integ, m_x)

    if x == 'mmmm':
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_m_new[:,i] * u_m_new[:,i] / (norm_m*norm_m)
                vec2 = u_m_new[:,j] * u_m_new[:,j] / (norm_m*norm_m)
                integ = mass_func.dndm * vec1 * vec2
                trispec_1h[i,j] = trapz(integ, m_x)

    trispec_1h_interp = RectBivariateSpline(krange, krange, trispec_1h, kx=1, ky=1)
    #trispec_1h_interp = interp2d(krange, krange, trispec_1h)
    return trispec_1h_interp


def halo_model_integrals(mass_func, uk, bias, rho_mean, ngal, population_cen, population_sat, m_x, x):
    
    if x == 'g':
        integ1 = mass_func.dndm * bias * (population_cen + population_sat * uk)
        I = trapz(integ1, m_x, axis=1)/ngal
    
    if x == 'm':
        integ2 = mass_func.dndm * bias * uk * m_x
        I = trapz(integ2, m_x, axis=1)/rho_mean

    if x == 'gg':
        integ3 = mass_func.dndm * bias * (population_cen * population_sat * uk + population_sat**2.0 * uk**2.0)
        I = trapz(integ3, m_x, axis=1)/(ngal**2.0)

    if x == 'gm':
        integ4 = mass_func.dndm * bias * uk * m_x * (population_cen + population_sat * uk)
        I = trapz(integ4, m_x, axis=1)/(rho_mean*ngal)

    if x == 'mm':
        integ5 = mass_func.dndm * bias * (uk * m_x)**2.0
        I = trapz(integ5, m_x, axis=1)/(rho_mean**2.0)

    if x == 'mmm':
        integ6 = mass_func.dndm * bias * (uk * m_x)**3.0
        I = trapz(integ6, m_x, axis=1)/(rho_mean**3.0)

    return I


def calc_cov_non_gauss(params):
    
    b_i, b_j, i, j, rvir_range_2d_i, b_g, W_p, volume = params
    r_i, r_j = rvir_range_2d_i[i], rvir_range_2d_i[j]
    
    delta = np.eye(b_g.size)
    
    # the number of steps to fit into a half-period at high-k.
    # 6 is better than 1e-4.
    minsteps = 8
    
    # set min_k, 1e-6 should be good enough
    mink = 1e-6
    
    temp_min_k = 1.0
    
    # getting maxk here is the important part. It must be a half multiple of
    # pi/r to be at a "zero", it must be >1 AND it must have a number of half
    # cycles > 38 (for 1E-5 precision).
    min_k = (2.0 * np.ceil((temp_min_k * np.sqrt(r_i*r_j) / np.pi - 1.0) / 2.0) + 0.5) * np.pi / np.sqrt(r_i*r_j)
    maxk = max(501.5 * np.pi / np.sqrt(r_i*r_j), min_k)
    # Now we calculate the requisite number of steps to have a good dk at hi-k.
    nk = np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * np.sqrt(r_i*r_j)))))
    if nk > 10000:
        nk = 10000
    
    lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
    
    Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Awr_j = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_j)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Aw_rr = simps(np.exp(lnk)**2.0 * (sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j))/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    
    T234_i = T234h[b_i](np.exp(lnk), np.exp(lnk))
    T234_j = T234h[b_j](np.exp(lnk), np.exp(lnk))
    
    
    Tgmgm_i = Tgmgm[b_i](np.exp(lnk), np.exp(lnk))
    Tgggm_i = Tgggm[b_i](np.exp(lnk), np.exp(lnk))
    Tgggg_i = Tgggg[b_i](np.exp(lnk), np.exp(lnk))

    Tgmgm_j = Tgmgm[b_j](np.exp(lnk), np.exp(lnk))
    Tgggm_j = Tgggm[b_j](np.exp(lnk), np.exp(lnk))
    Tgggg_j = Tgggg[b_j](np.exp(lnk), np.exp(lnk))

    integ1 = np.outer(np.exp(lnk)**(1.0) * sp.jv(0, np.exp(lnk) * r_i), np.exp(lnk)**(1.0) * sp.jv(0, np.exp(lnk) * r_j)) * (np.sqrt(Tgggg_i * Tgggg_j) + b_g[b_i]*b_g[b_i]*b_g[b_j]*b_g[b_j]*np.sqrt(T234_i * T234_j))
    integ2 = np.outer(np.exp(lnk)**(1.0) * sp.jv(2, np.exp(lnk) * r_i), np.exp(lnk)**(1.0) * sp.jv(2, np.exp(lnk) * r_j)) * (np.sqrt(Tgmgm_i * Tgmgm_j) + b_g[b_i]*b_g[b_j]*np.sqrt(T234_i * T234_j))
    integ3 = np.outer(np.exp(lnk)**(1.0) * sp.jv(0, np.exp(lnk) * r_i), np.exp(lnk)**(1.0) * sp.jv(2, np.exp(lnk) * r_j)) * (np.sqrt(Tgggm_i * Tgggm_j) + b_g[b_i]*b_g[b_j]*np.sqrt(b_g[b_i]*b_g[b_j])*np.sqrt(T234_i * T234_j))

    T234_i, T234_j, Tgmgm_i, Tgmgm_j, Tgggm_i, Tgggm_j, Tgggg_i, Tgggg_j = [], [], [], [], [], [], [], []

    I_wp = trapz(trapz(integ1, dx=dlnk, axis=0), dx=dlnk)/volume
    I_esd = trapz(trapz(integ2, dx=dlnk, axis=0), dx=dlnk)/volume
    I_cross = trapz(trapz(integ3, dx=dlnk, axis=0), dx=dlnk)/volume

    a = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * I_wp
    b = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * I_esd
    c = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * I_cross
    
    return b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j, [a, b, c]


def cov_non_gauss(rvir_range_2d_i, b_g, W_p, volume, cov_wp, cov_esd, cov_cross):
    
    print('Calculating the connected (non-Gaussian) part of the covariance ...')
    
    b_i = xrange(len(b_g))
    b_j = xrange(len(b_g))
    i = xrange(len(rvir_range_2d_i))
    j = xrange(len(rvir_range_2d_i))
    
    paramlist = [list(tup) for tup in itertools.product(b_i,b_j,i,j)]
    for i in paramlist:
        i.extend([rvir_range_2d_i, b_g, W_p, volume])
    #print(calc_cov_non_gauss(paramlist[0]))
    #quit()
    pool = multi.Pool(processes=12)
    for i, j, val in pool.imap(calc_cov_non_gauss, paramlist):
        #print(i, j, val)
        cov_wp[i,j] = val[0]
        cov_esd[i,j] = val[1]
        cov_cross[i,j] = val[2]
    
    return cov_wp, cov_esd, cov_cross


def calc_cov_ssc(params):
    
    b_i, b_j, i, j, rvir_range_2d_i, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var = params
    r_i, r_j = rvir_range_2d_i[i], rvir_range_2d_i[j]
    
    delta = np.eye(b_g.size)
    
    # the number of steps to fit into a half-period at high-k.
    # 6 is better than 1e-4.
    minsteps = 8
    
    # set min_k, 1e-6 should be good enough
    mink = 1e-6
    
    temp_min_k = 1.0
    
    # getting maxk here is the important part. It must be a half multiple of
    # pi/r to be at a "zero", it must be >1 AND it must have a number of half
    # cycles > 38 (for 1E-5 precision).
    min_k = (2.0 * np.ceil((temp_min_k * np.sqrt(r_i*r_j) / np.pi - 1.0) / 2.0) + 0.5) * np.pi / np.sqrt(r_i*r_j)
    maxk = max(501.5 * np.pi / np.sqrt(r_i*r_j), min_k)
    # Now we calculate the requisite number of steps to have a good dk at hi-k.
    nk = np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * np.sqrt(r_i*r_j)))))
    #nk = 10000
    
    lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
    
    Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Awr_j = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_j)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Aw_rr = simps(np.exp(lnk)**2.0 * (sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j))/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    
    P_gm_i = Pgm[b_i](lnk)
    P_gm_j = Pgm[b_j](lnk)
    
    P_gg_i = Pgg[b_i](lnk)
    P_gg_j = Pgg[b_j](lnk)
    
    P_lin_i = P_lin[b_i](lnk)
    P_lin_j = P_lin[b_j](lnk)
    
    dP_lin_i = dlnP_lin[b_i](lnk)
    dP_lin_j = dlnP_lin[b_j](lnk)
    
    Ig_i = I_g[b_i](lnk)
    Ig_j = I_g[b_j](lnk)
    
    Im_i = I_m[b_i](lnk)
    Im_j = I_m[b_j](lnk)
    
    Igg_i = I_gg[b_i](lnk)
    Igg_j = I_gg[b_j](lnk)
    
    Igm_i = I_gm[b_i](lnk)
    Igm_j = I_gm[b_j](lnk)
    
    # Responses
    ps_deriv_gg_i = (68.0/21.0 - (1.0/3.0)*(dP_lin_i)) * np.exp(P_lin_i) * np.exp(Ig_i)*np.exp(Ig_i) + np.exp(Igg_i) - 2.0 * b_g[b_i] * np.exp(P_gg_i)
    ps_deriv_gg_j = (68.0/21.0 - (1.0/3.0)*(dP_lin_j)) * np.exp(P_lin_j) * np.exp(Ig_j)*np.exp(Ig_j) + np.exp(Igg_j) - 2.0 * b_g[b_j] * np.exp(P_gg_j)
    
    ps_deriv_gm_i = (68.0/21.0 - (1.0/3.0)*(dP_lin_i)) * np.exp(P_lin_i) * np.exp(Ig_i)*np.exp(Im_i) + np.exp(Igm_i) - b_g[b_i] * np.exp(P_gm_i)
    ps_deriv_gm_j = (68.0/21.0 - (1.0/3.0)*(dP_lin_j)) * np.exp(P_lin_j) * np.exp(Ig_j)*np.exp(Ig_j) + np.exp(Igm_j) - b_g[b_j] * np.exp(P_gm_i)
    
    # wp
    
    integ1 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * (np.sqrt(survey_var[b_i])*np.sqrt(survey_var[b_j])) * ps_deriv_gg_i * ps_deriv_gg_j
    a = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * trapz(integ1, dx=dlnk)
    
    # ESD
    
    integ2 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * (np.sqrt(survey_var[b_i])*np.sqrt(survey_var[b_j])) * ps_deriv_gm_i * ps_deriv_gm_j
    b = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * trapz(integ2, dx=dlnk)
    
    # cross
    
    integ3 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * (np.sqrt(survey_var[b_i])*np.sqrt(survey_var[b_j])) * ps_deriv_gg_i * ps_deriv_gm_j
    c = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * trapz(integ3, dx=dlnk)
    

    return b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j, [a, b, c]


def cov_ssc(rvir_range_2d_i, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var, cov_wp, cov_esd, cov_cross):
    
    print('Calculating the super-sample covariance ...')
    
    b_i = xrange(len(Pgm))
    b_j = xrange(len(Pgm))
    i = xrange(len(rvir_range_2d_i))
    j = xrange(len(rvir_range_2d_i))
    
    paramlist = [list(tup) for tup in itertools.product(b_i,b_j,i,j)]
    for i in paramlist:
        i.extend([rvir_range_2d_i, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var])

    pool = multi.Pool(processes=12)
    for i, j, val in pool.map(calc_cov_ssc, paramlist):
        #print(i, j, val)
        cov_wp[i,j] = val[0]
        cov_esd[i,j] = val[1]
        cov_cross[i,j] = val[2]
    
    return cov_wp, cov_esd, cov_cross


def calc_cov_gauss(params):
    
    b_i, b_j, i, j, rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal = params
    r_i, r_j = rvir_range_2d_i[i], rvir_range_2d_i[j]
    
    delta = np.eye(shape_noise.size)
    
    # the number of steps to fit into a half-period at high-k.
    # 6 is better than 1e-4.
    minsteps = 8
    
    # set min_k, 1e-6 should be good enough
    mink = 1e-6
    
    temp_min_k = 1.0
    
    # getting maxk here is the important part. It must be a half multiple of
    # pi/r to be at a "zero", it must be >1 AND it must have a number of half
    # cycles > 38 (for 1E-5 precision).
    min_k = (2.0 * np.ceil((temp_min_k * np.sqrt(r_i*r_j) / np.pi - 1.0) / 2.0) + 0.5) * np.pi / np.sqrt(r_i*r_j)
    maxk = max(501.5 * np.pi / np.sqrt(r_i*r_j), min_k)
    # Now we calculate the requisite number of steps to have a good dk at hi-k.
    nk = np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * np.sqrt(r_i*r_j)))))
    #nk = 10000
                
    lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
        
    Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Awr_j = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_j)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Aw_rr = simps(np.exp(lnk)**2.0 * (sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j))/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
                    
    P_gm_i = P_inter[b_i](lnk)
    P_gm_j = P_inter[b_j](lnk)
                    
    P_gg_i = P_inter_2[b_i](lnk)
    P_gg_j = P_inter_2[b_j](lnk)
                    
    P_mm_i = P_inter_3[b_i](lnk)
    P_mm_j = P_inter_3[b_j](lnk)
                    
                    
    # wp
    integ1 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * (np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j])*(np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j])
    integ2 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j]))
    a = ((2.0*Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ1, dx=dlnk) + ((4.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ2, dx=dlnk)
                    
    # ESD
    #integ3 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( np.sqrt(np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j])*np.sqrt(np.exp(P_mm_j) + shape_noise[b_j]* delta[b_i, b_j]) * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j])) + np.exp(P_gm_i)*np.exp(P_gm_j) )
    #integ4 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * (np.sqrt(np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j])*np.sqrt(np.exp(P_mm_j) + shape_noise[b_j]* delta[b_i, b_j]))
    integ3 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j]) * (np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j]) + np.exp(P_gm_i)*np.exp(P_gm_j) )
    integ4 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * (np.sqrt(np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j])*np.sqrt(np.exp(P_mm_j) + shape_noise[b_j]* delta[b_i, b_j]))
    b = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ3, dx=dlnk) + ((2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ4, dx=dlnk)
                    
    # cross
    #integ5 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.sqrt(np.exp(P_gm_i))*np.sqrt(np.exp(P_gm_j))) * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j])) )
    #integ6 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j]))
    #c = ((2.0*Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk) + ((4.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk)
    
    integ5 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.exp(P_gm_i)* (np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j])) + (np.exp(P_gm_j)* (np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j])) )
    integ6 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j]))
    c = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk) + ((4.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk)
    
    return b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j, [a, b, c]


def cov_gauss(rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_wp, cov_esd, cov_cross):

    print('Calculating the Gaussian part of the covariance ...')

    b_i = xrange(len(P_inter_2))
    b_j = xrange(len(P_inter_2))
    i = xrange(len(rvir_range_2d_i))
    j = xrange(len(rvir_range_2d_i))
    
    paramlist = [list(tup) for tup in itertools.product(b_i,b_j,i,j)]
    for i in paramlist:
        i.extend([rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal])
    
    pool = multi.Pool(processes=12)
    for i, j, val in pool.map(calc_cov_gauss, paramlist):
        #print(i, j, val)
        cov_wp[i,j] = val[0]
        cov_esd[i,j] = val[1]
        cov_cross[i,j] = val[2]

    return cov_wp, cov_esd, cov_cross


def covariance(theta, R, h=0.7, Om=0.315, Ol=0.685, n_bins=10000, lnk_min=-13., lnk_max=17.):
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
    if not np.iterable(beta):
        beta = np.array([beta]*M_bin_min.size)


    concentration = np.array([Con(np.float64(z_i), mass_range, np.float64(f_i))\
                          for z_i, f_i in _izip(z,f)])
    
    n_bins_obs = M_bin_min.size
    bias = np.array([bias]*k_range_lin.size).T
    
    hod_mass = _array([_logspace(Mi, Mx, 200, endpoint=False,
                                 dtype=np.longdouble)
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
    uk_s = _array([NFW_f(np.float64(z_i), rho_mean_i, np.float64(fc_nsat_i), \
                mass_range, rvir_range_lin_i, k_range_lin)
                for rvir_range_lin_i, rho_mean_i, z_i, fc_nsat_i in \
                _izip(rvir_range_lin, rho_mean, z, fc_nsat)])
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
    Pgg_2h = bias * _array([TwoHalo_gg(hmf_i, ngal_i, pop_g_i, k_range_lin, rvir_range_lin_i, mass_range)[0]
                    for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i in \
                    _izip(rvir_range_lin, hmf, ngal, pop_g)])
    """
    ncen = _array([n_gal(hmf_i, pop_c_i, mass_range)
                    for hmf_i, pop_c_i in _izip(hmf, pop_c)])
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
                            
    Pgg_cs = F_k1 * _array([GG_cen_sat_analy(hmf_i, u_k_i, pop_c_i, pop_s_i, ngal_i, mass_range)
                    for hmf_i, pop_c_i, pop_s_i, ngal_i, u_k_i in\
                    _izip(hmf, pop_c, pop_s, ngal, uk_s)])
                            
    Pgg_k = _array([(Pgg_c_i + (2.0 * Pgg_cs_i) + Pgg_s_i) + Pgg_2h_i
                    for Pgg_c_i, Pgg_cs_i, Pgg_s_i, Pgg_2h_i
                    in _izip(Pgg_c, Pgg_cs, Pgg_s, Pgg_2h)])
                            
                            
    # Matter - matter spectra
    Pmm_1h = F_k1 * _array([MM_analy(hmf_i, u_k_i, rho_mean_i, mass_range)
                    for hmf_i, u_k_i, rho_mean_i, beta_i in\
                    _izip(hmf, u_k, rho_mean, beta)])
                            
    Pmm = _array([Pmm_1h_i + hmf_i.power
                    for Pmm_1h_i, hmf_i
                    in _izip(Pmm_1h, hmf)])


    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                    for Pg_k_i in _izip(Pg_k)]
        
    P_inter_2 = [UnivariateSpline(k_range, np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in _izip(Pgg_k)]

    P_inter_3 = [UnivariateSpline(k_range, np.log(Pmm_i), s=0, ext=0)
                    for Pmm_i in _izip(Pmm)]
                    
            
    # Evaluate halo model integrals needed for SSC
    
    I_g = _array([halo_model_integrals(hmf_i, uk_i, Bias_Tinker10(hmf_i, 0), rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'g')
                                   for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                                   _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
                                   
    I_m = _array([halo_model_integrals(hmf_i, uk_i, Bias_Tinker10(hmf_i, 0), rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'm')
                                    for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                                    _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
                                    
    I_gg = _array([halo_model_integrals(hmf_i, uk_i, Bias_Tinker10(hmf_i, 0), rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'gg')
                                    for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                                    _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
                                    
    I_gm = _array([halo_model_integrals(hmf_i, uk_i, Bias_Tinker10(hmf_i, 0), rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'gm')
                                    for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                                    _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
                                    
    I_mm = _array([halo_model_integrals(hmf_i, uk_i, Bias_Tinker10(hmf_i, 0), rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'mm')
                                    for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                                    _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
    
    I_inter_g = [UnivariateSpline(k_range, np.log(I_g_i), s=0, ext=0)
               for I_g_i in _izip(I_g)]
               
    I_inter_m = [UnivariateSpline(k_range, np.log(I_m_i), s=0, ext=0)
                for I_m_i in _izip(I_m)]
                 
    I_inter_gg = [UnivariateSpline(k_range, np.log(I_gg_i), s=0, ext=0)
                for I_gg_i in _izip(I_gg)]
                 
    I_inter_gm = [UnivariateSpline(k_range, np.log(I_gm_i), s=0, ext=0)
                for I_gm_i in _izip(I_gm)]
                
    I_inter_mm = [UnivariateSpline(k_range, np.log(I_mm_i), s=0, ext=0)
                for I_mm_i in _izip(I_mm)]
    
    P_lin_inter = [UnivariateSpline(k_range, np.log(hmf_i.power), s=0, ext=0)
                for hmf_i in hmf]
               
    k3P_lin_inter = [UnivariateSpline(k_range, np.log(k_range_lin**3.0 * hmf_i.power), s=0, ext=0)
                for hmf_i in hmf]
                
    dlnk3P_lin_interdlnk = [f.derivative() for f in k3P_lin_inter]
    
    
    # Start covariance calculations (and for now set survey details)

    Pi_max = 100.0
    
    kids_area = 180 * 3600.0 #500 #To be in arminutes!
    eff_density = 8.53#6.0 #1.2#1.4#2.34#8.53 #1.5#1.85
    
    kids_variance_squared = 0.076 #0.275#0.076
    z_kids = 0.6
    
    sigma_crit = sigma_crit_kids(hmf, z, 0.2, 0.9, '/home/dvornik/data2_dvornik/KidsCatalogues/IMSIM_Gall30th_2016-01-14_deepspecz_photoz_1000_4_specweight.cat') * hmf[0].cosmo.h
    eff_density_in_mpc = eff_density / ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value / hmf[0].cosmo.h )**2.0
    
    shape_noise = (sigma_crit**2.0) * hmf[0].cosmo.H(z_kids).value * (kids_variance_squared / eff_density_in_mpc)/ (3.0*10.0**6.0)
    
    #radius = np.sqrt(kids_area/np.pi) * ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) / hmf[0].cosmo.h # conversion of area in deg^2 to Mpc/h!
    radius = np.sqrt(kids_area) * ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) / hmf[0].cosmo.h
    
    print(radius)
    print(eff_density_in_mpc)
    #ngal = 2.0*ngal
    
    print(shape_noise)
    print(1.0/ngal)
    #quit()


    print(rvir_range_2d_i.shape)

    cov_wp = np.empty((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size))
    cov_esd = np.empty((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size))
    cov_cross = np.empty((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size))

    #W = 2.0 * np.pi * radius**2.0 * sp.jv(1, k_range_lin*radius) / (k_range_lin*radius)
    #W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)
    #survey_var = [survey_variance(hmf_i, W_p, k_range, np.pi*radius**2.0*Pi_max) for hmf_i in hmf]
    
    W = 2.0*np.pi*radius**2.0 * sp.jv(1, k_range_lin*radius) / (k_range_lin*radius)
    W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)
    survey_var = [survey_variance(hmf_i, W_p, k_range, np.pi*radius**2.0*Pi_max) for hmf_i in hmf]
    
    #W = 500.0**3.0 * sp.jv(1, k_range_lin*500.0) / (k_range_lin*500.0)
    #W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)
    #survey_var = [survey_variance(hmf_i, W_p, k_range, 500.0**3.0) for hmf_i in hmf]
    
    
    # Test non-Gaussian
    
    print('Halo integrals done.')
    lnk_min, lnk_max = np.log(0.001), np.log(100.0)
    k_temp = np.linspace(lnk_min, lnk_max, 100, endpoint=True)
    k_temp_lin = np.exp(k_temp)
    #"""
    global Tgggg, Tgggm, Tgmgm, T234h
    
    Tgggg = _array([trispectra_1h(k_temp_lin, hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, k_range_lin, 'gggg')
                    for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                    _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
                    
    Tgggm = _array([trispectra_1h(k_temp_lin, hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, k_range_lin, 'gggm')
                    for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                    _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
                    
    Tgmgm = _array([trispectra_1h(k_temp_lin, hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i, mass_range, k_range_lin, 'gmgm')
                    for hmf_i, uk_i, rho_mean_i, ngal_i, pop_c_i, pop_s_i in
                    _izip(hmf, u_k, rho_mean, ngal, pop_c, pop_s)])
    
    T234h = _array([trispectra_234h(k_temp_lin, P_lin_inter_i, hmf_i, u_k_i, Bias_Tinker10(hmf_i, 0), rho_mean_i, mass_range, k_range_lin)
                    for P_lin_inter_i, hmf_i, u_k_i, rho_mean_i in
                    _izip(P_lin_inter, hmf, u_k, rho_mean)])
    #"""
    print('Trispectra done.')
    """
    shape_noise[0] = 0.0
    test_gauss = np.zeros((len(k_temp_lin), len(k_temp_lin)))
    delta = np.eye(len(k_temp_lin))
    for i, k in enumerate(k_temp_lin):
        for j, l in enumerate(k_temp_lin):
            test_gauss[i,j] = 2.0 * ((np.sqrt(np.exp(P_inter_3[0](np.log(k)))) * np.sqrt(np.exp(P_inter_3[0](np.log(l))))) + delta[i,j]*shape_noise[0])**2.0

    test_gauss = delta * test_gauss
    #test_gauss = 2.0 * np.outer((np.exp(P_inter_3[0](k_temp)) + shape_noise[0]), (np.exp(P_inter_3[0](k_temp)) + shape_noise[0]).T)
    
    ps_deriv_mm = ((68.0/21.0 - (1.0/3.0)*np.sqrt(dlnk3P_lin_interdlnk[0](k_temp))*np.sqrt(dlnk3P_lin_interdlnk[0](k_temp))) * np.sqrt(np.exp(P_lin_inter[0](k_temp)))*np.sqrt(np.exp(P_lin_inter[0](k_temp))) * np.exp(I_inter_m[0](k_temp))*np.exp(I_inter_m[0](k_temp)) + np.sqrt(np.exp(I_inter_mm[0](k_temp)))*np.sqrt(np.exp(I_inter_mm[0](k_temp))) )/ (np.exp(P_inter_3[0](k_temp)))
    
    test_1h = trispectra_1h(k_temp_lin, hmf[0], u_k[0], rho_mean[0], ngal[0], pop_c[0], pop_s[0], mass_range, k_range_lin, 'mmmm')
    test_1h = test_1h(k_temp_lin, k_temp_lin)
    import matplotlib.pyplot as pl
    #pl.imshow(test_1h, interpolation='nearest')
    #pl.show()
    #print(test_1h)

    test = trispectra_234h(k_temp_lin, P_lin_inter[0], hmf[0], u_k[0], Bias_Tinker10(hmf[0], 0), rho_mean[0], mass_range, k_range_lin)
    test = test(k_temp_lin, k_temp_lin)
    #test = test/100.0
    #test_block = test/np.sqrt(np.outer(np.diag(test), np.diag(test.T)))
    test_tot = test_1h + test

    pl.imshow(test, interpolation='nearest')
    pl.show()
    #print(test)

    #pl.plot(mass_range, hmf[0].dndm)
    #pl.plot(mass_range, hmf[1].dndm)
    #pl.plot(mass_range, hmf[2].dndm)
    #pl.xscale('log')
    #pl.yscale('log')
    #pl.show()
    
    #pl.plot(k_temp_lin, ps_deriv_mm)
    #pl.xscale('log')
    #pl.yscale('log')
    #pl.xlim([0.01, 2.5])
    #pl.ylim([0.15, 4])
    #pl.xlabel('k [h/Mpc]')
    #pl.ylabel(r'$\rm{d \ln} P(k) / \rm{d \delta_{b}}$')
    #pl.savefig('/home/dvornik/GalaxyBias_link/data/ssc_mm.png', bbox_inches='tight')
    #pl.show()


    volume = 500.0**3.0#np.pi*radius**2.0*Pi_max*2.0
    loc = 0.51
    index = np.argmin(np.abs(k_temp_lin - loc))
    print(k_temp_lin[index])
    print(index)
    st = (k_temp_lin[index+1]-k_temp_lin[index-1])/(k_temp_lin[index+1]+k_temp_lin[index-1])#/k_temp_lin[index]
    print(st)
    Nmode = 0.5*(k_temp_lin[index]**2.0 * volume * st) / (2.0*np.pi)**2.0
    
    denom = np.outer(np.exp(P_inter_3[0](k_temp)), np.exp(P_inter_3[0](k_temp)).T)
    
    pl.plot(k_temp_lin, np.sqrt(((test_gauss/Nmode) / denom))[:,index], color='red')
    pl.plot(k_temp_lin, np.sqrt((test_gauss/Nmode + test_tot/volume)/denom + (survey_var[0] * np.outer(ps_deriv_mm, ps_deriv_mm)))[:,index], color='black')
    pl.plot(k_temp_lin, np.sqrt((test/volume)/denom)[:,index], color='orange', ls='-.')
    pl.plot(k_temp_lin, np.sqrt((test_1h/volume)/denom)[:,index], color='orange', ls='--')
    pl.plot(k_temp_lin, np.sqrt((survey_var[0] * ps_deriv_mm * ps_deriv_mm)), color='blue')
    pl.xscale('log')
    pl.xlim([0.01, 1.0])
    pl.ylim([0.0, 0.08])
    #pl.yscale('log')
    pl.xlabel('k [h/Mpc]')
    pl.ylabel(r'$\rm{\sqrt{Cov/P(k)P(k\prime)}}$')
    pl.title(r'$k\prime = %f $'%loc)
    pl.savefig('/home/dvornik/GalaxyBias_link/data/tot_mm.png', bbox_inches='tight')
    pl.show()

    pl.plot(k_temp_lin, (k_temp_lin**3.0 / (2.0*np.pi)**2.0) * np.diag(test)**(1.0/3.0))
    pl.xscale('log')
    pl.xlim([0.01, 100.0])
    #pl.ylim([0.0, 0.08])
    pl.yscale('log')
    pl.show()
    
    quit()
    """
    #cov_wp_gauss, cov_esd_gauss, cov_cross_gauss = cov_wp.copy(), cov_esd.copy(), cov_cross.copy()
    #cov_wp_ssc, cov_esd_ssc, cov_cross_ssc = cov_wp.copy(), cov_esd.copy(), cov_cross.copy()
    cov_wp_gauss, cov_esd_gauss, cov_cross_gauss = cov_gauss(rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_wp.copy(), cov_esd.copy(), cov_cross.copy())
    cov_wp_ssc, cov_esd_ssc, cov_cross_ssc = cov_ssc(rvir_range_2d_i, P_lin_inter, dlnk3P_lin_interdlnk, P_inter, P_inter_2, I_inter_g, I_inter_m, I_inter_gg, I_inter_gm, W_p, Pi_max, bias_out, survey_var, cov_wp.copy(), cov_esd.copy(), cov_cross.copy())
    
    #cov_wp_non_gauss, cov_esd_non_gauss, cov_cross_non_gauss = cov_non_gauss(rvir_range_2d_i, bias_out, W_p, np.pi*radius**2.0*Pi_max, cov_wp.copy(), cov_esd.copy(), cov_cross.copy())
    cov_wp_non_gauss, cov_esd_non_gauss, cov_cross_non_gauss = cov_wp.copy(), cov_esd.copy(), cov_cross.copy()
    
    cov_wp_tot = cov_wp_gauss + cov_wp_ssc #+ cov_wp_non_gauss
    cov_esd_tot = cov_esd_gauss + cov_esd_ssc #+ cov_esd_non_gauss
    cov_cross_tot = cov_cross_gauss + cov_cross_ssc #+ cov_cross_non_gauss
    
    return cov_wp_tot, cov_esd_tot, cov_cross_tot, M_bin_min.size
    #return cov_wp_gauss, cov_esd_gauss, cov_cross_gauss, M_bin_min.size
    #return cov_wp_non_gauss, cov_esd_non_gauss, cov_cross_non_gauss, M_bin_min.size
    #return cov_wp_ssc, cov_esd_ssc, cov_cross_ssc, M_bin_min.size


if __name__ == '__main__':
    print(0)

























