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

from astropy.units import eV
import multiprocessing as multi
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import scipy
import sys
from numpy import (arange, array, exp, expand_dims, iterable, linspace,
                   logspace, ones)
from scipy import special as sp
from scipy.integrate import simps, trapz, quad
from scipy.interpolate import interp1d, interp2d, UnivariateSpline, \
    SmoothBivariateSpline, RectBivariateSpline, interpn, RegularGridInterpolator
from itertools import count
if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange
from time import time
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM

from hmf import MassFunction
from hmf import fitting_functions as ff
from hmf import transfer_models as tf

from . import baryons, longdouble_utils as ld, nfw
#from . import covariance
from . import profiles
from .tools import (
    fill_nan, load_hmf, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, sigma_crit,
    power_to_corr_ogata, wp, wp_beta_correction)
from .dark_matter import (
    mm_analy, gm_cen_analy, gm_sat_analy, gg_cen_analy,
    gg_sat_analy, gg_cen_sat_analy, two_halo_gm, two_halo_gg)
from .. import hod


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


def survey_variance(mass_func, W_p, k_range, radius):
    
    # Seems to be about right! To be checked again!.
    
    P_lin = mass_func.power
    
    #v_w = radius
    #integ2 = W_p(np.exp(k_range))**2.0 * np.exp(k_range)**2.0 * P_lin
    #sigma = (1.0 / (2.0*np.pi**2.0 * v_w**2.0)) * simps(integ2, np.exp(k_range))
    
    
    Awr = simps(np.exp(k_range) * sp.jv(0, np.exp(k_range) * radius)/(2.0*np.pi) * W_p(np.exp(k_range))**2.0, k_range)
    integ2 = W_p(np.exp(k_range))**2.0 * np.exp(k_range)**2.0 * P_lin
    sigma = (1.0 / (2.0*np.pi**2.0 * Awr)) * simps(integ2, np.exp(k_range))
    
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


def cov_non_gauss(rvir_range_2d_i, b_g, W_p, volume, cov_wp, cov_esd, cov_cross, nproc):
    
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
    pool = multi.Pool(processes=nproc)
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
    ps_deriv_gm_j = (68.0/21.0 - (1.0/3.0)*(dP_lin_j)) * np.exp(P_lin_j) * np.exp(Ig_j)*np.exp(Ig_j) + np.exp(Igm_j) - b_g[b_j] * np.exp(P_gm_j)
    
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


def cov_ssc(rvir_range_2d_i, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var, cov_wp, cov_esd, cov_cross, nproc):
    
    print('Calculating the super-sample covariance ...')
    
    b_i = xrange(len(Pgm))
    b_j = xrange(len(Pgm))
    i = xrange(len(rvir_range_2d_i))
    j = xrange(len(rvir_range_2d_i))
    
    paramlist = [list(tup) for tup in itertools.product(b_i,b_j,i,j)]
    for i in paramlist:
        i.extend([rvir_range_2d_i, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var])

    pool = multi.Pool(processes=nproc)
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
    integ3 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j]) * (np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j]) + np.exp(P_gm_i)*np.exp(P_gm_j) )
    integ4 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * (np.sqrt(np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j])*np.sqrt(np.exp(P_mm_j) + shape_noise[b_j]* delta[b_i, b_j]))
    b = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ3, dx=dlnk) + ((2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ4, dx=dlnk)
                    
    # cross
    integ5 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.exp(P_gm_i)* (np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j])) + (np.exp(P_gm_j)* (np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j])) )
    integ6 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j]))
    c = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk) + ((4.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk)
    
    return b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j, [a, b, c]


def cov_gauss(rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_wp, cov_esd, cov_cross, nproc):

    print('Calculating the Gaussian part of the covariance ...')

    b_i = xrange(len(P_inter_2))
    b_j = xrange(len(P_inter_2))
    i = xrange(len(rvir_range_2d_i))
    j = xrange(len(rvir_range_2d_i))
    
    paramlist = [list(tup) for tup in itertools.product(b_i,b_j,i,j)]
    for i in paramlist:
        i.extend([rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal])
    
    pool = multi.Pool(processes=nproc)
    for i, j, val in pool.map(calc_cov_gauss, paramlist):
        #print(i, j, val)
        cov_wp[i,j] = val[0]
        cov_esd[i,j] = val[1]
        cov_cross[i,j] = val[2]

    return cov_wp, cov_esd, cov_cross


def covariance(theta, R, calculate_covariance=False):
    np.seterr(
        divide='ignore', over='ignore', under='ignore', invalid='ignore')

    # this has to happen before because theta is re-purposed below
    if calculate_covariance:
        covar = theta[1][theta[0].index('covariance')]

    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
            for name in ('observables', 'selection', 'ingredients',
                    'parameters', 'setup')]

    assert len(observables) == 1, \
        'working with more than one observable is not yet supported.' \
        ' If you would like this feature added please raise an issue.'
    observable = observables[0]
    hod_observable = observable.sampling

    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta

    sigma8, h, omegam, omegab, n, w0, wa, Neff, z = cosmo[:9]

    

    if ingredients['nzlens']:
        nz = cosmo[9].T
        size_cosmo = 10
    else:
        size_cosmo = 9
    # cheap hack. I'll use this for CMB lensing, but we can
    # also use this to account for difference between shear
    # and reduced shear
    if len(cosmo) == size_cosmo+1:
        zs = cosmo[-1]
    elif setup['return'] == 'kappa':
        raise ValueError(
            'If return=kappa then you must provide a source redshift as' \
            ' the last cosmological parameter. Alternatively, make sure' \
            ' that the redshift parameters are properly set given your' \
            ' choice for the zlens parameter')

    integrate_zlens = ingredients['nzlens']

    # HMF set up parameters
    k_step = (setup['lnk_max']-setup['lnk_min']) / setup['lnk_bins']
    k_range = arange(setup['lnk_min'], setup['lnk_max'], k_step)
    k_range_lin = exp(k_range)
    # endpoint must be False for mass_range to be equal to hmf.m
    mass_range = 10**linspace(
        setup['logM_min'], setup['logM_max'], setup['logM_bins'],
        endpoint=False)
    setup['mstep'] = (setup['logM_max']-setup['logM_min']) \
                    / setup['logM_bins']

    nbins = observable.nbins
    # if a single value is given for more than one bin, assign same
    # value to all bins
    if not np.iterable(z):
        z = np.array([z])
    if z.size == 1 and nbins > 1:
        z = array(list(z)*nbins)
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    z = expand_dims(z, -1)

    cosmo_model = Flatw0waCDM(
        H0=100*h, Ob0=omegab, Om0=omegam, Tcmb0=2.725, m_nu=0.06*eV,
        Neff=Neff, w0=w0, wa=wa)

    # Tinker10 should also be read from theta!
    transfer_params = \
        {'sigma_8': sigma8, 'n': n, 'lnk_min': setup['lnk_min'],
        'lnk_max': setup['lnk_max'], 'dlnk': k_step}
    hmf, rho_mean = load_hmf(z, setup, cosmo_model, transfer_params)

    mass_range = hmf[0].m
    rho_bg = rho_mean if setup['delta_ref'] == 'mean' \
        else rho_mean / omegam
    # same as with redshift
    rho_bg = expand_dims(rho_bg, -1)

    concentration = c_concentration[0](mass_range, *c_concentration[1:])
    if ingredients['satellites']:
        concentration_sat = s_concentration[0](
            mass_range, *s_concentration[1:])

    rvir_range_lin = virial_radius(
        mass_range, rho_bg, setup['delta'])
    rvir_range_3d = logspace(-3.2, 4, 250, endpoint=True)
    # these are not very informative names but the "i" stands for
    # interpolated
    rvir_range_3d_i = logspace(-2.5, 1.2, 25, endpoint=True)
    #rvir_range_3d_i = logspace(-4, 2, 60, endpoint=True)
    # integrate over redshift later on
    # assuming this means arcmin for now -- implement a way to check later!
    #if setup['distances'] == 'angular':
        #R = R * cosmo.
    rvir_range_2d_i = R[0][1:]

    # Calculating halo model
    
    # interpolate selection function to the same grid as redshift and
    # observable to be used in trapz
    
    completeness = np.ones(hod_observable.shape)
    if ingredients['centrals']:
        pop_c, prob_c = hod.number(
            hod_observable, mass_range, c_mor[0], c_scatter[0],
            c_mor[1:], c_scatter[1:], completeness,
            obs_is_log=observable.is_log)
    else:
        pop_c = np.zeros((nbins,mass_range.size))
        prob_c = np.zeros((nbins,hod_observable.shape[1],mass_range.size))

    if ingredients['satellites']:
        pop_s, prob_s = hod.number(
            hod_observable, mass_range, s_mor[0], s_scatter[0],
            s_mor[1:], s_scatter[1:], completeness,
            obs_is_log=observable.is_log)
    else:
        pop_s = np.zeros(pop_c.shape)
        prob_s = np.zeros(prob_c.shape)

    pop_g = pop_c + pop_s
    prob_g = prob_c + prob_s

    # note that pop_g already accounts for incompleteness
    dndm = array([hmf_i.dndm for hmf_i in hmf])
    ngal = hod.nbar(dndm, pop_g, mass_range)
    meff = hod.Mh_effective(
        dndm, pop_g, mass_range, return_log=observable.is_log)
                   
    """
    # Power spectrum
    """
                   
                   
    # damping of the 1h power spectra at small k
    F_k1 = sp.erf(k_range_lin/0.1)
    F_k2 = sp.erfc(k_range_lin/1500.0)
    #F_k1 = np.ones_like(k_range_lin)
    # Fourier Transform of the NFW profile
    
    if ingredients['centrals']:
        uk_c = nfw.uk(
            k_range_lin, mass_range, rvir_range_lin, concentration, rho_bg,
            setup['delta'])
        uk_c = uk_c / expand_dims(uk_c[...,0], -1)
    else:
        uk_c = np.ones((nbins,mass_range.size,k_range_lin.size))
    # and of the NFW profile of the satellites
    if ingredients['satellites']:
        uk_s = nfw.uk(
            k_range_lin, mass_range, rvir_range_lin, concentration_sat, rho_bg,
            setup['delta'])
        uk_s = uk_s / expand_dims(uk_s[...,0], -1)
    else:
        uk_s = np.ones((nbins,mass_range.size,k_range_lin.size))

    
    
    # Galaxy - dark matter spectra (for lensing)
    bias = c_twohalo
    bias = array([bias]*k_range_lin.size).T
    
   
    rho_bg = rho_bg[...,0]
    
    Pgm_2h = F_k2 * bias * array(
            [two_halo_gm(hmf_i, ngal_i, pop_g_i,
                    rvir_range_lin_i, mass_range)[0]
            for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
            in zip(rvir_range_lin, hmf, expand_dims(ngal, -1),
                    expand_dims(pop_g, -2))])

    if ingredients['centrals']:
        Pgm_c = F_k1 * gm_cen_analy(
            dndm, uk_c, rho_bg, pop_c, ngal, mass_range)
    else:
        Pgm_c = F_k1 * np.zeros((nbins,setup['lnk_bins']))

    if ingredients['satellites']:
        Pgm_s = F_k1 * gm_sat_analy(
            dndm, uk_c, uk_s, rho_bg, pop_s, ngal, mass_range)
    else:
        Pgm_s = F_k1 * np.zeros(Pgm_c.shape)

    Pgm_k = Pgm_c + Pgm_s + Pgm_2h
    
    
    
    # Galaxy - galaxy spectra (for clustering)
    
    Pgg_2h = F_k2 * bias * array(
            [two_halo_gg(hmf_i, ngal_i, pop_g_i,
                    rvir_range_lin_i, mass_range)[0]
            for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
            in zip(rvir_range_lin, hmf, expand_dims(ngal, -1),
                    expand_dims(pop_g, -2))])
        
    ncen = hod.nbar(dndm, pop_c, mass_range)
    nsat = hod.nbar(dndm, pop_s, mass_range)

    if ingredients['centrals']:
        """
        Pgg_c = F_k1 * gg_cen_analy(dndm, ncen, ngal, (nbins,setup['lnk_bins']), mass_range)
        """
        Pgg_c = F_k1 * np.zeros((nbins,setup['lnk_bins']))
    else:
        Pgg_c = F_k1 * np.zeros((nbins,setup['lnk_bins']))
    
    if ingredients['satellites']:
        beta = s_beta
        Pgg_s = F_k1 * gg_sat_analy(dndm, uk_s, pop_s, ngal, beta, mass_range)
    else:
        Pgg_s = F_k1 * np.zeros(Pgg_c.shape)
        
    if ingredients['centrals'] and ingredients['satellites']:
        Pgg_cs = F_k1 * gg_cen_sat_analy(dndm, uk_s, pop_c, pop_s, ngal, mass_range)
    else:
        Pgg_cs = F_k1 * np.zeros(Pgg_c.shape)
        
    Pgg_k = Pgg_c + (2.0 * Pgg_cs) + Pgg_s + Pgg_2h
                            
                            
    # Matter - matter spectra
   
    if ingredients['centrals']:
        Pmm_1h = F_k1 * mm_analy(dndm, uk_c, rho_bg, mass_range)
    else:
        Pmm_1h = F_k1 * np.zeros((nbins,setup['lnk_bins']))
                            
    Pmm_k = Pmm_1h + F_k2 * array([hmf_i.power for hmf_i in hmf])



    P_inter = [UnivariateSpline(k_range, np.log(Pgm_k_i), s=0, ext=0)
                    for Pgm_k_i in Pgm_k]
        
    P_inter_2 = [UnivariateSpline(k_range, np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in Pgg_k]

    P_inter_3 = [UnivariateSpline(k_range, np.log(Pmm_k_i), s=0, ext=0)
                    for Pmm_k_i in Pmm_k]
                    
    #### TO-DO!
    #### Stuff below needs updating!
            
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

    #Pi_max = 100.0
    
    #kids_area = 180 * 3600.0 #500 #To be in arminutes!
    #eff_density = 8.53#6.0 #1.2#1.4#2.34#8.53 #1.5#1.85
    
    #kids_variance_squared = 0.082#0.076 #0.275#0.076
    #z_kids = 0.6
    
    #sigma_crit_old = sigma_crit_kids(hmf, z, 0.2, 0.9, spec_z_path) * hmf[0].cosmo.h
    sigma_crit = sigma_crit_kids(hmf, z, 0.2, 0.9, spec_z_path) * hmf[0].cosmo.h * 10.0**12.0 / (1.0+z)**2.0
    print(sigma_crit/10.0**12.0)
    
    
    eff_density_in_mpc = eff_density  / ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value / hmf[0].cosmo.h )**2.0
    #shape_noise_old = ((sigma_crit_old)**2.0) * hmf[0].cosmo.H(z_kids).value * (kids_variance_squared / eff_density_in_mpc)/ (3.0*10.0**6.0)
    #print(shape_noise_old)
    
    #eff_density_in_mpc = eff_density
    #shape_noise = np.zeros(sigma_crit.shape)
    #shape_noise = ((sigma_crit / rho_mean[0])**2.0) * (kids_variance_squared / eff_density_in_mpc) / (Pi_max)**2.0# * ((hmf[0].cosmo.angular_diameter_distance(z).value)**2.0 / (2.0 * Pi_max))
    eff_density_in_rad = eff_density * (10800.0/np.pi)**2.0
    shape_noise = ((sigma_crit / rho_mean[0])**2.0) * (kids_variance_squared / eff_density_in_rad)  * ((hmf[0].cosmo.angular_diameter_distance(z).value)**2.0 / hmf[0].cosmo.angular_diameter_distance_z1z2(z,z_kids).value)#(8.0 * Pi_max))
    
    
    #radius = np.sqrt(kids_area/np.pi) * ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) / hmf[0].cosmo.h # conversion of area in deg^2 to Mpc/h!
    radius = np.sqrt(kids_area) * ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) / hmf[0].cosmo.h
    
    print(radius)
    print(eff_density_in_mpc)
    #ngal = 2.0*ngal
    
    print(shape_noise * rho_mean[0]**2.0 / 10.0**24.0)
    print(1.0/ngal)
    #quit()


    print(rvir_range_2d_i.shape)

    cov_wp = np.zeros((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size), dtype=np.float64)
    cov_esd = np.zeros((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size), dtype=np.float64)
    cov_cross = np.zeros((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size), dtype=np.float64)

    #W = 2.0 * np.pi * radius**2.0 * sp.jv(1, k_range_lin*radius) / (k_range_lin*radius)
    #W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)
    #survey_var = [survey_variance(hmf_i, W_p, k_range, np.pi*radius**2.0*Pi_max) for hmf_i in hmf]
    
    W = 2.0*np.pi*radius**2.0 * sp.jv(1, k_range_lin*radius) / (k_range_lin*radius)
    W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)
    #survey_var = [survey_variance(hmf_i, W_p, k_range, np.pi*radius**2.0*Pi_max) for hmf_i in hmf]
    survey_var = [survey_variance(hmf_i, W_p, k_range, radius) for hmf_i in hmf]
    
    
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
    cov_wp_tot = cov_wp.copy()
    cov_esd_tot = cov_esd.copy()
    cov_cross_tot = cov_cross.copy()
    
    if gauss == True:
        cov_wp_gauss, cov_esd_gauss, cov_cross_gauss = cov_gauss(rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_wp.copy(), cov_esd.copy(), cov_cross.copy(), nproc)
        cov_wp_tot += cov_wp_gauss
        cov_esd_tot += cov_esd_gauss
        cov_cross_tot += cov_cross_gauss
    else:
        pass#cov_wp_gauss, cov_esd_gauss, cov_cross_gauss = cov_wp.copy(), cov_esd.copy(), cov_cross.copy()

    if ssc == True:
        cov_wp_ssc, cov_esd_ssc, cov_cross_ssc = cov_ssc(rvir_range_2d_i, P_lin_inter, dlnk3P_lin_interdlnk, P_inter, P_inter_2, I_inter_g, I_inter_m, I_inter_gg, I_inter_gm, W_p, Pi_max, bias_out, survey_var, cov_wp.copy(), cov_esd.copy(), cov_cross.copy(), nproc)
        cov_wp_tot += cov_wp_ssc
        cov_esd_tot += cov_esd_ssc
        cov_cross_tot += cov_cross_ssc
    else:
        pass#cov_wp_ssc, cov_esd_ssc, cov_cross_ssc = cov_wp.copy(), cov_esd.copy(), cov_cross.copy()

    if ng == True:
        cov_wp_non_gauss, cov_esd_non_gauss, cov_cross_non_gauss = cov_non_gauss(rvir_range_2d_i, bias_out, W_p, np.pi*radius**2.0*Pi_max, cov_wp.copy(), cov_esd.copy(), cov_cross.copy(), nproc)
        cov_wp_tot += cov_wp_non_gauss
        cov_esd_tot += cov_esd_non_gauss
        cov_cross_tot += cov_cross_non_gauss
    else:
        pass#cov_wp_non_gauss, cov_esd_non_gauss, cov_cross_non_gauss = cov_wp.copy(), cov_esd.copy(), cov_cross.copy()
    
    #cov_wp_tot = cov_wp_gauss + cov_wp_ssc + cov_wp_non_gauss
    #cov_esd_tot = cov_esd_gauss + cov_esd_ssc + cov_esd_non_gauss
    #cov_cross_tot = cov_cross_gauss + cov_cross_ssc + cov_cross_non_gauss
    
    return cov_wp_tot, (cov_esd_tot * rho_mean[0]**2.0) / 10.0**24.0, (cov_cross_tot * rho_mean[0]) / 10.0**12.0, M_bin_min.size
    #return cov_wp_gauss, cov_esd_gauss, cov_cross_gauss, M_bin_min.size
    #return cov_wp_non_gauss, cov_esd_non_gauss, cov_cross_non_gauss, M_bin_min.size
    #return cov_wp_ssc, cov_esd_ssc, cov_cross_ssc, M_bin_min.size


if __name__ == '__main__':
    print(0)

























