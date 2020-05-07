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
                   logspace, ones, zeros)
from scipy import special as sp
from scipy.integrate import simps, trapz, quad
from scipy.interpolate import interp1d, interp2d, UnivariateSpline, \
    SmoothBivariateSpline, RectBivariateSpline, interpn, RegularGridInterpolator
from itertools import count, product
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
    power_to_corr, power_to_corr_multi, sigma, d_sigma, sigma_crit as sigma_crit_func,
    power_to_corr_ogata, wp, wp_beta_correction)
from .dark_matter import (
    bias_tinker10, mm_analy, gm_cen_analy, gm_sat_analy, gg_cen_analy,
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
            
    for i in range(lens_redshifts.size):
        srcmask *= (lens_redshifts[i]+z_epsilon <= spec_cat['Z_B']) & (spec_cat['Z_B'] < srclim)
        srcNZ_k, spec_weight_k = Z_S[srcmask], spec_weight[srcmask]
                    
        srcPZ_k, bins_k = np.histogram(srcNZ_k, range=[0.025, 3.5], bins=70, weights=spec_weight_k, density=1)
        srcPZ_k = srcPZ_k/srcPZ_k.sum()
        k[i], kmask = calc_Sigmacrit(np.array([lens_comoving[i]]), np.array([lens_angular[i]]), Dcsbins, srcPZ_k, 3, Dc_epsilon)
            
    k_interpolated = interp1d(lens_redshifts, k, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0))

    return 1.0/k_interpolated(z_in)


def survey_variance_test(mass_func, W_p, k_range, radius):

    # Seems to be about right! To be checked again!.
    P_lin = mass_func.power

    # This works when testing against Mohammed et al. 2017. Survey variance for a simulation cube!
    v_w = radius
    integ2 = W_p(np.exp(k_range))**2.0 * np.exp(k_range)**2.0 * P_lin
    sigma = (1.0 / (2.0*np.pi**2.0 * v_w**2.0)) * simps(integ2, np.exp(k_range))

    return sigma


def survey_variance(mass_func, W_p, k_range, radius):
    
    # Seems to be about right! To be checked again!.
    
    P_lin = mass_func.power

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
        
        idx = np.where((1.0-mu) < trispec_matter_mulim)
        k_m[idx] = 0.0
        mu_1m[idx] = 0.0
        alpha_m[idx] = 1.0
        beta_m[idx] = 0.0
        

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
        
        idx = np.where((mu+1.0) < trispec_matter_mulim)
        k_p[idx] = 0.0
        mu_1p[idx] = 0.0
        alpha_p[idx] = 1.0
        beta_p[idx] = 0.0
        
            
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
        
        idx = np.where((1.0-mu) < trispec_matter_mulim)
        k_m[idx] = 0.0
        mu_1m[idx] = 0.0
        mu_2m[idx] = 0.0
        p_m[idx] = 0.0
        F2_1m[idx] = 0.0
        F2_2m[idx] = 0.0
        

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
        
        idx = np.where((mu+1.0) < trispec_matter_mulim)
        k_p[idx] = 0.0
        mu_1p[idx] = 0.0
        mu_2p[idx] = 0.0
        p_p[idx] = 0.0
        F2_1p[idx] = 0.0
        F2_2p[idx] = 0.0
        

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
        
        idx = np.where((mu+1.0) < trispec_matter_mulim)
        k_p[idx] = 0.0
        term2[idx] = 0.0
        term3[idx] = 0.0
        
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
        idx = np.where((1.0-mu) < trispec_matter_mulim)
        k_m[idx] = 0.0
        p_m[idx] = 0.0

    if np.fabs(k1-k2) < trispec_matter_klim:
        k_p = np.zeros(mu.shape)
        p_p = np.zeros(mu.shape)
    else:
        k_p = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu)
        p_p = np.exp(P_lin_inter(np.log(k_p)))
        idx  = np.where((mu+1.0) < trispec_matter_mulim)
        k_p[idx] = 0.0
        p_p[idx] = 0.0

    return p_p + p_m


def intg_for_trispec_matter_parallel_3h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    return bispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim) + bispec_parallel_pt(k1, k2, (-1.0)*mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim)


def intg_for_trispec_matter_parallel_4h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    return trispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim)


def trispectra_234h(krange, P_lin_inter, mass_func, uk, bias, rho_mean, m_x, k_x):
    
    trispec_matter_mulim = 0.001
    trispec_matter_klim = 100.0#0.001*3000.0#100.0#0.001 ## units of k in Benjamin's code are c/H0, this is h/Mpc!
    
    trispec_2h = np.zeros((krange.size, krange.size))
    trispec_3h = np.zeros((krange.size, krange.size))
    trispec_4h = np.zeros((krange.size, krange.size))
    
    # Evaluate u(k) on different k grid!
    u_k = np.array([UnivariateSpline(k_x, uk[m,:], s=0, ext=0) for m in range(len(m_x))])
    u_k_new = np.array([u_k[m](krange) for m in range(len(m_x))])
    
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
    
    x = np.linspace(0.0, 2.0*np.pi, num=100, endpoint=True)
    
    for i, k1 in enumerate(krange):
        for j, k2 in enumerate(krange):
            trispec_2h[i,j] = 2.0 * Immm(i, j, j, mass_func, u_k_new, bias, rho_mean, m_x) * Im(i, mass_func, u_k_new, bias, rho_mean, m_x) * np.exp(P_lin_inter(np.log(k1))) + 2.0 * 2.0 * Immm(i, i, j, mass_func, u_k_new, bias, rho_mean, m_x) * Im(j, mass_func, u_k_new, bias, rho_mean, m_x) * np.exp(P_lin_inter(np.log(k2))) + (Imm(i, j, mass_func, u_k_new, bias, rho_mean, m_x)**2.0) * trapz(intg_for_trispec_matter_parallel_2h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)
            trispec_3h[i,j] = 2.0 * Imm(i, j, mass_func, u_k_new, bias, rho_mean, m_x) * Im(i, mass_func, u_k_new, bias, rho_mean, m_x) * Im(j, mass_func, u_k_new, bias, rho_mean, m_x) * trapz(intg_for_trispec_matter_parallel_3h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)
            trispec_4h[i,j] = (Im(i, mass_func, u_k_new, bias, rho_mean, m_x))**2.0 * (Im(j, mass_func, u_k_new, bias, rho_mean, m_x))**2.0 * trapz(intg_for_trispec_matter_parallel_4h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), x)/(2.0*np.pi)

    trispec_2h = np.nan_to_num(trispec_2h)
    trispec_3h = np.nan_to_num(trispec_3h)
    trispec_4h = np.nan_to_num(trispec_4h)

    trispec_tot = trispec_2h + trispec_3h + trispec_4h
    trispec_tot_interp = RectBivariateSpline(krange, krange, trispec_tot, kx=1, ky=1)
    
    #trispec_2h_interp = RectBivariateSpline(krange, krange, trispec_2h, kx=1, ky=1)
    #trispec_3h_interp = RectBivariateSpline(krange, krange, trispec_3h, kx=1, ky=1)
    #trispec_4h_interp = RectBivariateSpline(krange, krange, trispec_4h, kx=1, ky=1)
    
    return trispec_tot_interp
    #return trispec_tot_interp, trispec_2h_interp, trispec_3h_interp, trispec_4h_interp

def trispectra_1h(krange, mass_func, uk_c, uk_s, rho_mean, ngal, population_cen, population_sat, m_x, k_x, x):
    
    trispec_1h = np.zeros((krange.size, krange.size))

    u_g_prod = (expand_dims(population_cen, -1) + expand_dims(population_sat, -1) * uk_s)
    u_m_prod = expand_dims(m_x, -1) * uk_c
    norm_g = ngal
    norm_m = rho_mean
    
    # Evaluate u(k) on different k grid!
    u_g = np.array([UnivariateSpline(k_x, u_g_prod[m,:], s=0, ext=0) for m in range(len(m_x))])
    u_m = np.array([UnivariateSpline(k_x, u_m_prod[m,:], s=0, ext=0) for m in range(len(m_x))])
    
    u_m_new = np.array([u_m[m](krange) for m in range(len(m_x))])
    u_g_new = np.array([u_g[m](krange) for m in range(len(m_x))])
    
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


def halo_model_integrals(dndm, uk_c, uk_s, bias, rho_mean, ngal, population_cen, population_sat, Mh, x):
    
    if x == 'g':
        integ1 = expand_dims(dndm * bias, -1) * (expand_dims(population_cen, -1) + expand_dims(population_sat, -1) * uk_s)
        I = trapz(integ1, Mh, axis=0)/ngal
    
    if x == 'm':
        rho_mean = expand_dims(rho_mean, -1)
        integ2 = expand_dims(dndm * bias * Mh, -1) * uk_c
        I = trapz(integ2, Mh, axis=0)/rho_mean

    if x == 'gg':
        integ3 = expand_dims(dndm * bias, -1) * (expand_dims(population_cen * population_sat, -1) * uk_s + expand_dims(population_sat**2.0, -1) * uk_s**2.0)
        I = trapz(integ3, Mh, axis=0)/(ngal**2.0)

    if x == 'gm':
        rho_mean = expand_dims(rho_mean, -1)
        integ4 = expand_dims(dndm * bias * Mh, -1) * uk_c * (expand_dims(population_cen, -1) + expand_dims(population_sat, -1) * uk_s)
        I = trapz(integ4, Mh, axis=0)/(rho_mean*ngal)

    if x == 'mm':
        rho_mean = expand_dims(rho_mean, -1)
        integ5 = expand_dims(dndm * bias * Mh**2.0, -1) * uk_c**2.0
        I = trapz(integ5, Mh, axis=0)/(rho_mean**2.0)

    if x == 'mmm':
        rho_mean = expand_dims(rho_mean, -1)
        integ6 = expand_dims(dndm * bias * Mh**3.0, -1) * uk_c**3.0
        I = trapz(integ6, Mh, axis=0)/(rho_mean**3.0)

    return I


def calc_cov_non_gauss(params):
    
    b_i, b_j, i, j, radius_1, radius_2, T1h, T234h, b_g, W_p, volume, rho_bg, ingredient, idx_1, idx_2, size_1, size_2 = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    bg_i = b_g[idx_1][b_i][0]
    bg_j = b_g[idx_2][b_j][0]
    rho_i = rho_bg[idx_1]
    rho_j = rho_bg[idx_2]
    
    #delta = np.eye(b_g.size)
    
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
    if nk > 1000:
        nk = 1000
    
    lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
    
    Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Awr_j = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_j)/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    Aw_rr = simps(np.exp(lnk)**2.0 * (sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j))/(2.0*np.pi) * W_p(np.exp(lnk))**2.0, dx=dlnk)
    
    T234_i = T234h[idx_1][b_i](np.exp(lnk), np.exp(lnk))
    T234_j = T234h[idx_2][b_j](np.exp(lnk), np.exp(lnk))
    
    T_i = T1h[idx_1][b_i](np.exp(lnk), np.exp(lnk))
    T_j = T1h[idx_2][b_j](np.exp(lnk), np.exp(lnk))
    
    if ingredient == 'gg':
        integ1 = np.outer(np.exp(lnk)**(1.0) * sp.jv(0, np.exp(lnk) * r_i), np.exp(lnk)**(1.0) * sp.jv(0, np.exp(lnk) * r_j)) * (np.sqrt(T_i * T_j.T) + bg_i*bg_i*bg_j*bg_j*np.sqrt(T234_i * T234_j.T))
        I_wp = trapz(trapz(integ1, dx=dlnk, axis=0), dx=dlnk)/volume
        val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * I_wp
        
    if ingredient == 'gm':
        integ2 = np.outer(np.exp(lnk)**(1.0) * sp.jv(2, np.exp(lnk) * r_i), np.exp(lnk)**(1.0) * sp.jv(2, np.exp(lnk) * r_j)) * (np.sqrt(T_i * T_j.T) + bg_i*bg_j*np.sqrt(T234_i * T234_j.T))
        I_esd = trapz(trapz(integ2, dx=dlnk, axis=0), dx=dlnk)/volume
        val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * I_esd * rho_i[b_i]*rho_j[b_j] / 1e24

    if ingredient == 'cross':
        integ3 = np.outer(np.exp(lnk)**(1.0) * sp.jv(0, np.exp(lnk) * r_i), np.exp(lnk)**(1.0) * sp.jv(2, np.exp(lnk) * r_j)) * (np.sqrt(T_i * T_j.T) + bg_i*bg_j*np.sqrt(bg_i*bg_j)*np.sqrt(T234_i * T234_j.T))
        I_cross = trapz(trapz(integ3, dx=dlnk, axis=0), dx=dlnk)/volume
        val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * I_cross * np.sqrt(rho_i[b_i]*rho_j[b_j]) / 1e12

    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val


def cov_non_gauss(radius_1, radius_2, T1h, T234h, b_g, W_p, volume, cov_out, nproc, rho_bg, ingredient, idx_1, idx_2, size_1, size_2):
    
    paramlist = []
    for a in range(len(radius_1)):
        for b in range(len(radius_2)):
            for c in range(len(radius_1[a])):
                for d in range(len(radius_2[b])):
                    paramlist.append([a, b, c, d])
                    
    for i in paramlist:
        i.extend([radius_1, radius_2, T1h, T234h, b_g, W_p, volume, rho_bg, ingredient, idx_1, idx_2, size_1, size_2])
    
    pool = multi.Pool(processes=nproc)
    for i, j, val in pool.imap(calc_cov_non_gauss, paramlist):
        #print(i, j, val)
        cov_out[i,j] = val

    
    return cov_out


def calc_cov_ssc(params):
    
    b_i, b_j, i, j, radius_1, radius_2, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var, rho_bg, ingredient, idx_1, idx_2, size_1, size_2 = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    bg_i = b_g[idx_1][b_i][0]
    bg_j = b_g[idx_2][b_j][0]
    rho_i = rho_bg[idx_1]
    rho_j = rho_bg[idx_2]
    survey_var_i = survey_var[idx_1]
    survey_var_j = survey_var[idx_2]
    
    #delta = np.eye(b_g.size)
    
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
    
    P_gm_i = Pgm[idx_1][b_i](lnk)
    P_gm_j = Pgm[idx_2][b_j](lnk)
    
    P_gg_i = Pgg[idx_1][b_i](lnk)
    P_gg_j = Pgg[idx_2][b_j](lnk)
    
    P_lin_i = P_lin[idx_1][b_i](lnk)
    P_lin_j = P_lin[idx_2][b_j](lnk)
    
    dP_lin_i = dlnP_lin[idx_1][b_i](lnk)
    dP_lin_j = dlnP_lin[idx_2][b_j](lnk)
    
    Ig_i = I_g[idx_1][b_i](lnk)
    Ig_j = I_g[idx_2][b_j](lnk)
    
    Im_i = I_m[idx_1][b_i](lnk)
    Im_j = I_m[idx_2][b_j](lnk)
    
    Igg_i = I_gg[idx_1][b_i](lnk)
    Igg_j = I_gg[idx_2][b_j](lnk)
    
    Igm_i = I_gm[idx_1][b_i](lnk)
    Igm_j = I_gm[idx_2][b_j](lnk)
    
    # Responses
    ps_deriv_gg_i = (68.0/21.0 - (1.0/3.0)*(dP_lin_i)) * np.exp(P_lin_i) * np.exp(Ig_i)*np.exp(Ig_i) + np.exp(Igg_i) - 2.0 * bg_i * np.exp(P_gg_i)
    ps_deriv_gg_j = (68.0/21.0 - (1.0/3.0)*(dP_lin_j)) * np.exp(P_lin_j) * np.exp(Ig_j)*np.exp(Ig_j) + np.exp(Igg_j) - 2.0 * bg_j * np.exp(P_gg_j)
    
    ps_deriv_gm_i = (68.0/21.0 - (1.0/3.0)*(dP_lin_i)) * np.exp(P_lin_i) * np.exp(Ig_i)*np.exp(Im_i) + np.exp(Igm_i) - bg_i * np.exp(P_gm_i)
    ps_deriv_gm_j = (68.0/21.0 - (1.0/3.0)*(dP_lin_j)) * np.exp(P_lin_j) * np.exp(Ig_j)*np.exp(Im_j) + np.exp(Igm_j) - bg_j * np.exp(P_gm_j)
    
    # wp
    if ingredient == 'gg':
        integ1 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * (np.sqrt(survey_var_i[b_i])*np.sqrt(survey_var_j[b_j])) * ps_deriv_gg_i * ps_deriv_gg_j
        val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * trapz(integ1, dx=dlnk)
    
    # ESD
    if ingredient == 'gm':
        integ2 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * (np.sqrt(survey_var_i[b_i])*np.sqrt(survey_var_j[b_j])) * ps_deriv_gm_i * ps_deriv_gm_j
        val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * trapz(integ2, dx=dlnk) * rho_i[b_i]*rho_j[b_j] / 1e24
    
    # cross
    if ingredient == 'cross':
        integ3 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * (np.sqrt(survey_var_i[b_i])*np.sqrt(survey_var_j[b_j])) * np.sqrt(np.abs(ps_deriv_gg_i*ps_deriv_gg_j * ps_deriv_gm_i*ps_deriv_gm_j)) * np.sign(ps_deriv_gg_i*ps_deriv_gg_j * ps_deriv_gm_i*ps_deriv_gm_j)
        val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * trapz(integ3, dx=dlnk) * np.sqrt(rho_i[b_i]*rho_j[b_j]) / 1e12


    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val


def cov_ssc(radius_1, radius_2, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var, cov_out, nproc, rho_bg, ingredient, idx_1, idx_2, size_1, size_2):
    
    paramlist = []
    for a in range(len(radius_1)):
        for b in range(len(radius_2)):
            for c in range(len(radius_1[a])):
                for d in range(len(radius_2[b])):
                    paramlist.append([a, b, c, d])
                    
    for i in paramlist:
        i.extend([radius_1, radius_2, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, W_p, Pi_max, b_g, survey_var, rho_bg, ingredient, idx_1, idx_2, size_1, size_2])

    pool = multi.Pool(processes=nproc)
    for i, j, val in pool.map(calc_cov_ssc, paramlist):
        #print(i, j, val)
        cov_out[i,j] = val
        
    return cov_out
    

def calc_cov_gauss(params):
    
    b_i, b_j, i, j, radius_1, radius_2, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, rho_bg, ingredient, subtract_randoms, idx_1, idx_2, size_1, size_2 = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    shape_noise_i = shape_noise[idx_1]
    shape_noise_j = shape_noise[idx_2]
    ngal_i = ngal[idx_1]
    ngal_j = ngal[idx_2]
    rho_i = rho_bg[idx_1]
    rho_j = rho_bg[idx_2]
    delta = np.eye(len(radius_1), len(radius_2))
    
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
                
    P_gm_i = np.exp(P_inter[idx_1][b_i](lnk))
    P_gm_j = np.exp(P_inter[idx_2][b_j](lnk))
                    
    P_gg_i = np.exp(P_inter_2[idx_1][b_i](lnk))
    P_gg_j = np.exp(P_inter_2[idx_2][b_j](lnk))
                    
    P_mm_i = np.exp(P_inter_3[idx_1][b_i](lnk))
    P_mm_j = np.exp(P_inter_3[idx_2][b_j](lnk))
                    
                    
    # wp
    if ingredient == 'gg':
        integ1 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * (P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j])*(P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j])
        val1 = ((2.0*Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ1, dx=dlnk)
        if not subtract_randoms:
            integ2 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j]))*np.sqrt((P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j]))
            val2 = (4.0*(2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ2, dx=dlnk)
        else:
            val2 = 0.0
        val = val1 + val2
    
                    
    # ESD
    if ingredient == 'gm':
        integ3 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.sqrt(P_mm_i + shape_noise_i[b_i]* delta[b_i, b_j])*np.sqrt(P_mm_j + shape_noise_j[b_j]* delta[b_i, b_j])) * (np.sqrt(P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j])*np.sqrt(P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j])) + (P_gm_i*P_gm_j) )
        val1 = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ3, dx=dlnk)
        if not subtract_randoms:
            integ4 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * (np.sqrt(P_mm_i + shape_noise_i[b_i]* delta[b_i, b_j])*np.sqrt(P_mm_j + shape_noise_j[b_j]* delta[b_i, b_j]))
            val2 = ((2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ4, dx=dlnk)
        else:
            val2 = 0.0
        val = (val1 + val2) * rho_i[b_i]*rho_j[b_j] / 1e24
    
    
    # cross
    if ingredient == 'cross':
        integ5 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (P_gm_i * (P_gg_i + 1.0/ngal_i[b_i]*delta[b_i, b_j])) + (P_gm_j  * (P_gg_j + 1.0/ngal_j[b_j]*delta[b_i, b_j])) )
        val1 = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk)
        if not subtract_randoms:
            integ6 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j]))*np.sqrt((P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j]))
            val2 = (2.0*(2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk)
        else:
            val2 = 0.0
            #integ7 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt(P_gm_i*P_gm_j)
            #val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk) + (2.0*(2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk) + (2.0*(2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ7, dx=dlnk) * np.sqrt(rho_i[b_i]*rho_j[b_j]) / 1e12
        val = (val1 + val2) * np.sqrt(rho_i[b_i]*rho_j[b_j]) / 1e12
    
    
    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val


def cov_gauss(radius_1, radius_2, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_out, nproc, rho_bg, ingredient, subtract_randoms, idx_1, idx_2, size_1, size_2):

    paramlist = []
    for a in range(len(radius_1)):
        for b in range(len(radius_2)):
            for c in range(len(radius_1[a])):
                for d in range(len(radius_2[b])):
                    paramlist.append([a, b, c, d])
    
    for i in paramlist:
        i.extend([radius_1, radius_2, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, rho_bg, ingredient, subtract_randoms, idx_1, idx_2, size_1, size_2])
    
    pool = multi.Pool(processes=nproc)
    for i, j, val in pool.map(calc_cov_gauss, paramlist):
        #print(i, j, val)
        cov_out[i,j] = val
    
    return cov_out


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

    #assert len(observables) == 1, \
    #    'working with more than one observable is not yet supported.' \
    #    ' If you would like this feature added please raise an issue.'
    
    # We might want to move this outside of the model code, but I am not sure where.
    nbins = 0
    ingredient_gm, ingredient_gg, ingredient_mm, ingredient_func = False, False, False, False
    hod_observable = None
    for i, observable in enumerate(observables):
        if observable.ingredient == 'gm':
            ingredient_gm = True
            observable_gm = observable
            hod_observable_gm = observable.sampling
            if hod_observable is None:
                hod_observable = hod_observable_gm
            else:
                hod_observable = np.concatenate([hod_observable, hod_observable_gm], axis=0)
            nbins_gm = observable.nbins
            idx_gm = np.s_[nbins:nbins+nbins_gm]
            nbins += nbins_gm
        if observable.ingredient == 'gg':
            ingredient_gg = True
            observable_gg = observable
            hod_observable_gg = observable.sampling
            if hod_observable is None:
                hod_observable = hod_observable_gg
            else:
                hod_observable = np.concatenate([hod_observable, hod_observable_gg], axis=0)
            nbins_gg = observable.nbins
            idx_gg = np.s_[nbins:nbins+nbins_gg]
            nbins += nbins_gg

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

    # if a single value is given for more than one bin, assign same
    # value to all bins
    #if not np.iterable(z):
    #    z = np.array(z)
    if z.size == 1 and nbins > 1:
        z = z*np.ones(nbins)
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    #z = expand_dims(z, -1)
    if z.size != nbins:
        raise ValueError(
            'Number of redshift bins should be equal to the number of observable bins!')

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
    #rvir_range_2d_i = R[0][1:]
    #rvir_range_2d_i = R[:,1:]
    if ingredient_gm:
        rvir_range_2d_i_gm = [r[1:].astype('float64') for r in R[idx_gm]]
        #rvir_range_2d_i_gm = [logspace(-2, 2, 25, endpoint=True) for r in R[idx_gm]]
        size_r_gm = np.array([len(r) for r in rvir_range_2d_i_gm])
    if ingredient_gg:
        rvir_range_2d_i_gg = [r[1:].astype('float64') for r in R[idx_gg]]
        #rvir_range_2d_i_gg = [logspace(-2, 2, 25, endpoint=True) for r in R[idx_gg]]
        size_r_gg = np.array([len(r) for r in rvir_range_2d_i_gg])
    #if ingredients['mm']:
        #rvir_range_2d_i_mm = [r[1:].astype('float64') for r in R[idx_mm]] # mm not used in this code!
    # We might want to move this in the configuration part of the code!
    # Same goes for the bins above
    
    # Calculating halo model
    
    # interpolate selection function to the same grid as redshift and
    # observable to be used in trapz
    
    completeness = np.ones(hod_observable.shape)
    if ingredients['centrals']:
        pop_c, prob_c = hod.number(
            hod_observable, mass_range, c_mor[0], c_scatter[0],
            c_mor[1:], c_scatter[1:], completeness,
            obs_is_log=observable_gm.is_log)
    else:
        pop_c = np.zeros((nbins,mass_range.size))
        prob_c = np.zeros((nbins,hod_observable.shape[1],mass_range.size))

    if ingredients['satellites']:
        pop_s, prob_s = hod.number(
            hod_observable, mass_range, s_mor[0], s_scatter[0],
            s_mor[1:], s_scatter[1:], completeness,
            obs_is_log=observable_gm.is_log)
    else:
        pop_s = np.zeros(pop_c.shape)
        prob_s = np.zeros(prob_c.shape)

    pop_g = pop_c + pop_s
    prob_g = prob_c + prob_s

    # note that pop_g already accounts for incompleteness
    dndm = array([hmf_i.dndm for hmf_i in hmf])
    ngal = hod.nbar(dndm, pop_g, mass_range)
    meff = hod.Mh_effective(
        dndm, pop_g, mass_range, return_log=observable_gm.is_log)
                   
    """
    # Power spectrum
    """
                   
                   
    # damping of the 1h power spectra at small k
    F_k1 = sp.erf(k_range_lin/0.1)
    F_k2 = np.ones_like(k_range_lin)
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
            [two_halo_gm(hmf_i, ngal_i, pop_g_i, mass_range)[0]
            for hmf_i, ngal_i, pop_g_i
            in zip(hmf, expand_dims(ngal, -1),
                    expand_dims(pop_g, -2))])

    bias_num = bias * array(
            [two_halo_gm(hmf_i, ngal_i, pop_g_i, mass_range)[1]
            for hmf_i, ngal_i, pop_g_i
            in zip(hmf, expand_dims(ngal, -1),
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
            [two_halo_gg(hmf_i, ngal_i, pop_g_i, mass_range)[0]
            for hmf_i, ngal_i, pop_g_i
            in zip(hmf, expand_dims(ngal, -1),
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
                    
    
    # Evaluate halo model integrals needed for SSC
    
    I_g = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(hmf_i), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'g')
                                   for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                                   zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
                                   
    I_m = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(hmf_i), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'm')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
                                    
    I_gg = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(hmf_i), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'gg')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
                                    
    I_gm = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(hmf_i), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'gm')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
                                    
    I_mm = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(hmf_i), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'mm')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
    
    I_inter_g = [UnivariateSpline(k_range, np.log(I_g_i), s=0, ext=0)
               for I_g_i in I_g]
               
    I_inter_m = [UnivariateSpline(k_range, np.log(I_m_i), s=0, ext=0)
                for I_m_i in I_m]
                 
    I_inter_gg = [UnivariateSpline(k_range, np.log(I_gg_i), s=0, ext=0)
                for I_gg_i in I_gg]
                 
    I_inter_gm = [UnivariateSpline(k_range, np.log(I_gm_i), s=0, ext=0)
                for I_gm_i in I_gm]
                
    I_inter_mm = [UnivariateSpline(k_range, np.log(I_mm_i), s=0, ext=0)
                for I_mm_i in I_mm]
    
    P_lin_inter = [UnivariateSpline(k_range, np.log(hmf_i.power), s=0, ext=0)
                for hmf_i in hmf]
               
    k3P_lin_inter = [UnivariateSpline(k_range, np.log(k_range_lin**3.0 * hmf_i.power), s=0, ext=0)
                for hmf_i in hmf]
                
    dlnk3P_lin_interdlnk = [f.derivative() for f in k3P_lin_inter]

    
    # Start covariance calculations (and for now set survey details)

    # These need to be inputs!!!!!
    #Pi_max, kids_area, eff_density, kids_variance_squared, z_kids, gauss, ssc, ng, cross, spec_z_path, nproc, subtract_randoms
    
    # Check Benjamin's code how the realistic survey geometry is accounted for!
    
    Pi_max = 100.0
    kids_area = 140*3600.0#180.0*3600.0
    eff_density = 8.53
    kids_variance_squared = 0.082
    z_kids = 0.6
    gauss = True
    ssc = False
    ng = False
    cross = True
    subtract_randoms = False # if randoms are not subtracted, this will increase the error bars
    nproc = 4

    #sigma_crit = sigma_crit_kids(hmf, z, 0.2, 0.9, spec_z_path) * hmf[0].cosmo.h * 10.0**12.0 / (1.0+z)**2.0
    sigma_crit = sigma_crit_func(cosmo_model, z, 1.2)
    print(sigma_crit)
    
    eff_density_in_mpc = eff_density  / ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value / hmf[0].cosmo.h )**2.0
    eff_density_in_rad = eff_density * (10800.0/np.pi)**2.0
    #shape_noise = ((sigma_crit / rho_bg)**2.0) * (kids_variance_squared / eff_density_in_rad)  * ((hmf[0].cosmo.angular_diameter_distance(z).value)**2.0 / hmf[0].cosmo.angular_diameter_distance_z1z2(0,z_kids).value)
    shape_noise = ((sigma_crit / rho_bg)**2.0) * (kids_variance_squared / eff_density_in_rad)  * ((hmf[0].cosmo.angular_diameter_distance(z).value)**2.0 / hmf[0].cosmo.angular_diameter_distance(z_kids).value)

    

    radius = np.sqrt(kids_area) * ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) / hmf[0].cosmo.h
    
    print(radius)
    print(eff_density_in_mpc)
    
    #shape_noise = np.zeros_like(shape_noise)
    #ngal = 1e50 * np.ones_like(ngal)
    
    print(shape_noise)
    print(1.0/ngal)
    
    
    # Check survey variance, and W_p (will depend on survey geometry what kind of values are needed)!
    W = 2.0*np.pi*radius**2.0 * sp.jv(1, k_range_lin*radius) / (k_range_lin*radius)
    W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)
    #survey_var = [survey_variance(hmf_i, W_p, k_range, np.pi*radius**2.0*Pi_max) for hmf_i in hmf]
    survey_var = [survey_variance(hmf_i, W_p, k_range, radius) for hmf_i in hmf]
    
        
    print('Halo integrals done.')
    lnk_min, lnk_max = np.log(0.001), np.log(100.0)
    k_temp = np.linspace(lnk_min, lnk_max, 100, endpoint=True)
    k_temp_lin = np.exp(k_temp)
  
    if ng == True:
        #global Tgggg, Tgggm, Tgmgm, T234h
        
        Tgggg = array([trispectra_1h(k_temp_lin, hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, k_range_lin, 'gggg')
                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
    
        Tgggm = array([trispectra_1h(k_temp_lin, hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, k_range_lin, 'gggm')
                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
                    
        Tgmgm = array([trispectra_1h(k_temp_lin, hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, k_range_lin, 'gmgm')
                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
    
        T234h = array([trispectra_234h(k_temp_lin, P_lin_inter_i, hmf_i, u_k_i, bias_tinker10(hmf_i), rho_bg_i, mass_range, k_range_lin)
                    for P_lin_inter_i, hmf_i, u_k_i, rho_bg_i in
                    zip(P_lin_inter, hmf, uk_c, rho_bg)])
        print('Trispectra done.')
   

    if ingredient_gm:
        cov_esd = np.zeros((size_r_gm.sum(), size_r_gm.sum()), dtype=np.float64)
        cov_esd_tot = cov_esd.copy()
    if ingredient_gg:
        cov_wp = np.zeros((size_r_gg.sum(), size_r_gg.sum()), dtype=np.float64)
        cov_wp_tot = cov_wp.copy()
    if ingredient_gm and ingredient_gg:
        cov_cross = np.zeros((size_r_gg.sum(), size_r_gm.sum()), dtype=np.float64)
        cov_cross_tot = cov_cross.copy()
    
    """
    
    # For simulation cube as in Mohammed et al. 2017!
    W = 500.0**3.0 * sp.jv(1, k_range_lin*500.0) / (k_range_lin*500.0)
    W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)
    survey_var = [survey_variance_test(hmf_i, W_p, k_range, 500.0**3.0) for hmf_i in hmf]
    
    
    #shape_noise[0] = 0.0
    test_gauss = np.zeros((len(k_temp_lin), len(k_temp_lin)))
    delta = np.eye(len(k_temp_lin))
    for i, k in enumerate(k_temp_lin):
        for j, l in enumerate(k_temp_lin):
            test_gauss[i,j] = 2.0 * ((np.sqrt(np.exp(P_inter_3[0](np.log(k)))) * np.sqrt(np.exp(P_inter_3[0](np.log(l))))) + delta[i,j]*shape_noise[0])**2.0

    test_gauss = delta * test_gauss
    
    ps_deriv_mm = ((68.0/21.0 - (1.0/3.0)*np.sqrt(dlnk3P_lin_interdlnk[0](k_temp))*np.sqrt(dlnk3P_lin_interdlnk[0](k_temp))) * np.sqrt(np.exp(P_lin_inter[0](k_temp)))*np.sqrt(np.exp(P_lin_inter[0](k_temp))) * np.exp(I_inter_m[0](k_temp))*np.exp(I_inter_m[0](k_temp)) + np.sqrt(np.exp(I_inter_mm[0](k_temp)))*np.sqrt(np.exp(I_inter_mm[0](k_temp))) )/ (np.exp(P_inter_3[0](k_temp)))
    
    test_1h = trispectra_1h(k_temp_lin, hmf[0], uk_c[0], uk_s[0], rho_bg[0], ngal[0], pop_c[0], pop_s[0], mass_range, k_range_lin, 'mmmm')
    test_1h = test_1h(k_temp_lin, k_temp_lin)
    import matplotlib.pyplot as pl
    pl.imshow(test_1h, interpolation='nearest')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/test_1h.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    #print(test_1h)

    test_ah = trispectra_234h(k_temp_lin, P_lin_inter[0], hmf[0], uk_c[0], bias_tinker10(hmf[0]), rho_bg[0], mass_range, k_range_lin)
    test = test_ah[0](k_temp_lin, k_temp_lin)
    test_2h = test_ah[1](k_temp_lin, k_temp_lin)
    test_3h = test_ah[2](k_temp_lin, k_temp_lin)
    test_4h = test_ah[3](k_temp_lin, k_temp_lin)
    #test = test/100.0
    #test_block = test/np.sqrt(np.outer(np.diag(test), np.diag(test.T)))
    test_tot = test_1h + test

    pl.imshow(test, interpolation='nearest')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/test_tri.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    #print(test)

    #pl.plot(mass_range, hmf[0].dndm)
    #pl.plot(mass_range, hmf[1].dndm)
    #pl.plot(mass_range, hmf[2].dndm)
    #pl.xscale('log')
    #pl.yscale('log')
    #pl.show()
    
    pl.plot(k_temp_lin, ps_deriv_mm)
    pl.xscale('log')
    pl.yscale('log')
    pl.xlim([0.01, 2.5])
    pl.ylim([0.15, 4])
    pl.xlabel('k [h/Mpc]')
    pl.ylabel(r'$\rm{d \ln} P(k) / \rm{d \delta_{b}}$')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/ssc_mm.png', bbox_inches='tight')
    pl.show()
    pl.clf()


    volume = 500.0**3.0#np.pi*radius**2.0*Pi_max*2.0
    loc = 0.51
    index = np.argmin(np.abs(k_temp_lin - loc))
    #index = 25
    print(k_temp_lin[index])
    print(index)
    st = (k_temp_lin[index+1]-k_temp_lin[index-1])/(k_temp_lin[index+1]+k_temp_lin[index-1])#/k_temp_lin[index]
    print(st)
    Nmode = 0.5*(k_temp_lin[index]**2.0 * volume * st) / (2.0*np.pi)**2.0
    
    denom = np.outer(np.exp(P_inter_3[0](k_temp)), np.exp(P_inter_3[0](k_temp)).T)
    
    #pl.plot(k_temp_lin, np.sqrt(((test_gauss/Nmode) / denom))[:,index], color='red', label='gauss')
    #pl.plot(k_temp_lin, np.sqrt((test_gauss/Nmode + test_tot/volume)/denom + (survey_var[0] * np.outer(ps_deriv_mm, ps_deriv_mm)))[:,index], color='black', label='tot')
    pl.plot(k_temp_lin, np.sqrt((test/volume)/denom)[:,index], color='orange', ls='-.', label='tri')
    pl.plot(k_temp_lin, np.sqrt((test_2h/volume)/denom)[:,index], label='2h')
    pl.plot(k_temp_lin, np.sqrt((test_3h/volume)/denom)[:,index], label='3h')
    pl.plot(k_temp_lin, np.sqrt((test_4h/volume)/denom)[:,index], label='4h')
    pl.plot(k_temp_lin, np.sqrt((test_1h/volume)/denom)[:,index], color='orange', ls='--', label='1h')
    #pl.plot(k_temp_lin, np.sqrt((survey_var[0] * np.outer(ps_deriv_mm, ps_deriv_mm)))[:,index], color='blue', label='ssc')
    pl.xscale('log')
    pl.xlim([0.01, 1.0])
    pl.ylim([0.0, 0.08])
    pl.legend()
    #pl.yscale('log')
    pl.xlabel('k [h/Mpc]')
    pl.ylabel(r'$\rm{\sqrt{Cov/P(k)P(k\prime)}}$')
    pl.title(r'$k\prime = %f $'%loc)
    pl.savefig('/home/dvornik/test_pipeline2/covariance/tot_mm.png', bbox_inches='tight')
    pl.show()
    pl.clf()

    #pl.plot(k_temp_lin, (k_temp_lin**3.0 / (2.0*np.pi)**2.0) * np.diag(test)**(1.0/3.0))
    pl.plot(k_temp_lin, (k_temp_lin**3.0 / (2.0*np.pi)**2.0) * np.diag(test_1h)**(1.0/3.0), label='1h')
    pl.plot(k_temp_lin, (k_temp_lin**3.0 / (2.0*np.pi)**2.0) * np.diag(test_2h)**(1.0/3.0), label='2h')
    pl.plot(k_temp_lin, (k_temp_lin**3.0 / (2.0*np.pi)**2.0) * np.diag(test_3h)**(1.0/3.0), label='3h')
    pl.plot(k_temp_lin, (k_temp_lin**3.0 / (2.0*np.pi)**2.0) * np.diag(test_4h)**(1.0/3.0), label='4h')
    pl.plot(k_temp_lin, (k_temp_lin**3.0 / (2.0*np.pi)**2.0) * np.diag(test_tot)**(1.0/3.0), label='tot')
    pl.xscale('log')
    pl.xlim([0.01, 100.0])
    #pl.ylim([0.0, 0.08])
    pl.legend()
    pl.yscale('log')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/diag_mm.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    
    quit()
    #"""
    
    if gauss == True:
        print('Calculating the Gaussian part of the covariance ...')
        if ingredient_gm:
            cov_esd_gauss = cov_gauss(rvir_range_2d_i_gm, rvir_range_2d_i_gm, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_esd.copy(), nproc, rho_bg, 'gm', subtract_randoms, idx_gm, idx_gm, size_r_gm, size_r_gm)
            cov_esd_tot += cov_esd_gauss
        if ingredient_gg:
            cov_wp_gauss = cov_gauss(rvir_range_2d_i_gg, rvir_range_2d_i_gg, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_wp.copy(), nproc, rho_bg, 'gg', subtract_randoms, idx_gg, idx_gg, size_r_gg, size_r_gg)
            cov_wp_tot += cov_wp_gauss
        if ingredient_gm and ingredient_gg and cross:
            cov_cross_gauss = cov_gauss(rvir_range_2d_i_gg, rvir_range_2d_i_gm, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_cross.copy(), nproc, rho_bg, 'cross', subtract_randoms, idx_gg, idx_gm, size_r_gg, size_r_gm)
            cov_cross_tot += cov_cross_gauss
    else:
        pass
        

    if ssc == True:
        print('Calculating the super-sample covariance ...')
        if ingredient_gm:
            cov_esd_ssc = cov_ssc(rvir_range_2d_i_gm, rvir_range_2d_i_gm, P_lin_inter, dlnk3P_lin_interdlnk, P_inter, P_inter_2, I_inter_g, I_inter_m, I_inter_gg, I_inter_gm, W_p, Pi_max, bias_num, survey_var, cov_esd.copy(), nproc, rho_bg, 'gm', idx_gm, idx_gm, size_r_gm, size_r_gm)
            cov_esd_tot += cov_esd_ssc
        if ingredient_gg:
            cov_wp_ssc = cov_ssc(rvir_range_2d_i_gg, rvir_range_2d_i_gg, P_lin_inter, dlnk3P_lin_interdlnk, P_inter, P_inter_2, I_inter_g, I_inter_m, I_inter_gg, I_inter_gm, W_p, Pi_max, bias_num, survey_var, cov_wp.copy(), nproc, rho_bg, 'gg', idx_gg, idx_gg, size_r_gg, size_r_gg)
            cov_wp_tot += cov_wp_ssc
        if ingredient_gm and ingredient_gg and cross:
            cov_cross_ssc = cov_ssc(rvir_range_2d_i_gg, rvir_range_2d_i_gm, P_lin_inter, dlnk3P_lin_interdlnk, P_inter, P_inter_2, I_inter_g, I_inter_m, I_inter_gg, I_inter_gm, W_p, Pi_max, bias_num, survey_var, cov_cross.copy(), nproc, rho_bg, 'cross', idx_gg, idx_gm, size_r_gg, size_r_gm)
            cov_cross_tot += cov_cross_ssc
    else:
        pass


    if ng == True:
        print('Calculating the connected (non-Gaussian) part of the covariance ...')
        if ingredient_gm:
            cov_esd_non_gauss = cov_non_gauss(rvir_range_2d_i_gm, rvir_range_2d_i_gm, Tgmgm, T234h, bias_num, W_p, np.pi*radius**2.0*Pi_max, cov_esd.copy(), nproc, rho_bg, 'gm', idx_gm, idx_gm, size_r_gm, size_r_gm)
            cov_esd_tot += cov_esd_non_gauss
        if ingredient_gg:
            cov_wp_non_gauss = cov_non_gauss(rvir_range_2d_i_gg, rvir_range_2d_i_gg, Tgggg, T234h, bias_num, W_p, np.pi*radius**2.0*Pi_max, cov_wp.copy(), nproc, rho_bg, 'gg', idx_gg, idx_gg, size_r_gg, size_r_gg)
            cov_wp_tot += cov_wp_non_gauss
        if ingredient_gm and ingredient_gg and cross:
            cov_cross_non_gauss = cov_non_gauss(rvir_range_2d_i_gg, rvir_range_2d_i_gm, Tgggm, T234h, bias_num, W_p, np.pi*radius**2.0*Pi_max, cov_cross.copy(), nproc, rho_bg, 'cross', idx_gg, idx_gm, size_r_gg, size_r_gm)
            cov_cross_tot += cov_cross_non_gauss
    else:
        pass
        
    """
    if ingredient_gm:
        return  cov_esd_tot
    if ingredient_gg:
        return cov_wp_tot
    if ingredient_gm and ingredient_gg:
        return np.block([[cov_esd_tot, cov_cross_tot.T], [cov_cross_tot, cov_wp_tot]])
    """
    import matplotlib.pyplot as pl
    pl.imshow(cov_esd_tot/np.sqrt(np.outer(np.diag(cov_esd_tot), np.diag(cov_esd_tot.T))))
    pl.savefig('/home/dvornik/test_pipeline2/covariance/esd_tot.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    
    pl.imshow(cov_wp_tot/np.sqrt(np.outer(np.diag(cov_wp_tot), np.diag(cov_wp_tot.T))))
    pl.savefig('/home/dvornik/test_pipeline2/covariance/wp_tot.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    
    """
    cov_esd_gauss = cov_esd_gauss**0.5
    #cov_esd_non_gauss = cov_esd_non_gauss**0.5
    #cov_esd_ssc = cov_esd_ssc**0.5
    
    cov_wp_gauss = cov_wp_gauss**0.5
    #cov_wp_non_gauss = cov_wp_non_gauss**0.5
    #cov_wp_ssc = cov_wp_ssc**0.5
    
    pl.plot(rvir_range_2d_i_gm[0], rvir_range_2d_i_gm[0]*np.diag(cov_esd_gauss)[:size_r_gm[0]], ls='-', color='black', label='Gauss')
    pl.plot(rvir_range_2d_i_gm[1], rvir_range_2d_i_gm[1]*np.diag(cov_esd_gauss)[size_r_gm[0]:size_r_gm[0]+size_r_gm[1]], ls='--', color='black')
    pl.plot(rvir_range_2d_i_gm[2], rvir_range_2d_i_gm[2]*np.diag(cov_esd_gauss)[size_r_gm[0]+size_r_gm[1]:], ls='-.', color='black')
    
    #pl.plot(rvir_range_2d_i_gm[0], rvir_range_2d_i_gm[0]*np.diag(cov_esd_non_gauss)[:size_r_gm[0]], ls='-', color='red', label='non-Gauss')
    #pl.plot(rvir_range_2d_i_gm[1], rvir_range_2d_i_gm[1]*np.diag(cov_esd_non_gauss)[size_r_gm[0]:size_r_gm[0]+size_r_gm[1]], ls='--', color='red')
    #pl.plot(rvir_range_2d_i_gm[2], rvir_range_2d_i_gm[2]*np.diag(cov_esd_non_gauss)[size_r_gm[0]+size_r_gm[1]:], ls='-.', color='red')
    
    #pl.plot(rvir_range_2d_i_gm[0], rvir_range_2d_i_gm[0]*np.diag(cov_esd_ssc)[:size_r_gm[0]], ls='-', color='blue', label='SSC')
    #pl.plot(rvir_range_2d_i_gm[1], rvir_range_2d_i_gm[1]*np.diag(cov_esd_ssc)[size_r_gm[0]:size_r_gm[0]+size_r_gm[1]], ls='--', color='blue')
    #pl.plot(rvir_range_2d_i_gm[2], rvir_range_2d_i_gm[2]*np.diag(cov_esd_ssc)[size_r_gm[0]+size_r_gm[1]:], ls='-.', color='blue')
    
    pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.xlabel(r'$r_{\mathrm{p}}\, (\mathrm{Mpc}/h)$')
    pl.ylabel(r'$\sigma_{\Delta \Sigma_{\mathrm{gm}}}(r_{\mathrm{p}})$')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/cov_esd_gauss.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    
    pl.plot(rvir_range_2d_i_gg[0], rvir_range_2d_i_gg[0]*np.diag(cov_wp_gauss)[:size_r_gg[0]], ls='-', color='black', label='Gauss')
    pl.plot(rvir_range_2d_i_gg[1], rvir_range_2d_i_gg[1]*np.diag(cov_wp_gauss)[size_r_gg[0]:size_r_gg[0]+size_r_gg[1]], ls='--', color='black')
    pl.plot(rvir_range_2d_i_gg[2], rvir_range_2d_i_gg[2]*np.diag(cov_wp_gauss)[size_r_gg[0]+size_r_gg[1]:], ls='-.', color='black')
    
    #pl.plot(rvir_range_2d_i_gg[0], rvir_range_2d_i_gg[0]*np.diag(cov_wp_non_gauss)[:size_r_gg[0]], ls='-', color='red', label='non-Gauss')
    #pl.plot(rvir_range_2d_i_gg[1], rvir_range_2d_i_gg[1]*np.diag(cov_wp_non_gauss)[size_r_gg[0]:size_r_gg[0]+size_r_gg[1]], ls='--', color='red')
    #pl.plot(rvir_range_2d_i_gg[2], rvir_range_2d_i_gg[2]*np.diag(cov_wp_non_gauss)[size_r_gg[0]+size_r_gg[1]:], ls='-.', color='red')
    
    #pl.plot(rvir_range_2d_i_gg[0], rvir_range_2d_i_gg[0]*np.diag(cov_wp_ssc)[:size_r_gg[0]], ls='-', color='blue', label='SSC')
    #pl.plot(rvir_range_2d_i_gg[1], rvir_range_2d_i_gg[1]*np.diag(cov_wp_ssc)[size_r_gg[0]:size_r_gg[0]+size_r_gg[1]], ls='--', color='blue')
    #pl.plot(rvir_range_2d_i_gg[2], rvir_range_2d_i_gg[2]*np.diag(cov_wp_ssc)[size_r_gg[0]+size_r_gg[1]:], ls='-.', color='blue')
    
    pl.xscale('log')
    pl.yscale('log')
    pl.legend()
    pl.xlabel(r'$r_{\mathrm{p}}\, (\mathrm{Mpc}/h)$')
    pl.ylabel(r'$\sigma_{w_{\mathrm{p}}}(r_{\mathrm{p}})$')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/cov_wp_gauss.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    
    
    pl.plot(rvir_range_2d_i_gm[0], rvir_range_2d_i_gm[0]*np.diag(cov_esd_gauss)[:size_r_gm[0]], ls='-', color='black', label='Gauss')
    pl.plot(rvir_range_2d_i_gm[1], rvir_range_2d_i_gm[1]*np.diag(cov_esd_gauss)[size_r_gm[0]:size_r_gm[0]+size_r_gm[1]], ls='--', color='black')
    pl.plot(rvir_range_2d_i_gm[2], rvir_range_2d_i_gm[2]*np.diag(cov_esd_gauss)[size_r_gm[0]+size_r_gm[1]:], ls='-.', color='black')
    
    pl.xscale('log')
    pl.yscale('log')
    pl.xlim(1, 100)
    pl.legend()
    pl.xlabel(r'$r_{\mathrm{p}}\, (\mathrm{Mpc}/h)$')
    pl.ylabel(r'$r_{\mathrm{p}}\, \sigma_{\Delta \Sigma_{\mathrm{gm}}}(r_{\mathrm{p}})$')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/cov_esd_gauss_zoom.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    
    pl.plot(rvir_range_2d_i_gg[0], rvir_range_2d_i_gg[0]*np.diag(cov_wp_gauss)[:size_r_gg[0]], ls='-', color='black', label='Gauss')
    pl.plot(rvir_range_2d_i_gg[1], rvir_range_2d_i_gg[1]*np.diag(cov_wp_gauss)[size_r_gg[0]:size_r_gg[0]+size_r_gg[1]], ls='--', color='black')
    pl.plot(rvir_range_2d_i_gg[2], rvir_range_2d_i_gg[2]*np.diag(cov_wp_gauss)[size_r_gg[0]+size_r_gg[1]:], ls='-.', color='black')
    
    pl.xscale('log')
    pl.yscale('log')
    pl.xlim(1, 100)
    pl.legend()
    pl.xlabel(r'$r_{\mathrm{p}}\, (\mathrm{Mpc}/h)$')
    pl.ylabel(r'$r_{\mathrm{p}}\, \sigma_{w_{\mathrm{p}}}(r_{\mathrm{p}})$')
    pl.savefig('/home/dvornik/test_pipeline2/covariance/cov_wp_gauss_zoom.png', bbox_inches='tight')
    pl.show()
    pl.clf()
    
    #"""
    
    
    cov_block = np.block([[cov_esd_tot, cov_cross_tot.T],
                        [cov_cross_tot, cov_wp_tot]])
    if not subtract_randoms:
        np.save('/home/dvornik/test_pipeline2/covariance/cov_no_sub_rand.npy', cov_block)
    else:
        np.save('/home/dvornik/test_pipeline2/covariance/cov_sub_rand.npy', cov_block)
    pl.rcParams["figure.figsize"] = (10,10)
    pl.imshow(cov_block/np.sqrt(np.outer(np.diag(cov_block), np.diag(cov_block.T))))
    #[pl.axhline(_x, lw=1, color='white') for _x in [7.5, 15.5, 21.5, 29.5]]
    #[pl.axvline(_x, lw=1, color='white') for _x in [7.5, 15.5, 21.5, 29.5]]
    pl.savefig('/home/dvornik/test_pipeline2/covariance/comb_tot.png', bbox_inches='tight', dpi=360)
    pl.show()
    pl.clf()
    
    return 0
    

if __name__ == '__main__':
    print(0)

























