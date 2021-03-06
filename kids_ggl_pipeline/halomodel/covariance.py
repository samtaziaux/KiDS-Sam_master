#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from astropy.units import Quantity
import astropy.io.fits as fits
import healpy as hp

from hmf import MassFunction
import hmf.mass_function.fitting_functions as ff
import hmf.density_field.transfer_models as tf

from . import baryons, longdouble_utils as ld, nfw
from .tools import (
    fill_nan, load_hmf_cov, virial_mass, virial_radius)
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

def sigma_crit_kids(hmf, z_in, z_epsilon, srclim, lens_photoz, galsigma, lens_pz_redshift, spec_cat_path):
    """
    This uses the sigma_crit calculation from ESD extraction part of the code,
        that is accounting for KiDS specific setup.

    Parameters
    ----------
    Check the relevant files in esd_production!

    """
    
    from kids_ggl_pipeline.esd_production.shearcode_modules import calc_Sigmacrit
    
    # This might be removed!
    if lens_photoz:
        lens_photoz = True
        if lens_pz_redshift:
            lens_pz_redshift = True
        else:
            lens_pz_redshift = False
    else:
        lens_photoz = False
        lens_pz_redshift = False
        galsigma = 0.0
        
    
    zsrcbins = np.arange(0.025,3.5,0.05)
    zlensbins =  np.linspace(0.025, 0.7, 100)
    
    Dcsbins = np.array((hmf[0].cosmo.comoving_distance(zsrcbins).to('pc')).value)
    Dclbins = np.array((hmf[0].cosmo.comoving_distance(zlensbins).to('pc')).value)
    Dc_epsilon = (hmf[0].cosmo.comoving_distance(z_epsilon).to('pc')).value

    spec_cat = fits.open(spec_cat_path, memmap=True)[1].data
    Z_S = spec_cat['z_spec']
    spec_weight = spec_cat['spec_weight']
    manmask = spec_cat['MASK']
    srcmask = (manmask==0)

    sigma_selection = {}
    # 10 lens redshifts for calculation of Sigma_crit
    lens_redshifts = np.arange(0.001, srclim-z_epsilon, 0.05)
    lens_comoving = np.array((hmf[0].cosmo.comoving_distance(lens_redshifts).to('pc')).value)
            
            
    lens_angular = lens_comoving/(1.0+lens_redshifts)
    k = np.zeros_like(lens_redshifts)
            
    for i in range(lens_redshifts.size):
        srcmask *= (lens_redshifts[i]+z_epsilon <= spec_cat['Z_B']) & (spec_cat['Z_B'] < srclim)
        srcNZ_k, spec_weight_k = Z_S[srcmask], spec_weight[srcmask]
                    
        srcPZ_k, bins_k = np.histogram(srcNZ_k, range=[0.025, 3.5], bins=70, weights=spec_weight_k, density=1)
        srcPZ_k = srcPZ_k/srcPZ_k.sum()
        k[i], kmask = calc_Sigmacrit(np.array([lens_comoving[i]]), np.array([lens_angular[i]]), \
                        Dcsbins, zlensbins, Dclbins, lens_photoz, galsigma, lens_pz_redshift, \
                        srcPZ_k, 3, Dc_epsilon, np.array([lens_redshifts[i]]), True)
            
    k_interpolated = interp1d(lens_redshifts, k, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0))

    return 1.0/k_interpolated(z_in)


def survey_variance_test(hmf, W_p, k_range, radius):

    # Seems to be about right! To be checked again!.
    P_lin = hmf.power

    # This works when testing against Mohammed et al. 2017. Survey variance for a simulation cube!
    v_w = radius
    integ2 = W_p(np.exp(k_range))**2.0 * np.exp(k_range)**2.0 * P_lin
    sigma = (1.0 / (2.0*np.pi**2.0 * v_w**2.0)) * simps(integ2, np.exp(k_range))

    return sigma


def survey_variance(hmf, W_p, k_range, radius):
    
    # Seems to be about right! To be checked again!.
    
    P_lin = hmf.power

    Awr = simps(np.exp(k_range)**2.0 * sp.jv(0, np.exp(k_range) * radius)/(2.0*np.pi) * W_p(np.exp(k_range))**2.0, k_range)
    integ2 = W_p(np.exp(k_range))**2.0 * np.exp(k_range)**2.0 * P_lin
    sigma = (1.0 / (2.0*np.pi**2.0 * Awr)) * simps(integ2, np.exp(k_range))
    
    return sigma
    
    
def calculate_alms(maskfile, binary):
    # functions borrowed from Benjamin's code

    data = fits.getdata(maskfile, 1).field(0).flatten()
    nside = hp.npix2nside(len(data))

    if (binary == 1):
        data_final = np.where(data<1.0,0,1)
    else:
        data_final = data

    survey_area = np.count_nonzero(data_final)*hp.nside2pixarea(nside,degrees=True)
    
    lmax = 3*nside-1  # default range suggested by healpy
    Cl = hp.anafast(data_final, use_weights=True)

    l_range = np.arange(lmax)
    alms = (2*l_range+1)*Cl[:lmax]
    
    return l_range, alms, survey_area
    
    
def survey_variance_from_healpy(hmf, k_range, z, l_range, alms, survey_area):
    # functions borrowed from Benjamin's code

    P_lin_inter = UnivariateSpline(k_range, np.log(hmf.power), s=0, ext=0)

    fk = hmf.cosmo.angular_diameter_distance(z).value
    k = l_range/fk
    ps = np.exp(P_lin_inter(np.log(k)))
    
    sigma = np.nansum(ps * alms)/survey_area**2.0
    
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
    
    if (np.fabs(k1-k2) < trispec_matter_klim) and ((1.0-mu) < trispec_matter_mulim):
        k_m = np.zeros(mu.shape) #avoid nan in sqrt
        mu_1m = np.zeros(mu.shape)   #undefined
        alpha_m = np.ones(mu.shape)
        beta_m = np.zeros(mu.shape)  # undefined
    
    else:
        k_m = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu) # |k_-|
        mu_1m = (k2/k_m)*mu - (k1/k_m) # (k1*k_-)/[k1 k_-]
        alpha_m = pt_kernel_alpha(k_m, k1, mu_1m)
        beta_m = pt_kernel_beta(k1, k_m, mu_1m)

    if (np.fabs(k1-k2) < trispec_matter_klim) and ((mu+1.0) < trispec_matter_mulim):
        k_p = np.zeros(mu.shape) # avoid nan in sqrt
        mu_1p = np.zeros(mu.shape) # undefined
        alpha_p = np.ones(mu.shape)
        beta_p = np.zeros(mu.shape) # undefined
        
    else:
        k_p = np.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu) # |k_+|
        mu_1p = (k1/k_p) + mu*(k2/k_p) # (k1*k_+)/[k1 k_+]
        alpha_p = pt_kernel_alpha(k_p, k1, (-1.0)*mu_1p)
        beta_p = pt_kernel_beta(k1, k_p, (-1.0)*mu_1p)
            
    F2_plus=pt_kernel_f2(k1, k2, mu)
    F2_minus=pt_kernel_f2(k1, k2, (-1.0)*mu)
    G2_plus=pt_kernel_g2(k1, k2, mu)
    G2_minus=pt_kernel_g2(k1, k2, (-1.0)*mu)

    return ((7.0/54.0)*(pt_kernel_alpha(k1,k_m,mu_1m)*F2_minus + pt_kernel_alpha(k1,k_p,(-1.0)*mu_1p)*F2_plus) + (4.0/54.0)*(beta_m*G2_minus + beta_p*G2_plus) + (7.0/54.0)*(alpha_m*G2_minus + alpha_p*G2_plus))

def trispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    
    if (np.fabs(k1-k2) < trispec_matter_klim) and ((1.0-mu) < trispec_matter_mulim):
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

    if (np.fabs(k1-k2) < trispec_matter_klim) and ((mu+1.0) < trispec_matter_mulim):
        k_p = np.zeros(mu.shape) #avoid nan in sqrt
        mu_1p = np.zeros(mu.shape)   #undefined
        mu_2p = np.zeros(mu.shape)
        p_p = np.zeros(mu.shape)
        F2_1p = np.zeros(mu.shape)  # undefined
        F2_2p = np.zeros(mu.shape)
    else:

        k_p = np.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu) # |k_+|
        mu_1p = (k1/k_p) + mu*(k2/k_p) # (k1*k_+)/[k1 k_+]
        mu_2p = (k1/k_p)*mu + (k2/k_p) # (k2*k_+)/[k2 k_+]
        p_p = np.exp(P_lin_inter(np.log(k_p)))
        F2_1p = pt_kernel_f2(k1, k_p, (-1.0)*mu_1p)
        F2_2p = pt_kernel_f2(k2, k_p, (-1.0)*mu_2p)

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
    
    if (np.fabs(k1-k2) < trispec_matter_klim) and ((mu+1.0) < trispec_matter_mulim):
        k_p = np.zeros(mu.shape)
        term2 = np.zeros(mu.shape)
        term3 = np.zeros(mu.shape)

    else:
        k_p = np.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu)
        p_p = np.exp(P_lin_inter(np.log(k_p)))
        mu_1p = (k1/k_p) + mu*(k2/k_p) # (k1*k_+)/[k1 k_+]
        mu_2p = (k1/k_p)*mu + (k2/k_p) # (k2*k_+)/[k2 k_+]
        term2 = 2.0*pt_kernel_f2(k1, k_p, (-1.0)*mu_1p)*p1*p_p
        term3 = 2.0*pt_kernel_f2(k2, k_p, (-1.0)*mu_2p)*p2*p_p
    
    return term1 + term2 + term3


# These are integrated over 2PI to get the angular average, for each k1, k2 combination!
def intg_for_trispec_matter_parallel_2h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    if (np.fabs(k1-k2) < trispec_matter_klim) and ((1.0-mu) < trispec_matter_mulim):
        k_m = np.zeros(mu.shape)
        p_m = np.zeros(mu.shape)
    else:
        k_m = np.sqrt(k1*k1 + k2*k2 - 2.0*k1*k2*mu)
        p_m = np.exp(P_lin_inter(np.log(k_m)))

    if (np.fabs(k1-k2) < trispec_matter_klim) and ((mu+1.0) < trispec_matter_mulim):
        k_p = np.zeros(mu.shape)
        p_p = np.zeros(mu.shape)
    else:
        k_p = np.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu)
        p_p = np.exp(P_lin_inter(np.log(k_p)))
    
    return p_p + p_m


def intg_for_trispec_matter_parallel_3h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    return bispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim) + bispec_parallel_pt(k1, k2, (-1.0)*mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim)


def intg_for_trispec_matter_parallel_4h(x, k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim):
    mu = np.cos(x)
    return trispec_parallel_pt(k1, k2, mu, P_lin_inter, trispec_matter_mulim, trispec_matter_klim)


def trispectra_234h(krange, P_lin_inter, mass_func, uk, bias, rho_bg, m_x, k_x):
    
    trispec_matter_mulim = 0.001
    trispec_matter_klim = 0.001
    
    trispec_2h = np.zeros((krange.size, krange.size))
    trispec_3h = np.zeros((krange.size, krange.size))
    trispec_4h = np.zeros((krange.size, krange.size))
    
    # Evaluate u(k) on different k grid!
    u_k = np.array([UnivariateSpline(k_x, uk[m,:], s=0, ext=0) for m in range(len(m_x))])
    u_k_new = np.array([u_k[m](krange) for m in range(len(m_x))])
    
    def Im(i, mass_func, uk, bias, rho_bg, m_x):
        integ = mass_func.dndm * bias * uk[:,i] * m_x
        I = trapz(integ, m_x)/(rho_bg)
        return I
    
    def Imm(i, j, mass_func, uk, bias, rho_bg, m_x):
        integ = mass_func.dndm * bias * uk[:,i] * uk[:,j] * m_x**2.0
        I = trapz(integ, m_x)/(rho_bg**2.0)
        return I
    
    def Immm(i, j, k,  mass_func, uk, bias, rho_bg, m_x):
        integ = mass_func.dndm * bias * uk[:,i] * uk[:,j] * uk[:,k] * m_x**3.0
        I = trapz(integ, m_x)/(rho_bg**3.0)
        return I
    
    x = np.linspace(0.0, 2.0*np.pi, num=100, endpoint=True)
    
    for i, k1 in enumerate(krange):
        for j, k2 in enumerate(krange):
            integral_2h = quad(intg_for_trispec_matter_parallel_2h, 0.0, 2.0*np.pi, args=(k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), limit=50, maxp1=50, limlst=50)[0]/(2.0*np.pi)
            integral_3h = quad(intg_for_trispec_matter_parallel_3h, 0.0, 2.0*np.pi, args=(k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), limit=50, maxp1=50, limlst=50)[0]/(2.0*np.pi)
            integral_4h = quad(intg_for_trispec_matter_parallel_4h, 0.0, 2.0*np.pi, args=(k1, k2, P_lin_inter, trispec_matter_mulim, trispec_matter_klim), limit=50, maxp1=50, limlst=50)[0]/(2.0*np.pi)
            
            trispec_2h[i,j] = 2.0 * Immm(i, j, j, mass_func, u_k_new, bias, rho_bg, m_x) * Im(i, mass_func, u_k_new, bias, rho_bg, m_x) * np.exp(P_lin_inter(np.log(k1))) + 2.0 * Immm(i, i, j, mass_func, u_k_new, bias, rho_bg, m_x) * Im(j, mass_func, u_k_new, bias, rho_bg, m_x) * np.exp(P_lin_inter(np.log(k2))) + (Imm(i, j, mass_func, u_k_new, bias, rho_bg, m_x)**2.0) * integral_2h
            trispec_3h[i,j] = 2.0 * Imm(i, j, mass_func, u_k_new, bias, rho_bg, m_x) * Im(i, mass_func, u_k_new, bias, rho_bg, m_x) * Im(j, mass_func, u_k_new, bias, rho_bg, m_x) * integral_3h
            trispec_4h[i,j] = (Im(i, mass_func, u_k_new, bias, rho_bg, m_x))**2.0 * (Im(j, mass_func, u_k_new, bias, rho_bg, m_x))**2.0 * integral_4h
    
    trispec_2h = np.nan_to_num(trispec_2h)
    trispec_3h = np.nan_to_num(trispec_3h)
    trispec_4h = np.nan_to_num(trispec_4h)

    """
    # Test
    trispec_2h = np.triu(trispec_2h,0) + np.tril(trispec_2h.T,-1)
    trispec_3h = np.triu(trispec_3h,0) + np.tril(trispec_3h.T,-1)
    trispec_4h = np.triu(trispec_4h,0) + np.tril(trispec_4h.T,-1)
    #trispec_4h = trispec_4h - np.diag(trispec_4h)*np.eye(krange.size)
    #"""

    trispec_tot = trispec_2h + trispec_3h + trispec_4h
    trispec_tot_interp = RectBivariateSpline(krange, krange, trispec_tot, kx=1, ky=1)
    
    """
    # Test mode:
    trispec_2h_interp = RectBivariateSpline(krange, krange, trispec_2h, kx=1, ky=1)
    trispec_3h_interp = RectBivariateSpline(krange, krange, trispec_3h, kx=1, ky=1)
    trispec_4h_interp = RectBivariateSpline(krange, krange, trispec_4h, kx=1, ky=1)
    return trispec_tot_interp, trispec_2h_interp, trispec_3h_interp, trispec_4h_interp
    #"""
    return trispec_tot_interp
    

def poisson(mu, fac=0.9):
    res=1.0
    for i in range(2,mu):
        res *= (mu - 1.0) * fac - mu + 2.0
    return res
    

def trispectra_1h(krange, mass_func, uk_c, uk_s, rho_bg, ngal, population_cen, population_sat, m_x, k_x, x):
    
    trispec_1h = np.zeros((krange.size, krange.size))

    if x == 'gmgm':
        u_g_prod2 = 2.0 * expand_dims(population_cen, -1) * expand_dims(population_sat, -1) * uk_s + expand_dims(population_sat, -1)**2.0 * uk_s**2.0
        u_m_prod = (expand_dims(m_x, -1) * uk_c)**2.0
        
        u_g2 = np.array([UnivariateSpline(k_x, u_g_prod2[m,:], s=0, ext=0) for m in range(len(m_x))])
        u_m = np.array([UnivariateSpline(k_x, u_m_prod[m,:], s=0, ext=0) for m in range(len(m_x))])
        u_m_new = np.array([u_m[m](krange) for m in range(len(m_x))])
        u_g_new2 = np.array([u_g2[m](krange) for m in range(len(m_x))])
        
        norm_g = ngal
        norm_m = rho_bg
        
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_g_new2[:,i] * u_m_new[:,i]
                vec2 = u_g_new2[:,j] * u_m_new[:,j]
                integ = mass_func.dndm * (vec1 * vec2)**0.5
                trispec_1h[i,j] = trapz(integ, m_x) / (norm_g*norm_g*norm_m*norm_m)
    
    if x == 'gggm':
        u_g_prod3 = 3.0 * expand_dims(population_cen, -1) * expand_dims(population_sat, -1)**2.0 * uk_s**2.0 + expand_dims(population_sat, -1)**3.0 * uk_s**3.0
        u_m_prod = expand_dims(m_x, -1) * uk_c
        
        u_g3 = np.array([UnivariateSpline(k_x, u_g_prod3[m,:], s=0, ext=0) for m in range(len(m_x))])
        u_m = np.array([UnivariateSpline(k_x, u_m_prod[m,:], s=0, ext=0) for m in range(len(m_x))])
        u_m_new = np.array([u_m[m](krange) for m in range(len(m_x))])
        u_g_new3 = np.array([u_g3[m](krange) for m in range(len(m_x))])
        
        norm_g = ngal
        norm_m = rho_bg
        
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_g_new3[:,i] * u_m_new[:,i]
                vec2 = u_g_new3[:,j] * u_m_new[:,j]
                integ = mass_func.dndm * (vec1 * vec2)**0.5 * poisson(3)
                trispec_1h[i,j] = trapz(integ, m_x) / (norm_g*norm_g*norm_g*norm_m)
    
    if x == 'gggg':
        u_g_prod4 = 4.0 * expand_dims(population_cen, -1) * expand_dims(population_sat, -1)**3.0 * uk_s**3.0 + expand_dims(population_sat, -1)**4.0 * uk_s**4.0
        
        u_g4 = np.array([UnivariateSpline(k_x, u_g_prod4[m,:], s=0, ext=0) for m in range(len(m_x))])
        u_g_new4 = np.array([u_g4[m](krange) for m in range(len(m_x))])
        
        norm_g = ngal
    
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_g_new4[:,i]
                vec2 = u_g_new4[:,j]
                integ = mass_func.dndm * (vec1 * vec2)**0.5 * poisson(4)
                trispec_1h[i,j] = trapz(integ, m_x) / (norm_g*norm_g*norm_g*norm_g)

    if x == 'mmmm':
        u_m_prod = expand_dims(m_x, -1) * uk_c
        
        u_m = np.array([UnivariateSpline(k_x, u_m_prod[m,:], s=0, ext=0) for m in range(len(m_x))])
        u_m_new = np.array([u_m[m](krange) for m in range(len(m_x))])
        
        norm_m = rho_bg
    
        for i, k1 in enumerate(krange):
            for j, k2 in enumerate(krange):
                vec1 = u_m_new[:,i] * u_m_new[:,i]
                vec2 = u_m_new[:,j] * u_m_new[:,j]
                integ = mass_func.dndm * vec1 * vec2
                trispec_1h[i,j] = trapz(integ, m_x) / (norm_m*norm_m*norm_m*norm_m)

    trispec_1h_interp = RectBivariateSpline(krange, krange, trispec_1h, kx=1, ky=1)
    #trispec_1h_interp = interp2d(krange, krange, trispec_1h)
    return trispec_1h_interp


def halo_model_integrals(dndm, uk_c, uk_s, bias, rho_bg, ngal, population_cen, population_sat, Mh, x):
    
    if x == 'g':
        integ1 = expand_dims(dndm * bias, -1) * (expand_dims(population_cen, -1) + expand_dims(population_sat, -1) * uk_s)
        I = trapz(integ1, Mh, axis=0)/ngal
    
    if x == 'm':
        rho_bg = expand_dims(rho_bg, -1)
        integ2 = expand_dims(dndm * bias * Mh, -1) * uk_c
        I = trapz(integ2, Mh, axis=0)/rho_bg

    if x == 'gg':
        integ3 = expand_dims(dndm * bias, -1) * (2.0 * expand_dims(population_cen * population_sat, -1) * uk_s + expand_dims(population_sat**2.0, -1) * uk_s**2.0)
        I = trapz(integ3, Mh, axis=0)/(ngal**2.0)

    if x == 'gm':
        rho_bg = expand_dims(rho_bg, -1)
        integ4 = expand_dims(dndm * bias * Mh, -1) * uk_c * (expand_dims(population_cen, -1) + expand_dims(population_sat, -1) * uk_s)
        I = trapz(integ4, Mh, axis=0)/(rho_bg*ngal)

    if x == 'mm':
        rho_bg = expand_dims(rho_bg, -1)
        integ5 = expand_dims(dndm * bias * Mh**2.0, -1) * uk_c**2.0
        I = trapz(integ5, Mh, axis=0)/(rho_bg**2.0)

    if x == 'mmm':
        rho_bg = expand_dims(rho_bg, -1)
        integ6 = expand_dims(dndm * bias * Mh**3.0, -1) * uk_c**3.0
        I = trapz(integ6, Mh, axis=0)/(rho_bg**3.0)

    return I
    
    
def count_bispectrum(hmf, uk_c, bias, rho_bg, ngal, mlf, Mh):
    dndm = hmf.dndm
    rho_bg = expand_dims(rho_bg, -1)
    term1 = expand_dims(dndm * expand_dims(mlf, -1) * Mh**2.0, -1) * uk_c**2.0
    term2 = expand_dims(dndm * bias * Mh, -1) * uk_c
    term3 = expand_dims(dndm * bias * expand_dims(mlf, -1) * Mh, -1) * uk_c
    
    #print(term1.shape)
    #print(term2.shape)
    #print(term3.shape)
    
    I1 = trapz(term1, Mh, axis=1)/(rho_bg**2.0)
    I2 = trapz(term2, Mh, axis=0) * trapz(term3, Mh, axis=1)/(rho_bg**2.0)

    return I1 + 2.0 * hmf.power * I2


def calc_cov_non_gauss(params):
    
    b_i, b_j, i, j, radius_1, radius_2, T1h, T234h, b_g, area_norm_term, volume, rho_bg, ingredient, idx_1, idx_2, size_1, size_2, covar = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    bg_i = b_g[idx_1][b_i][0]
    bg_j = b_g[idx_2][b_j][0]
    rho_i = rho_bg[idx_1]
    rho_j = rho_bg[idx_2]
    
    #delta = np.eye(b_g.size)
    
    lnk, dlnk = k_adaptive(r_i, r_j, limit=1000)
    
    if covar['healpy']:
        Awr_i = area_norm_term
        Awr_j = area_norm_term
        Aw_rr = area_norm_term
    else:
        Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
        Awr_j = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_j)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
        Aw_rr = simps(np.exp(lnk)**2.0 * (sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j))/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
    
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


def calc_cov_ssc(params):
    
    b_i, b_j, i, j, radius_1, radius_2, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, area_norm_term, b_g, survey_var, rho_bg, ingredient, idx_1, idx_2, size_1, size_2, covar = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    bg_i = b_g[idx_1][b_i][0]
    bg_j = b_g[idx_2][b_j][0]
    rho_i = rho_bg[idx_1]
    rho_j = rho_bg[idx_2]
    survey_var_i = survey_var[idx_1]
    survey_var_j = survey_var[idx_2]
    
    #delta = np.eye(b_g.size)
    
    lnk, dlnk = k_adaptive(r_i, r_j)
    
    if covar['healpy']:
        Awr_i = area_norm_term
        Awr_j = area_norm_term
        Aw_rr = area_norm_term
    else:
        Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
        Awr_j = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_j)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
        Aw_rr = simps(np.exp(lnk)**2.0 * (sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j))/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
    
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


def calc_cov_gauss(params):
    
    b_i, b_j, i, j, radius_1, radius_2, Pgm, Pgg, Pmm, area_norm_term, W_p, Pi_max, shape_noise, ngal, rho_bg, ingredient, subtract_randoms, idx_1, idx_2, size_1, size_2, covar = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    shape_noise_i = shape_noise[idx_1]
    shape_noise_j = shape_noise[idx_2]
    ngal_i = ngal[idx_1]
    ngal_j = ngal[idx_2]
    rho_i = rho_bg[idx_1]
    rho_j = rho_bg[idx_2]
    delta = np.eye(len(radius_1), len(radius_2))
    
    lnk, dlnk = k_adaptive(r_i, r_j)
    
    if covar['healpy']:
        Awr_i = area_norm_term
        Awr_j = area_norm_term
        Aw_rr = area_norm_term
    else:
        Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
        Awr_j = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_j)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
        Aw_rr = simps(np.exp(lnk)**2.0 * (sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j))/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
                
    P_gm_i = np.exp(Pgm[idx_1][b_i](lnk))
    P_gm_j = np.exp(Pgm[idx_2][b_j](lnk))
                    
    P_gg_i = np.exp(Pgg[idx_1][b_i](lnk))
    P_gg_j = np.exp(Pgg[idx_2][b_j](lnk))
                    
    P_mm_i = np.exp(Pmm[idx_1][b_i](lnk))
    P_mm_j = np.exp(Pmm[idx_2][b_j](lnk))
                    
                    
    # wp
    if ingredient == 'gg':
        integ1 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * (P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j])*(P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j])
        val1 = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * 2.0 * simps(integ1, dx=dlnk)
        if subtract_randoms == 'False':
            integ2 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(0, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j]))*np.sqrt((P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j]))
            val2 = ((2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * 4.0 * simps(integ2, dx=dlnk)
        else:
            val2 = 0.0
        val = val1 + val2
    
                    
    # ESD
    if ingredient == 'gm':
        integ3 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.sqrt(P_mm_i + shape_noise_i[b_i]* delta[b_i, b_j])*np.sqrt(P_mm_j + shape_noise_j[b_j]* delta[b_i, b_j])) * (np.sqrt(P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j])*np.sqrt(P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j])) + (P_gm_i*P_gm_j) )
        val1 = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ3, dx=dlnk)
        if subtract_randoms == 'False':
            integ4 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * (np.sqrt(P_mm_i + shape_noise_i[b_i]* delta[b_i, b_j])*np.sqrt(P_mm_j + shape_noise_j[b_j]* delta[b_i, b_j]))
            val2 = ((2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ4, dx=dlnk)
        else:
            val2 = 0.0
        val = (val1 + val2) * rho_i[b_i]*rho_j[b_j] / 1e24
    
    
    # cross
    if ingredient == 'cross':
        integ5 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (P_gm_i * (P_gg_i + 1.0/ngal_i[b_i]*delta[b_i, b_j])) + (P_gm_j  * (P_gg_j + 1.0/ngal_j[b_j]*delta[b_i, b_j])) )
        val1 = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk)
        if subtract_randoms == 'False':
            integ6 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((P_gg_i + 1.0/ngal_i[b_i]* delta[b_i, b_j]))*np.sqrt((P_gg_j + 1.0/ngal_j[b_j]* delta[b_i, b_j]))
            val2 = (2.0*(2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk)
        else:
            val2 = 0.0
            #integ7 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt(P_gm_i*P_gm_j)
            #val = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk) + (2.0*(2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk) + (2.0*(2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ7, dx=dlnk) * np.sqrt(rho_i[b_i]*rho_j[b_j]) / 1e12
        val = (val1 + val2) * np.sqrt(rho_i[b_i]*rho_j[b_j]) / 1e12
    
    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val
    
    
def calc_cov_mlf_sn(params):
    
    b_i, b_j, i, j, radius_1, radius_2, vmax, m_bin, mlf_func, size_1, size_2, covar = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    delta_bin = np.eye(len(radius_1), len(radius_2))
    delta_r = np.eye(len(radius_1[b_i]), len(radius_2[b_j]))

    val = delta_bin[b_i,b_j] * delta_r[i,j] * mlf_func[b_i][i] / (m_bin[b_i] * vmax[b_i][i])
    
    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val * (r_i * r_j)
    
    
def calc_cov_mlf_ssc(params):

    b_i, b_j, i, j, radius_1, radius_2, vmax, m_bin, area_norm_term, mlf_til, survey_var, size_1, size_2, covar = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    val = area_norm_term**2.0 / (vmax[b_i][i] * vmax[b_j][j]) * (np.sqrt(survey_var[b_i])*np.sqrt(survey_var[b_j])) * mlf_til[b_i][i] * mlf_til[b_j][j]

    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val * (r_i * r_j)
    
    
def calc_cov_mlf_cross_sn(params):
    
    # radius 1 here should be for the 2-point function, radius 2 for 1-point function. Same goes for i: 2-point, j: 1-point

    b_i, b_j, i, j, radius_1, radius_2, area_norm_term, count_b, rho_bg, ingredient, idx_1, idx_2, size_1, size_2, covar = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]

    lnk, dlnk = k_adaptive(r_i, r_i)
    
    if covar['healpy']:
        Awr_i = area_norm_term
    else:
        Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)

    count_b_j = count_b[b_j][j](lnk)
    rho_i = rho_bg[idx_1]
    
    # wp
    if ingredient == 'gg':
        integ1 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * count_b_j
        val = (1.0/Awr_i)/(2.0*np.pi) * trapz(integ1, dx=dlnk)
    
    # ESD
    if ingredient == 'gm':
        integ2 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * count_b_j
        val = (1.0/Awr_i)/(2.0*np.pi) * trapz(integ2, dx=dlnk) * rho_i[b_i] / 1e12
    
    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val * r_j
    
    
def calc_cov_mlf_cross_ssc(params):

    # radius 1 here should be for the 2-point function, radius 2 for 1-point function. Same goes for i: 2-point, j: 1-point

    b_i, b_j, i, j, radius_1, radius_2, P_lin, dlnP_lin, Pgm, Pgg, I_g, I_m, I_gg, I_gm, mlf_til, area_norm_term, b_g, survey_var_1, survey_var_2, rho_bg, ingredient, idx_1, idx_2, size_1, size_2, covar = params
    r_i, r_j = radius_1[b_i][i], radius_2[b_j][j]
    
    bg_i = b_g[idx_1][b_i][0]
    rho_i = rho_bg[idx_1]
    survey_var_i = survey_var_1[idx_1]
    survey_var_j = survey_var_2
    mlf_j = mlf_til[b_j][j]
    
    lnk, dlnk = k_adaptive(r_i, r_i)
    
    if covar['healpy']:
        Awr_i = area_norm_term
    else:
        Awr_i = simps(np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i)/(2.0*np.pi) * area_norm_term(np.exp(lnk))**2.0, dx=dlnk)
    
    P_gm_i = Pgm[idx_1][b_i](lnk)
    P_gg_i = Pgg[idx_1][b_i](lnk)
    P_lin_i = P_lin[idx_1][b_i](lnk)
    dP_lin_i = dlnP_lin[idx_1][b_i](lnk)
    Ig_i = I_g[idx_1][b_i](lnk)
    Im_i = I_m[idx_1][b_i](lnk)
    Igg_i = I_gg[idx_1][b_i](lnk)
    Igm_i = I_gm[idx_1][b_i](lnk)
    
    # Responses
    ps_deriv_gg_i = (68.0/21.0 - (1.0/3.0)*(dP_lin_i)) * np.exp(P_lin_i) * np.exp(Ig_i)*np.exp(Ig_i) + np.exp(Igg_i) - 2.0 * bg_i * np.exp(P_gg_i)
    
    ps_deriv_gm_i = (68.0/21.0 - (1.0/3.0)*(dP_lin_i)) * np.exp(P_lin_i) * np.exp(Ig_i)*np.exp(Im_i) + np.exp(Igm_i) - bg_i * np.exp(P_gm_i)
    
    # wp
    if ingredient == 'gg':
        integ1 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * (np.sqrt(survey_var_i[b_i] * survey_var_j[b_j])) * ps_deriv_gg_i * mlf_j
        val = (1.0/Awr_i)/(2.0*np.pi) * trapz(integ1, dx=dlnk)
    
    # ESD
    if ingredient == 'gm':
        integ2 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * (np.sqrt(survey_var_i[b_i] * survey_var_j[b_j])) * ps_deriv_gm_i * mlf_j
        val = (1.0/Awr_i)/(2.0*np.pi) * trapz(integ2, dx=dlnk) * rho_i[b_i] / 1e12
    # Area normalisation is not yet clear here!
    return size_1[:b_i].sum()+i,size_2[:b_j].sum()+j, val * r_j


def parallelise(func, nproc, cov_out, radius_1, radius_2, *args):

    paramlist = []
    for a in range(len(radius_1)):
        for b in range(len(radius_2)):
            for c in range(len(radius_1[a])):
                for d in range(len(radius_2[b])):
                    paramlist.append([a, b, c, d])

    for p in paramlist:
        p.extend([radius_1, radius_2, *args])
        
    pool = multi.Pool(processes=nproc, maxtasksperchild=10)
    for i, j, val in pool.map(func, paramlist):
        cov_out[i,j] = val

    return cov_out
    
    
def k_adaptive(r_i, r_j, limit=None):

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
    nk = np.int(np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * np.sqrt(r_i*r_j))))))
    if limit is not None:
        if nk > limit:
            nk = limit
    
    lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
    
    return lnk, dlnk
    

def format_z(z, nbins):
    # if a single value is given for more than one bin, assign same
    # value to all bins
    if z.size == 1 and nbins > 1:
        z = z*np.ones(nbins)
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    #z = expand_dims(z, -1) # no expand dims in cov!
    return z
    

def populations(observables, ingredients, completeness, mass_range, theta, nbins):
    pop_c, pop_s = np.zeros((2,nbins,mass_range.size))

    c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta[1:]

    if ingredients['centrals']:
        pop_c, prob_c = hod.number(
            observables.sampling, mass_range, c_mor[0], c_scatter[0],
            c_mor[1:], c_scatter[1:], np.ones(observables.sampling.shape),
            obs_is_log=observables.gm.is_log)

    if ingredients['satellites']:
        pop_s, prob_s = hod.number(
            observables.sampling, mass_range, s_mor[0], s_scatter[0],
            s_mor[1:], s_scatter[1:], np.ones(observables.sampling.shape),
            obs_is_log=observables.gm.is_log)
                
    return pop_c, pop_s
    
    
def calculate_uk(setup, observables, ingredients, z, mass_range, rho_bg,
                 c_concentration, s_concentration, c_miscent, nbins):
    rvir_range_lin = virial_radius(mass_range, rho_bg, setup['delta'])
    # Fourier Transform of the NFW profile
    if ingredients['centrals']:
        concentration = c_concentration[0](mass_range, *c_concentration[1:])
        uk_c = nfw.uk(
            setup['k_range_lin'], mass_range, rvir_range_lin,
                concentration, rho_bg,
            setup['delta'])
        uk_c = uk_c / expand_dims(uk_c[...,0], -1)
    else:
        uk_c = np.ones((nbins,mass_range.size,setup['k_range_lin'].size))
    # and of the NFW profile of the satellites
    if ingredients['satellites']:
        concentration_sat = s_concentration[0](
            mass_range, *s_concentration[1:])
        uk_s = nfw.uk(
            setup['k_range_lin'], mass_range, rvir_range_lin,
                concentration_sat, rho_bg,
            setup['delta'])
        uk_s = uk_s / expand_dims(uk_s[...,0], -1)
    else:
        uk_s = np.ones((nbins,mass_range.size,setup['k_range_lin'].size))

    return uk_c, uk_s
    
    
def preamble(theta):
    """Preamble function

    This function is specified separately in the configuration file
    and is called only once when initializing the sampler module,
    rather than at every step in the MCMC. Include here all variable
    tests, for instance.

    This function does not return anything
    """
    np.seterr(
        divide='ignore', over='ignore', under='ignore', invalid='ignore')

    observables, selection, ingredients, params, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]

    R_units = ('pc','kpc','Mpc')
    if setup['R_unit'] not in R_units:
        err = f'R_unit must be one of {R_units}'
        raise ValueError(err)

    cosmo = params[0]
    # the order of elements of cosmo is set in
    # helpers.configuration.core.CosmoSection
    z = cosmo[10]

    nbins = observables.nbins
    if observables.mlf:
        nbins = nbins - observables.mlf.nbins

    if setup['backend'] == 'ccl':
        wrn = 'Backend ccl not available in halo.model. Falling back to hmf'
        warnings.warn(wrn)
    if setup['delta_ref'] == 'critical': setup['delta_ref'] = 'SOCritical'
    elif setup['delta_ref'] == 'matter': setup['delta_ref'] = 'SOMean'

    # if a single value is given for more than one bin, assign same
    # value to all bins
    if z.size == 1 and nbins > 1:
        z = z*np.ones(nbins)
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    z = expand_dims(z, -1)
    if ingredients['nzlens']:
        z_shape_test = (nz.shape[1] == nbins)
    else:
        z_shape_test = (z.size == nbins)
    if not z_shape_test:
        raise ValueError(
            'Number of redshift bins should be equal to the number of' \
            ' observable bins!')

    # we might also want to add a function in the configuration
    # functionality to update theta more easily
    theta[1][theta[0].index('setup')] = setup

    return theta


def covariance(theta, R):
    
    # ideally we would move this to somewhere separate later on
    #preamble(theta)
    
    np.seterr(
        divide='ignore', over='ignore', under='ignore', invalid='ignore')

    # this has to happen before because theta is re-purposed below
    # this is always true here, though
    covar = theta[1][theta[0].index('covariance')]

    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
            for name in ('observables', 'selection', 'ingredients',
                    'parameters', 'setup')]

    if observables.mlf:
        nbins = observables.nbins - observables.mlf.nbins
    else:
        nbins = observables.nbins
    output = np.empty(observables.nbins, dtype=object)

    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta

    #cosmo_model, sigma8, n_s, z = load_cosmology(cosmo)
    Om0, Ob0, h, sigma8, n_s, m_nu, Neff, w0, wa, Tcmb0, z, nz, z_mlf, zs \
        = cosmo

    cosmo_model = Flatw0waCDM(
        H0=100*h, Ob0=Ob0, Om0=Om0, Tcmb0=Tcmb0, m_nu=m_nu*eV,
        Neff=Neff, w0=w0, wa=wa)

    z = format_z(z, nbins)
    
        # Tinker10 should also be read from theta!
    hmf, rho_bg, nu, nu_s, fsigma_s = load_hmf_cov(z, setup, cosmo_model, sigma8, n_s)

    assert np.allclose(setup['mass_range'], hmf[0].m)
    # alias (should probably get rid of it)
    mass_range = setup['mass_range']


    
    # Remove, for testing purposes
    """
    print('Using test radii')
    if observables.gm:
        observables.gm.R = [logspace(-2, np.log10(30), 20, endpoint=True) for r in R[observables.gm.idx]] # for testing
        observables.gm.size = np.array([len(r) for r in observables.gm.R])
    if observables.gg:
        observables.gg.R = [logspace(-2, np.log10(30), 20, endpoint=True) for r in R[observables.gg.idx]] # for testing
        observables.gg.size = np.array([len(r) for r in observables.gg.R])
    if observables.mlf:
        observables.mlf.R = [logspace(7, 13, 30, endpoint=True) for r in range(2)]#R[idx_mlf]]
        n_bins = 30
        rvir_bins = np.linspace(7, 12.5, n_bins+1, endpoint=True, retstep=True)[0]
        rvir_center = (rvir_bins[1:] + rvir_bins[:-1])/2.0
        observables.mlf.R = [10.0**rvir_center for r in range(2)]
        observables.mlf.size = np.array([len(r) for r in observables.mlf.R])
    #"""


    print('Setting survey and observational details...')
    
    Pi_max = covar['pi_max'] # in Mpc/h
    eff_density = covar['eff_density'] # as defined in KiDS (gal / arcmin^2)
    kids_variance_squared = covar['variance_squared'] # as defined in KiDS papers
    z_kids = covar['mean_survey_redshift']
    gauss = covar['gauss']
    non_gauss = covar['non_gauss']
    ssc = covar['ssc']
    cross = covar['cross']
    cross_mlf = covar['cross_mlf']
    subtract_randoms = covar['subtract_randoms'] #False # if randoms are not subtracted, this will increase the error bars
    nproc = covar['threads'] #4
    
    # This deals with KiDS specific sigma crit parameters
    z_epsilon = covar['z_epsilon']
    z_max = covar['z_max']
    lens_photoz = covar['lens_photoz']
    galsigma = covar['lens_photoz_sigma']
    lens_pz_redshift = covar['lens_photoz_zdep']
    
    if observables.mlf:
        if covar['vmax_file'] == 'None':
            raise ValueError(
                'When calculating the SMF/LF covariance matrix, the Vmax file is needed.' /
                'It should be a text file with 2 columns, first column the log10 of stellar mass/luminosity'/
                'the second one Vmax at that stellar mass/luminosity.')


    # Calculate critical surface density, either simply or for KiDS
    if covar['kids_sigma_crit']:
        print('\t...using KiDS specific sigma_crit setup.')
        # KiDS specific sigma_crit, accounting for n(z)!
        spec_z_path = covar['specz_file']
        sigma_crit = sigma_crit_kids(hmf, z, z_epsilon, z_max, lens_photoz, galsigma, lens_pz_redshift, spec_z_path) * hmf[0].cosmo.h * 10.0**12.0
    else:
        sigma_crit = sigma_crit_func(cosmo_model, z, z_kids)
    
    eff_density_in_rad = eff_density * (10800.0/np.pi)**2.0 # convert to radians
    
    Pi_max_lens = hmf[0].cosmo.angular_diameter_distance(z_kids).value
    rho_sc = rho_bg[...,0]
    shape_noise = ((sigma_crit / rho_sc)**2.0) * (kids_variance_squared / eff_density_in_rad)  * ((hmf[0].cosmo.angular_diameter_distance(z).value)**2.0 / Pi_max_lens) # With lensing the projected distance is the distance between the observer and effective survey redshift.
    
    
    # Setup area and survey variance
    if covar['healpy']:
        healpy_data = covar['healpy_data']
        print('\t...using healpy map to determine survey variance and area.')
        l_range, alms, survey_area = calculate_alms(healpy_data, 1)
        survey_var = [survey_variance_from_healpy(hmf_i, setup['k_range'], z_kids, l_range, alms, survey_area) for hmf_i in hmf]
        kids_area = survey_area * 3600.0 # to arcmin^2
        area_norm_term = kids_area * (((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) * hmf[0].cosmo.h)**2.0
        area_sur = area_norm_term
        radius = np.sqrt(area_norm_term / np.pi)
        vol = radius**2.0 * Pi_max
        W = 2.0*np.pi*radius**2.0 * sp.jv(1, setup['k_range_lin']*radius) / (setup['k_range_lin']*radius)
        W_p = UnivariateSpline(setup['k_range_lin'], W, s=0, ext=0)
    else:
        kids_area = covar['area'] # in deg^2
        kids_area = kids_area * 3600.0 # to arcmin^2
        area_sur = kids_area * (((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) * hmf[0].cosmo.h)**2.0
        radius = np.sqrt(kids_area / np.pi) * ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) * hmf[0].cosmo.h
        vol = radius**2.0 * Pi_max
        W = 2.0*np.pi*radius**2.0 * sp.jv(1, setup['k_range_lin']*radius) / (setup['k_range_lin']*radius)
        W_p = UnivariateSpline(setup['k_range_lin'], W, s=0, ext=0)
        survey_var = [survey_variance(hmf_i, W_p, setup['k_range'], radius) for hmf_i in hmf]
        area_norm_term = W_p
    
    
    # Calculating halo model
    
    completeness = np.ones(observables.sampling.shape)
    pop_c, pop_s = populations(observables,
        ingredients, completeness, mass_range, theta, nbins)
    pop_g = pop_c + pop_s

    # note that pop_g already accounts for incompleteness
    dndm = array([hmf_i.dndm for hmf_i in hmf])
    power = array([hmf_i.power  for hmf_i in hmf])
    ngal = hod.nbar(dndm, pop_g, mass_range)
    meff = hod.Mh_effective(
        dndm, pop_g, mass_range, return_log=observables.gm.is_log)
        
    
    uk_c, uk_s = calculate_uk(
        setup, observables, ingredients, z, mass_range, rho_bg,
        c_concentration, s_concentration, c_miscent, nbins)
    
    
    # Luminosity or mass function as an output:
    if observables.mlf:
        print('Calculating stellar mass function terms...')
        # Needs independent redshift input!
        #z_mlf = z_in[idx_mlf]
        if z_mlf.size == 1 and observables.mlf.nbins > 1:
            z_mlf = z_mlf*np.ones(observables.mlf.nbins)
        if z_mlf.size != observables.mlf.nbins:
            raise ValueError(
                'Number of redshift bins should be equal to the number of observable bins!')
        hmf_mlf, _rho_mean, _nu, _nu_s, _fsigma_s = load_hmf_cov(z_mlf, setup, cosmo_model, sigma8, n_s)
        dndm_mlf = array([hmf_i.dndm for hmf_i in hmf_mlf])
        bias_mlf = array([bias_tinker10(nu_i, nus_i, fsigma_i, setup) for nu_i, nus_i, fsigma_i in zip(_nu, _nu_s, _fsigma_s)])
        
        pop_c_mlf = np.zeros(observables.mlf.sampling.shape)
        pop_s_mlf = np.zeros(observables.mlf.sampling.shape)
        pop_c_mlf_til = np.zeros(observables.mlf.sampling.shape)
        pop_s_mlf_til = np.zeros(observables.mlf.sampling.shape)
        
        if ingredients['centrals']:
            pop_c_mlf = hod.mlf(
                observables.mlf.sampling, dndm_mlf, mass_range, c_mor[0], c_scatter[0],
                c_mor[1:], c_scatter[1:],
                obs_is_log=observables.mlf.is_log)
            pop_c_mlf_til = hod.mlf_tilde(
                observables.mlf.sampling, dndm_mlf, bias_mlf, mass_range, c_mor[0], c_scatter[0],
                c_mor[1:], c_scatter[1:],
                obs_is_log=observables.mlf.is_log)

        if ingredients['satellites']:
            pop_s_mlf = hod.mlf(
                observables.mlf.sampling, dndm_mlf, mass_range, s_mor[0], s_scatter[0],
                s_mor[1:], s_scatter[1:],
                obs_is_log=observables.mlf.is_log)
            pop_s_mlf_til = hod.mlf_tilde(
                observables.mlf.sampling, dndm_mlf, bias_mlf, mass_range, s_mor[0], s_scatter[0],
                s_mor[1:], s_scatter[1:],
                obs_is_log=observables.mlf.is_log)
        pop_g_mlf = pop_c_mlf + pop_s_mlf
        pop_g_mlf_til = pop_c_mlf_til + pop_s_mlf_til
        
        mlf_inter = [UnivariateSpline(hod_i, np.log(ngal_i), s=0, ext=0)
                    for hod_i, ngal_i in zip(observables.mlf.sampling, pop_g_mlf)]
        mlf_tilde_inter = [UnivariateSpline(hod_i, np.log(ngal_i), s=0, ext=0)
                    for hod_i, ngal_i in zip(observables.mlf.sampling, pop_g_mlf_til)]
        for i,Ri in enumerate(observables.mlf.R):
            Ri = Quantity(Ri, unit='Mpc')
            observables.mlf.R[i] = Ri.to(setup['R_unit']).value
        mlf_out = [exp(mlf_i(np.log10(r_i))) for mlf_i, r_i
                    in zip(mlf_inter, observables.mlf.R)]
        mlf_til = [exp(mlf_i(np.log10(r_i))) for mlf_i, r_i
                    in zip(mlf_tilde_inter, observables.mlf.R)]
        m_bin = [np.diff(i)[0] for i in observables.mlf.R]
        count_b = array([count_bispectrum(hmf_i, uk_c_i, bias_tinker10(nu_i, nus_i, fsigma_i, setup), rho_bg_i, ngal_i, mlf_out_i, mass_range)
                for hmf_i, uk_c_i, rho_bg_i, ngal_i, mlf_out_i, nu_i, nus_i, fsigma_i in
                zip(hmf, uk_c, rho_bg, ngal, mlf_out, nu, nu_s, fsigma_s)])
        count_b_interp = [[UnivariateSpline(setup['k_range'], count_b_i, s=0, ext=0)
                    for count_b_i in count] for count in count_b]
        if covar['healpy']:
            survey_var_mlf = [survey_variance_from_healpy(hmf_i, setup['k_range'], z_kids, l_range, alms, survey_area) for hmf_i in hmf_mlf]
        else:
            survey_var_mlf = [survey_variance(hmf_i, W_p, setup['k_range'], radius) for hmf_i in hmf_mlf]
        
        vmax_data = np.genfromtxt(open(covar['vmax_file'], 'rb'), delimiter=None, comments='#')
        M_center = vmax_data[:,0]
        vmax_in = vmax_data[:,1]
        vmax_inter = UnivariateSpline(10.0**M_center, vmax_in, s=0, ext=0)
        vmax = [vmax_inter(i) for i in observables.mlf.R]
    
    
    """
    # Power spectrum
    """
    print('Calculating power spectra...')
    
    # damping of the 1h power spectra at small k
    F_k1 = sp.erf(setup['k_range_lin']/0.1)
    F_k2 = np.ones_like(setup['k_range_lin'])
    
    # Galaxy - dark matter spectra (for lensing)
    bias = c_twohalo
    bias = array([bias]*setup['k_range_lin'].size).T
    
    if setup['delta_ref'] == 'SOCritical':
        bias = bias * cosmo_model.Om0
   
    rho_bg = rho_bg[...,0]
                
    Pgm_2h = F_k2 * bias * array(
            [two_halo_gm(dndm_i, power_i, nu_i, nus_i, fsigma_i, ngal_i, pop_g_i, mass_range, setup)[0]
            for dndm_i, power_i, nu_i, nus_i, fsigma_i, ngal_i, pop_g_i
            in zip(dndm, power, nu, nu_s, fsigma_s, expand_dims(ngal, -1),
                    expand_dims(pop_g, -2))])

    bias_num = bias * array(
            [two_halo_gm(dndm_i, power_i, nu_i, nus_i, fsigma_i, ngal_i, pop_g_i, mass_range, setup)[1]
            for dndm_i, power_i, nu_i, nus_i, fsigma_i, ngal_i, pop_g_i
            in zip(dndm, power, nu, nu_s, fsigma_s, expand_dims(ngal, -1),
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
            [two_halo_gg(dndm_i, power_i, nu_i, nus_i, fsigma_i, ngal_i, pop_g_i, mass_range, setup)[0]
            for dndm_i, power_i, nu_i, nus_i, fsigma_i, ngal_i, pop_g_i
            in zip(dndm, power, nu, nu_s, fsigma_s, expand_dims(ngal, -1),
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



    P_gm_func = [UnivariateSpline(setup['k_range'], np.log(Pgm_k_i), s=0, ext=0)
                    for Pgm_k_i in Pgm_k]
        
    P_gg_func = [UnivariateSpline(setup['k_range'], np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in Pgg_k]

    P_mm_func = [UnivariateSpline(setup['k_range'], np.log(Pmm_k_i), s=0, ext=0)
                    for Pmm_k_i in Pmm_k]
                    
    
    # Evaluate halo model integrals needed for SSC
    
    I_g = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(nu_i, nus_i, fsigma_i, setup), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'g')
                                   for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, nu_i, nus_i, fsigma_i in
                                   zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s, nu, nu_s, fsigma_s)])
                                   
    I_m = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(nu_i, nus_i, fsigma_i, setup), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'm')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, nu_i, nus_i, fsigma_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s, nu, nu_s, fsigma_s)])
                                    
    I_gg = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(nu_i, nus_i, fsigma_i, setup), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'gg')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, nu_i, nus_i, fsigma_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s, nu, nu_s, fsigma_s)])
                                    
    I_gm = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(nu_i, nus_i, fsigma_i, setup), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'gm')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, nu_i, nus_i, fsigma_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s, nu, nu_s, fsigma_s)])
                                    
    I_mm = array([halo_model_integrals(hmf_i.dndm, uk_c_i, uk_s_i, bias_tinker10(nu_i, nus_i, fsigma_i, setup), rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, 'mm')
                                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, nu_i, nus_i, fsigma_i in
                                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s, nu, nu_s, fsigma_s)])
    
    I_g_func = [UnivariateSpline(setup['k_range'], np.log(I_g_i), s=0, ext=0)
               for I_g_i in I_g]
               
    I_inter_m = [UnivariateSpline(setup['k_range'], np.log(I_m_i), s=0, ext=0)
                for I_m_i in I_m]
                 
    I_gg_func = [UnivariateSpline(setup['k_range'], np.log(I_gg_i), s=0, ext=0)
                for I_gg_i in I_gg]
                 
    I_gm_func = [UnivariateSpline(setup['k_range'], np.log(I_gm_i), s=0, ext=0)
                for I_gm_i in I_gm]
                
    I_mm_func = [UnivariateSpline(setup['k_range'], np.log(I_mm_i), s=0, ext=0)
                for I_mm_i in I_mm]
    
    P_lin_inter = [UnivariateSpline(setup['k_range'], np.log(hmf_i.power), s=0, ext=0)
                for hmf_i in hmf]
               
    k3P_lin_inter = [UnivariateSpline(setup['k_range'], np.log(setup['k_range_lin']**3.0 * hmf_i.power), s=0, ext=0)
                for hmf_i in hmf]
                
    dlnk3P_lin_interdlnk = [f.derivative() for f in k3P_lin_inter]

    print('Halo integrals done.')
    
    # Start covariance calculations
    
    # Setting limited k-range for covariance matrix estimation.
    #lnk_min, lnk_max = np.log(0.01), np.log(1000.0)
    lnk_min, lnk_max = np.log(1e-4), np.log(1e4)
    k_temp = np.linspace(lnk_min, lnk_max, 100, endpoint=True)
    k_temp_lin = np.exp(k_temp)
  
    if non_gauss:
        Tgggg = array([trispectra_1h(k_temp_lin, hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, setup['k_range_lin'], 'gggg')
                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
    
        Tgggm = array([trispectra_1h(k_temp_lin, hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, setup['k_range_lin'], 'gggm')
                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
                    
        Tgmgm = array([trispectra_1h(k_temp_lin, hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i, mass_range, setup['k_range_lin'], 'gmgm')
                    for hmf_i, uk_c_i, uk_s_i, rho_bg_i, ngal_i, pop_c_i, pop_s_i in
                    zip(hmf, uk_c, uk_s, rho_bg, ngal, pop_c, pop_s)])
    
        T234h = array([trispectra_234h(k_temp_lin, P_lin_inter_i, hmf_i, u_k_i, bias_tinker10(nu_i, nus_i, fsigma_i, setup), rho_bg_i, mass_range, setup['k_range_lin'])
                    for P_lin_inter_i, hmf_i, u_k_i, rho_bg_i, nu_i, nus_i, fsigma_i in
                    zip(P_lin_inter, hmf, uk_c, rho_bg, nu, nu_s, fsigma_s)])
        print('Trispectra done.')
   
   
    # Initialise output arrays
    if observables.gm:
        cov_esd = np.zeros((observables.gm.size.sum(), observables.gm.size.sum()), dtype=np.float64)
        cov_esd_tot = cov_esd.copy()
    if observables.gg:
        cov_wp = np.zeros((observables.gg.size.sum(), observables.gg.size.sum()), dtype=np.float64)
        cov_wp_tot = cov_wp.copy()
    if observables.gm and observables.gg:
        cov_cross = np.zeros((observables.gg.size.sum(), observables.gm.size.sum()), dtype=np.float64)
        cov_cross_tot = cov_cross.copy()
    if observables.mlf:
        cov_mlf = np.zeros((observables.mlf.size.sum(), observables.mlf.size.sum()), dtype=np.float64)
        cov_mlf_tot = cov_mlf.copy()
    if observables.mlf and observables.gm:
        cov_mlf_cross_gm = np.zeros((observables.gm.size.sum(), observables.mlf.size.sum()), dtype=np.float64)
        cov_mlf_cross_gm_tot = cov_mlf_cross_gm.copy()
    if observables.mlf and observables.gg:
        cov_mlf_cross_gg = np.zeros((observables.gg.size.sum(), observables.mlf.size.sum()), dtype=np.float64)
        cov_mlf_cross_gg_tot = cov_mlf_cross_gg.copy()
    
   
    # Calculate the requested terms and combinations
    if gauss:
        print('Calculating the Gaussian part of the covariance...')
        if observables.gm:
            cov_esd_gauss = parallelise(calc_cov_gauss, nproc, cov_esd.copy(), observables.gm.R, observables.gm.R, P_gm_func, P_gg_func, P_mm_func, area_norm_term, W_p, Pi_max_lens, shape_noise, ngal, rho_bg, 'gm', subtract_randoms, observables.gm.idx, observables.gm.idx, observables.gm.size, observables.gm.size, covar)
            cov_esd_tot += cov_esd_gauss
        if observables.gg:
            cov_wp_gauss = parallelise(calc_cov_gauss, nproc, cov_wp.copy(), observables.gg.R, observables.gg.R, P_gm_func, P_gg_func, P_mm_func, area_norm_term, W_p, Pi_max, shape_noise, ngal, rho_bg, 'gg', subtract_randoms, observables.gg.idx, observables.gg.idx, observables.gg.size, observables.gg.size, covar)
            cov_wp_tot += cov_wp_gauss
        if observables.gm and observables.gg and cross:
            cov_cross_gauss = parallelise(calc_cov_gauss, nproc, cov_cross.copy(), observables.gg.R, observables.gm.R, P_gm_func, P_gg_func, P_mm_func, area_norm_term, W_p, Pi_max, shape_noise, ngal, rho_bg, 'cross', subtract_randoms, observables.gg.idx, observables.gm.idx, observables.gg.size, observables.gm.size, covar)
            cov_cross_tot += cov_cross_gauss
        
        if observables.mlf:
            print('Calculating the Gaussian part of the SMF/LF covariance...')
            cov_mlf_gauss = parallelise(calc_cov_mlf_sn, nproc, cov_mlf.copy(), observables.mlf.R, observables.mlf.R, vmax, m_bin, mlf_out, observables.mlf.size, observables.mlf.size, covar)
            cov_mlf_tot += np.log(10)**2.0 * cov_mlf_gauss
        if observables.mlf and observables.gm and cross_mlf:
            cov_mlf_cross_gauss_gm = parallelise(calc_cov_mlf_cross_sn, nproc, cov_mlf_cross_gm.copy(), observables.gm.R, observables.mlf.R, area_norm_term, count_b_interp, rho_bg, 'gm', observables.gm.idx, observables.mlf.idx, observables.gm.size, observables.mlf.size, covar)
            cov_mlf_cross_gm_tot += np.log(10) * cov_mlf_cross_gauss_gm
        if observables.mlf and observables.gg and cross_mlf:
            cov_mlf_cross_gauss_gg = parallelise(calc_cov_mlf_cross_sn, nproc, cov_mlf_cross_gg.copy(), observables.gg.R, observables.mlf.R, area_norm_term, count_b_interp, rho_bg, 'gg', observables.gg.idx, observables.mlf.idx, observables.gg.size, observables.mlf.size, covar)
            cov_mlf_cross_gg_tot += np.log(10) * cov_mlf_cross_gauss_gg

    if ssc:
        print('Calculating the super-sample covariance...')
        if observables.gm:
            cov_esd_ssc = parallelise(calc_cov_ssc, nproc, cov_esd.copy(), observables.gm.R, observables.gm.R, P_lin_inter, dlnk3P_lin_interdlnk, P_gm_func, P_gg_func, I_g_func, I_inter_m, I_gg_func, I_gm_func, area_norm_term, bias_num, survey_var, rho_bg, 'gm', observables.gm.idx, observables.gm.idx, observables.gm.size, observables.gm.size, covar)
            cov_esd_tot += cov_esd_ssc
        if observables.gg:
            cov_wp_ssc = parallelise(calc_cov_ssc, nproc, cov_wp.copy(), observables.gg.R, observables.gg.R, P_lin_inter, dlnk3P_lin_interdlnk, P_gm_func, P_gg_func, I_g_func, I_inter_m, I_gg_func, I_gm_func, area_norm_term, bias_num, survey_var, rho_bg, 'gg', observables.gg.idx, observables.gg.idx, observables.gg.size, observables.gg.size, covar)
            cov_wp_tot += cov_wp_ssc
        if observables.gm and observables.gg and cross:
            cov_cross_ssc = parallelise(calc_cov_ssc, nproc, cov_cross.copy(), observables.gg.R, observables.gm.R, P_lin_inter, dlnk3P_lin_interdlnk, P_gm_func, P_gg_func, I_g_func, I_inter_m, I_gg_func, I_gm_func, area_norm_term, bias_num, survey_var, rho_bg, 'cross', observables.gg.idx, observables.gm.idx, observables.gg.size, observables.gm.size, covar)
            cov_cross_tot += cov_cross_ssc
        
        if observables.mlf:
            print('Calculating the SMF/LF super-sample covariance...')
            cov_mlf_ssc = parallelise(calc_cov_mlf_ssc, nproc, cov_mlf.copy(), observables.mlf.R, observables.mlf.R, vmax, m_bin, area_sur, mlf_til, survey_var_mlf, observables.mlf.size, observables.mlf.size, covar)
            cov_mlf_tot += np.log(10)**2.0 * cov_mlf_ssc
        if observables.mlf and observables.gm and cross_mlf:
            cov_mlf_cross_ssc_gm = parallelise(calc_cov_mlf_cross_ssc, nproc, cov_mlf_cross_gm.copy(), observables.gm.R, observables.mlf.R, P_lin_inter, dlnk3P_lin_interdlnk, P_gm_func, P_gg_func, I_g_func, I_inter_m, I_gg_func, I_gm_func, mlf_til, area_norm_term, bias_num, survey_var, survey_var_mlf, rho_bg, 'gm', observables.gm.idx, observables.mlf.idx, observables.gm.size, observables.mlf.size, covar)
            cov_mlf_cross_gm_tot += np.log(10) * cov_mlf_cross_ssc_gm
        if observables.mlf and observables.gg and cross_mlf:
            cov_mlf_cross_ssc_gg = parallelise(calc_cov_mlf_cross_ssc, nproc, cov_mlf_cross_gg.copy(), observables.gg.R, observables.mlf.R, P_lin_inter, dlnk3P_lin_interdlnk, P_gm_func, P_gg_func, I_g_func, I_inter_m, I_gg_func, I_gm_func, mlf_til, area_norm_term, bias_num, survey_var, survey_var_mlf, rho_bg, 'gg', observables.gg.idx, observables.mlf.idx, observables.gg.size, observables.mlf.size, covar)
            cov_mlf_cross_gg_tot += np.log(10) * cov_mlf_cross_ssc_gg
        
    if non_gauss:
        print('Calculating the connected (non-Gaussian) part of the covariance...')
        if observables.gm:
            cov_esd_non_gauss = parallelise(calc_cov_non_gauss, nproc, cov_esd.copy(), observables.gm.R, observables.gm.R, Tgmgm, T234h, bias_num, area_norm_term, vol, rho_bg, 'gm', observables.gm.idx, observables.gm.idx, observables.gm.size, observables.gm.size, covar)
            cov_esd_tot += cov_esd_non_gauss
        if observables.gg:
            cov_wp_non_gauss = parallelise(calc_cov_non_gauss, nproc, cov_wp.copy(), observables.gg.R, observables.gg.R, Tgggg, T234h, bias_num, area_norm_term, vol, rho_bg, 'gg', observables.gg.idx, observables.gg.idx, observables.gg.size, observables.gg.size, covar)
            cov_wp_tot += cov_wp_non_gauss
        if observables.gm and observables.gg and cross:
            cov_cross_non_gauss = parallelise(calc_cov_non_gauss, nproc, cov_cross.copy(), observables.gg.R, observables.gm.R, Tgggm, T234h, bias_num, area_norm_term, vol, rho_bg, 'cross', observables.gg.idx, observables.gm.idx, observables.gg.size, observables.gm.size, covar)
            cov_cross_tot += cov_cross_non_gauss
            

    # To be removed, only for testing purposes
    """
    rescale = covar['area']/survey_area #1.0#
    aw_values = np.ones((observables.gm.size.sum(), observables.gm.size.sum(), 3), dtype=np.float64)
    
    cov_block = np.block([[cov_esd_tot, cov_cross_tot.T],
                        [cov_cross_tot, cov_wp_tot]])
    
    #cov_esd_gauss, cov_wp_gauss, cov_cross_gauss = np.zeros((observables.gm.size.sum(), observables.gm.size.sum()), dtype=np.float64), np.zeros((observables.gg.size.sum(), observables.gg.size.sum()), dtype=np.float64), np.zeros((observables.gg.size.sum(), observables.gm.size.sum()), dtype=np.float64)
    cov_esd_non_gauss, cov_wp_non_gauss, cov_cross_non_gauss = np.zeros_like(cov_esd_gauss), np.zeros_like(cov_wp_gauss), np.zeros_like(cov_cross_gauss)
    #cov_esd_ssc, cov_wp_ssc, cov_cross_ssc = np.zeros_like(cov_esd_gauss), np.zeros_like(cov_wp_gauss), np.zeros_like(cov_cross_gauss)
    
    all = np.array([observables.gm.size, observables.gg.size, observables.gm.R, observables.gg.R, cov_esd_gauss/rescale, cov_esd_non_gauss/rescale, cov_esd_ssc/rescale, cov_wp_gauss/rescale, cov_wp_non_gauss/rescale, cov_wp_ssc/rescale, cov_esd_tot/rescale, cov_wp_tot/rescale, cov_cross_tot/rescale, cov_block/rescale, aw_values, radius**2.0], dtype=object)
    
    np.save('/net/home/fohlen12/dvornik/test_pipeline2/covariance/cov_all.npy', all)
    
    cov_block2 = np.block([[cov_esd_tot, cov_cross_tot.T, cov_mlf_cross_gm_tot],
                            [cov_cross_tot, cov_wp_tot, cov_mlf_cross_gg_tot],
                            [cov_mlf_cross_gm_tot.T, cov_mlf_cross_gg_tot.T, cov_mlf_tot]])
    
    all2 = np.array([observables.gm.size, observables.gg.size, observables.mlf.size, observables.gm.R, observables.gg.R, observables.mlf.R, cov_esd_gauss/rescale, cov_esd_non_gauss/rescale, cov_esd_ssc/rescale, cov_wp_gauss/rescale, cov_wp_non_gauss/rescale, cov_wp_ssc/rescale, cov_mlf_gauss/rescale, cov_mlf_cross_gauss_gm/rescale, cov_mlf_cross_gauss_gg/rescale, cov_mlf_ssc/rescale, cov_mlf_cross_ssc_gm/rescale, cov_mlf_cross_ssc_gg/rescale, cov_esd_tot/rescale, cov_wp_tot/rescale, cov_cross_tot/rescale, cov_mlf_tot/rescale, cov_mlf_cross_gm_tot/rescale, cov_mlf_cross_gg_tot/rescale, cov_block/rescale, cov_block2/rescale, aw_values, radius**2.0, mlf_out], dtype=object)
    
    np.save('/net/home/fohlen12/dvornik/test_pipeline2/covariance/cov_all_smf.npy', all2)
    
    #"""
    
    
    print('Combining the matrices...')
    # I think this can be simplified a bit :/
    if observables.gm and not (observables.gg or observables.mlf):
        cov_block = cov_esd_tot
    if observables.gg and not (observables.gm or observables.mlf):
        cov_block = cov_wp_tot
    if observables.mlf and not (observables.gm or observables.gg):
        cov_block = cov_mlf_tot
    if observables.gm and observables.gg and not observables.mlf:
        cov_block = np.block([[cov_esd_tot, cov_cross_tot.T],
                              [cov_cross_tot, cov_wp_tot]])
    if observables.gm and observables.mlf and not observables.gg:
        cov_block = np.block([[cov_esd_tot, cov_mlf_cross_gm_tot],
                              [cov_mlf_cross_gm_tot.T, cov_mlf_tot]])
    if observables.gg and observables.mlf and not observables.gm:
        cov_block = np.block([[cov_wp_tot, cov_mlf_cross_gg_tot],
                              [cov_mlf_cross_gg_tot.T, cov_mlf_tot]])
    if observables.gm and observables.gg and observables.mlf:
        cov_block = np.block([[cov_esd_tot, cov_cross_tot.T, cov_mlf_cross_gm_tot],
                            [cov_cross_tot, cov_wp_tot, cov_mlf_cross_gg_tot],
                            [cov_mlf_cross_gm_tot.T, cov_mlf_cross_gg_tot.T, cov_mlf_tot]])
    detC = np.linalg.det(cov_block)
    if detC < 0:
        print('WARNING: Determinant of the covariance is negative!')
    if np.isnan(detC):
        print('WARNING: At least one entry of the covariace matrix is NaN!')
        
    return cov_block


    

if __name__ == '__main__':
    print(0)

























