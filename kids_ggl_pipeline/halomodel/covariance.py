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
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d, UnivariateSpline
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


def calc_cov(params):
    
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
    integ3 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( np.sqrt(np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j])*np.sqrt(np.exp(P_mm_j) + shape_noise[b_j]* delta[b_i, b_j]) * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j])) + np.exp(P_gm_i)*np.exp(P_gm_j) )
    integ4 = np.exp(lnk)**2.0 * sp.jv(2, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * (np.sqrt(np.exp(P_mm_i) + shape_noise[b_i]* delta[b_i, b_j])*np.sqrt(np.exp(P_mm_j) + shape_noise[b_j]* delta[b_i, b_j]))
    b = ((Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ3, dx=dlnk) + ((2.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ4, dx=dlnk)
                    
    # cross
    integ5 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * ( (np.sqrt(np.exp(P_gm_i))*np.sqrt(np.exp(P_gm_j))) * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j])) )
    integ6 = np.exp(lnk)**2.0 * sp.jv(0, np.exp(lnk) * r_i) * sp.jv(2, np.exp(lnk) * r_j) * W_p(np.exp(lnk))**2.0 * np.sqrt((np.exp(P_gg_i) + 1.0/ngal[b_i]* delta[b_i, b_j]))*np.sqrt((np.exp(P_gg_j) + 1.0/ngal[b_j]* delta[b_i, b_j]))
    c = ((2.0*Aw_rr)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ5, dx=dlnk) + ((4.0*Pi_max)/(Awr_i * Awr_j))/(2.0*np.pi) * simps(integ6, dx=dlnk)
    
    return b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j, [a, b, c]


def cov_func(rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_wp, cov_esd, cov_cross):
    #import progressbar

    b_i = xrange(len(P_inter_2))
    b_j = xrange(len(P_inter_2))
    i = xrange(len(rvir_range_2d_i))
    j = xrange(len(rvir_range_2d_i))
    
    paramlist = [list(tup) for tup in itertools.product(b_i,b_j,i,j)]
    for i in paramlist:
        i.extend([rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal])
    
    pool = multi.Pool(processes=12)
    for i, j, val in pool.map(calc_cov, paramlist):
        #print(i, j, val)
        cov_wp[i,j] = val[0]
        cov_esd[i,j] = val[1]
        cov_cross[i,j] = val[2]

    #print(cov_wp)

    
    #bar = progressbar.ProgressBar(maxval=rvir_range_2d_i.size*shape_noise.size*rvir_range_2d_i.size*shape_noise.size).start()
    """
    for b_i, bin_i in enumerate(P_inter_2):
        for b_j, bin_j in enumerate(P_inter_2):
            for i, r_i in enumerate(rvir_range_2d_i):
                for j, r_j in enumerate(rvir_range_2d_i):
                    
                    cov_wp[b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j], cov_esd[b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j], cov_cross[b_i*rvir_range_2d_i.size+i,b_j*rvir_range_2d_i.size+j] = calc_cov(b_i, b_j, r_i, r_j, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, delta, minsteps, mink, temp_min_k)
    """
                    
                    
                    
                    #bar.update((b_i+1)*(b_j+1)*(i+1)*(j+1))
    #bar.finish()
    return cov_wp, cov_esd, cov_cross



def covariance(theta, R, h=0.7, Om=0.315, Ol=0.685,
          expansion=100, expansion_stars=160, n_bins=10000,
          lnk_min=-13., lnk_max=17.):
    np.seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')
    # making them local doesn't seem to make any difference
    # but keeping for now
    _array = array
    _izip = izip
    _logspace = logspace
    _linspace = linspace
              
    sigma_8, H0, omegam, omegab_h2, omegav, n, \
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
                                    'dlnk': k_step, 'transfer_model': 'CAMB',
                                    'z':np.float64(z_i)})

    # Calculation
    # Tinker10 should also be read from theta!
    #to = time()
    hmf = _array([])
    h = H0/100.0
    cosmo_model = LambdaCDM(H0=H0, Ob0=omegab_h2/(h**2.0), Om0=omegam, Ode0=omegav)
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
    rho_dm = np.zeros(z.shape)

    omegab = hmf[0].cosmo.Ob0
    omegac = hmf[0].cosmo.Om0-omegab
    omegav = hmf[0].cosmo.Ode0

    for i in xrange(z.size):
        mass_func[i] = hmf[i].dndlnm
        rho_mean[i] = hmf[i].mean_density
        rho_crit[i] = rho_mean[i] / (omegac+omegab)
        rho_dm[i] = rho_mean[i] * baryons.f_dm(omegab, omegac)


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
    u_k = _array([NFW_f(np.float64(z_i), rho_dm_i, np.float64(f_i), mass_range,\
                rvir_range_lin_i, k_range_lin,\
                c=concentration_i) for rvir_range_lin_i, rho_dm_i, z_i,\
                f_i, concentration_i in _izip(rvir_range_lin, \
                rho_dm, z, f, concentration)])
                   
    # and of the NFW profile of the satellites
    uk_s = _array([NFW_f(np.float64(z_i), rho_dm_i, np.float64(fc_nsat_i), \
                mass_range, rvir_range_lin_i, k_range_lin)
                for rvir_range_lin_i, rho_dm_i, z_i, fc_nsat_i in \
                _izip(rvir_range_lin, rho_dm, z, fc_nsat)])
    uk_s = uk_s/uk_s[:,0][:,None]
                   
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
        Pg_c = F_k1 * _array([GM_cen_analy(hmf_i, u_k_i, rho_dm_i, pop_c_i,
                    ngal_i, mass_range)
                    for rho_dm_i, hmf_i, pop_c_i, ngal_i, u_k_i in\
                    _izip(rho_dm, hmf, pop_c, ngal, u_k)])
    else:
        Pg_c = np.zeros((n_bins_obs,n_bins))
    if satellites:
        Pg_s = F_k1 * _array([GM_sat_analy(hmf_i, u_k_i, uk_s_i, rho_dm_i,
                    pop_s_i, ngal_i, mass_range)
                    for rho_dm_i, hmf_i, pop_s_i, ngal_i, u_k_i, uk_s_i in\
                    _izip(rho_dm, hmf, pop_s, ngal, u_k, uk_s)])
    else:
        Pg_s = np.zeros((n_bins_obs,n_bins))



    Pg_k = _array([(rho_dm_i/rho_mean_i) * (Pg_c_i + Pg_s_i) + Pg_2h_i
                   for Pg_c_i, Pg_s_i, Pg_2h_i, rho_dm_i, rho_mean_i
                   in _izip(Pg_c, Pg_s, Pg_2h, rho_dm, rho_mean)])
    
    
    
    # Galaxy - galaxy spectra (for clustering)
    Pgg_2h = bias * _array([TwoHalo_gg(hmf_i, ngal_i, pop_g_i, k_range_lin,
                    rvir_range_lin_i, mass_range)[0]
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
    Pgg_s = F_k1 * _array([GG_sat_analy(hmf_i, uk_s_i, pop_s_i, ngal_i, beta_i, mass_range)
                    for hmf_i, uk_s_i, pop_s_i, ngal_i, beta_i in\
                    _izip(hmf, uk_s, pop_s, ngal, beta)])
                            
    Pgg_cs = F_k1 * _array([GG_cen_sat_analy(hmf_i, uk_s_i, pop_c_i,
                    pop_s_i, ngal_i, mass_range)
                    for hmf_i, pop_c_i, pop_s_i, ngal_i, uk_s_i in\
                    _izip(hmf, pop_c, pop_s, ngal, uk_s)])
                            
    Pgg_k = _array([(Pgg_c_i + 2.0 * Pgg_cs_i + Pgg_s_i) + Pgg_2h_i
                    for Pgg_c_i, Pgg_cs_i, Pgg_s_i, Pgg_2h_i
                    in _izip(Pgg_c, Pgg_cs, Pgg_s, Pgg_2h)])
                            

    # Matter - matter spectra
    Pmm_1h = F_k1 * _array([MM_analy(hmf_i, u_k_i, rho_dm_i, mass_range)
                    for hmf_i, u_k_i, rho_dm_i, beta_i in\
                    _izip(hmf, u_k, rho_dm, beta)])
                            
    Pmm = _array([(rho_dm_i/rho_mean_i) * Pmm_1h_i + hmf_i.power
                    for Pmm_1h_i, hmf_i, rho_dm_i, rho_mean_i
                    in _izip(Pmm_1h, hmf, rho_dm, rho_mean)])


    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                    for Pg_k_i in _izip(Pg_k)]
        
    P_inter_2 = [UnivariateSpline(k_range, np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in _izip(Pgg_k)]

    P_inter_3 = [UnivariateSpline(k_range, np.log(Pmm_i), s=0, ext=0)
                    for Pmm_i in _izip(Pmm)]

    xi2 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2[i] = power_to_corr_ogata(P_inter[i], rvir_range_3d)

    xi2_2 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2_2[i] = power_to_corr_ogata(P_inter_2[i], rvir_range_3d)


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

    sur_den2_2_out = _array([UnivariateSpline(rvir_range_3d_i,
                                          np.nan_to_num(wp_i), s=0)
                         for wp_i in _izip(w_p)])
    
    w_p_out = np.zeros((M_bin_min.size, rvir_range_2d_i.size))
    for i in xrange(M_bin_min.size):
        w_p_out[i] = sur_den2_2_out[i](rvir_range_2d_i)




    Pi_max = 100.0
    
    kids_area = 180 * 3600.0 #500 #To be in arminutes!
    eff_density = 2.34#8.53 #1.5#8.53
    kids_variance_squared = 0.076#0.275#0.076
    z_kids = 0.6
    
    sigma_crit = sigma_crit_kids(hmf, z, 0.2, 0.9, '/home/dvornik/data2_dvornik/KidsCatalogues/IMSIM_Gall30th_2016-01-14_deepspecz_photoz_1000_4_specweight.cat') * hmf[0].cosmo.h
    eff_density_in_mpc = eff_density / ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value / hmf[0].cosmo.h )**2.0
    shape_noise = (sigma_crit**2.0) * hmf[0].cosmo.H(z_kids).value * (kids_variance_squared / eff_density_in_mpc)/ (3.0*10.0**8.0)
    #shape_noise = (sigma_crit**2.0) * hmf[0].cosmo.H(z_kids).value * (eff_density_in_mpc) / (3.0*10.0**8.0)
    
    radius = np.sqrt(kids_area/np.pi) * ((hmf[0].cosmo.kpc_comoving_per_arcmin(z_kids).to('Mpc/arcmin')).value) / hmf[0].cosmo.h # conversion of area in deg^2 to Mpc/h!
    
    print(radius)
    print(eff_density_in_mpc)
    ngal = 2.0*ngal
    
    #shape_noise = (4.7**2.0) *10.0**6.0 * 100.0 * 0.076 / (2.0 * 3.0*10.0**8.0) * np.ones(3)
    print(hmf[0].cosmo.H(z_kids).value)
    print(shape_noise)
    print(1.0/ngal)
    #quit()


    print(rvir_range_2d_i.shape)

    cov_wp = np.empty((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size))
    cov_esd = np.empty((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size))
    cov_cross = np.empty((rvir_range_2d_i.size*M_bin_min.size, rvir_range_2d_i.size*M_bin_min.size))

    W = 2.0 * np.pi * radius**2.0 * sp.jv(1, k_range_lin*radius) / (k_range_lin*radius)
    W_p = UnivariateSpline(k_range_lin, W, s=0, ext=0)

    cov_wp, cov_esd, cov_cross = cov_func(rvir_range_2d_i, P_inter, P_inter_2, P_inter_3, W_p, Pi_max, shape_noise, ngal, cov_wp, cov_esd, cov_cross)

    """
    cor_wp = cov_wp/np.sqrt(np.outer(np.diag(cov_wp), np.diag(cov_wp.T)))
    pl.imshow(cor_wp, interpolation='nearest')
    pl.show()

    cor_esd = cov_esd/np.sqrt(np.outer(np.diag(cov_esd), np.diag(cov_esd.T)))
    pl.imshow(cor_esd, interpolation='nearest')
    pl.show()


    cor_cross = cov_cross/np.sqrt(np.outer(np.diag(cov_cross), np.diag(cov_cross.T)))
    pl.imshow(cov_cross, interpolation='nearest')
    pl.show()
    """

    
    return cov_wp, cov_esd, cov_cross, M_bin_min.size


if __name__ == '__main__':
    print(0)

























