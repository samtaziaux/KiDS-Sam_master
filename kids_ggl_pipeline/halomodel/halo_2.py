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
from scipy.interpolate import interp1d, UnivariateSpline
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
    Integrate, Integrate1, extrap1d, extrap2d, f_k, fill_nan, gas_concentration,
    load_hmf, star_concentration, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, sigma_crit,
    power_to_corr_ogata, wp, wp_beta_correction)
from .dark_matter import (
    DM_mm_spectrum, GM_cen_spectrum, GM_sat_spectrum,
    delta_NFW, MM_analy, GM_cen_analy, GM_sat_analy, GG_cen_analy,
    GG_sat_analy, GG_cen_sat_analy, miscenter, TwoHalo, TwoHalo_gg)
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
"""
#@memoize
def Mass_Function(M_min, M_max, step, name, **cosmology_params):
    return MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=step,
        mf_fit=name, delta_h=200.0, delta_wrt='mean',
        cut_fit=False, z2=None, nz=None, delta_c=1.686,
        **cosmology_params)
"""



#################
##
## Main function
##
#################


def model(theta, R, calculate_covariance=False):
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
        s_concentration, s_mor, s_scatter = theta

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
    # integrand
    rvir_range_3d_i = logspace(-2.5, 1.2, 25, endpoint=True)
    #rvir_range_3d_i = logspace(-4, 2, 60, endpoint=True)
    # integrate over redshift later on
    # assuming this means arcmin for now -- implement a way to check later!
    #if setup['distances'] == 'angular':
        #R = R * cosmo.
    rvir_range_2d_i = R[0][1:]

    # Calculating halo model
    
    """Calculating halo model"""

    # interpolate selection function to the same grid as redshift and
    # observable to be used in trapz
    if selection.filename == 'None':
        if integrate_zlens:
            completeness = np.ones((z.size,nbins,hod_observable.shape[1]))
        else:
            completeness = np.ones(hod_observable.shape)
    elif integrate_zlens:
        completeness = np.array(
            [[selection.interpolate([zi]*obs.size, obs, method='linear')
              for obs in hod_observable] for zi in z[:,0]])
    else:
        completeness = np.array(
            [selection.interpolate([zi]*obs.size, obs, method='linear')
             for zi, obs in zip(z[:,0], hod_observable)])

    if ingredients['centrals']:
        pop_c, prob_c = hod.number(
            hod_observable, mass_range, c_mor[0], c_scatter[0],
            c_mor[1:], c_scatter[1:], completeness,
            obs_is_log=observable.is_log)
    else:
        pop_c = np.zeros((nbins,mass_range.size))

    if ingredients['satellites']:
        pop_s, prob_s = hod.number(
            hod_observable, mass_range, s_mor[0], s_scatter[0],
            s_mor[1:], s_scatter[1:], completeness,
            obs_is_log=observable.is_log)
    else:
        pop_s = np.zeros(pop_c.shape)

    pop_g = pop_c + pop_s
    
    # note that pop_g already accounts for incompleteness
    dndm = array([hmf_i.dndm for hmf_i in hmf])
    ngal = hod.nbar(dndm, pop_g, mass_range)
    meff = hod.Mh_effective(
        dndm, pop_g, mass_range, return_log=observable.is_log)
    
    
    """Power spectra"""

    # damping of the 1h power spectra at small k
    F_k1 = f_k(k_range_lin)
    #F_k1 = 1
    # Fourier Transform of the NFW profile
    if ingredients['centrals']:
        uk_c = nfw.uk(
            k_range_lin, mass_range, rvir_range_lin, concentration, rho_bg,
            setup['delta'])
    elif integrate_zlens:
        uk_c = np.ones((nbins,z.size//nbins,mass_range.size,k_range_lin.size))
    else:
        uk_c = np.ones((nbins,mass_range.size,k_range_lin.size))
    # and of the NFW profile of the satellites
    if ingredients['satellites']:
        uk_s = nfw.uk(
            k_range_lin, mass_range, rvir_range_lin, concentration_sat, rho_bg,
            setup['delta'])
        uk_s = uk_s/uk_s[:,0][:,None]
    elif integrate_zlens:
        uk_s = np.ones((nbins,z.size//nbins,mass_range.size,k_range_lin.size))
    else:
        uk_s = np.ones((nbins,mass_range.size,k_range_lin.size))

    # If there is miscentring to be accounted for
    if ingredients['miscentring']:

        p_off, r_off = c_miscent[1:]
        
        uk_c = uk_c * miscenter(
            p_off, r_off, expand_dims(mass_range, -1),
            expand_dims(rvir_range_lin, -1), k_range_lin,
            expand_dims(concentration, -1), uk_c.shape)
    uk_c = uk_c / expand_dims(uk_c[...,0], -1)

    # Galaxy - dark matter spectra (for lensing)
    bias = c_twohalo
    bias = array([bias]*k_range_lin.size).T
    if ingredients['twohalo']:
        """
        # unused but not removing as we might want to use it later
        #bias_out = bias.T[0] * array(
            #[TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin, rvir_range_lin_i,
                     #mass_range)[1]
             #for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
             #in zip(rvir_range_lin, hmf, ngal, pop_g)])
        """
        Pgm_2h = bias * array(
            [TwoHalo(hmf_i, ngal_i, pop_g_i,
                     rvir_range_lin_i, mass_range)[0]
             for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
             in zip(rvir_range_lin, hmf, expand_dims(ngal, -1),
                    expand_dims(pop_g, -2))])
        #print('Pg_2h in {0:.2e} s'.format(time()-ti))
    #elif integrate_zlens:
        #Pg_2h = np.zeros((nbins,z.size//nbins,setup['lnk_bins']))
    else:
        Pgm_2h = np.zeros((nbins,setup['lnk_bins']))

    if not integrate_zlens:
        rho_bg = rho_bg[...,0]

    if ingredients['centrals']:

        Pgm_c = F_k1 * GM_cen_analy(
            dndm, uk_c, rho_bg, pop_c, ngal, mass_range)
    elif integrate_zlens:
        Pgm_c = np.zeros((z.size,nbins,setup['lnk_bins']))
    else:
        Pgm_c = np.zeros((nbins,setup['lnk_bins']))


    if ingredients['satellites']:

        Pgm_s = F_k1 * GM_sat_analy(
            dndm, uk_c, uk_s, rho_bg, pop_s, ngal, mass_range)
    else:
        Pgm_s = np.zeros(Pg_c.shape)

    Pgm_k = Pgm_c + Pg_s + Pgm_2h
    
    # finally integrate over (weight by, really) lens redshift
    if integrate_zlens:
        intnorm = np.sum(nz, axis=0)
        meff = np.sum(nz*meff, axis=0) / intnorm
    
    

    # Galaxy - galaxy spectra (for clustering)
    if ingredients['twohalo']:
        Pgg_2h = bias * array(
        [TwoHalo_gg(hmf_i, ngal_i, pop_g_i,
                 rvir_range_lin_i, mass_range)[0]
         for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
         in zip(rvir_range_lin, hmf, expand_dims(ngal, -1),
                expand_dims(pop_g, -2))])

    ncen = hod.nbar(dndm, pop_c, mass_range)
    nsat = hod.nbar(dndm, pop_s, mass_range)

    
    if ingredients['centrals']:
        """
        Pgg_c = F_k1 * GG_cen_analy(dndm, ncen, ngal, (nbins,setup['lnk_bins']), mass_range)
        """
        Pgg_c = np.zeros((nbins,setup['lnk_bins']))
    else:
        Pgg_c = np.zeros((nbins,setup['lnk_bins']))
    
    if ingredients['satellites']:
        Pgg_s = F_k1 * GG_sat_analy(dndm, uk_s, pop_s, ngal, beta, mass_range)
    else:
        Pgg_s = np.zeros(Pgg_c.shape)
        
    if ingredients['centrals'] and ingredients['satellites']:
        Pgg_cs = F_k1 * GG_cen_sat_analy(dndm, uk_s, pop_c, pop_s, ngal, mass_range)
    else:
        Pgg_cs = np.zeros(Pgg_c.shape)
        
    Pgg_k = Pgg_c + (2.0 * Pgg_cs) + Pgg_s + Pgg_2h
                    
                            
    
    # Matter - matter spectra
    Pmm_1h = F_k1 * MM_analy(dndm, uk_c, rho_bg, mass_range)
                            
    Pmm_k = Pmm_1h + array([hmf_i.power for hmf_i in hmf])
                    
    
         
    # not yet allowed
    if setup['return'] == 'power':
        # note this doesn't include the point mass! also, we probably
        # need to return k
        if integrate_zlens:
            Pg_k = np.sum(z*Pg_k, axis=1) / intnorm
        return [Pg_k, meff]
    if integrate_zlens:
        P_inter = [[UnivariateSpline(k_range, logPg_ij, s=0, ext=0)
                    for logPg_ij in logPg_i] for logPg_i in np.log(Pg_k)]
    else:
        P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                   for Pg_k_i in Pg_k]
    
    ######## !!!!!!!!!! ########
    # I am not completely sure how to either make the output of those parts optional,
    # or how to deal with the output if we want to have all parts output here.
    # Same goes for the correlation function that follow.
    ######## !!!!!!!!!! ########
    
    P_inter_2 = [UnivariateSpline(k_range, np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in _izip(Pgg_k)]
    
    P_inter_3 = [UnivariateSpline(k_range, np.log(Pmm_i), s=0, ext=0)
                    for Pmm_i in _izip(Pmm_k)]

         
    
    # correlation functions
    if integrate_zlens:
        #to = time()
        xi2 = np.array(
            [[power_to_corr_ogata(P_inter_ji, rvir_range_3d)
              for P_inter_ji in P_inter_j] for P_inter_j in P_inter])
        #print('xi2 in {0:.2e}'.format(time()-to))
    else:
        #to = time()
        xi2 = np.array(
            [power_to_corr_ogata(P_inter_i, rvir_range_3d)
             for P_inter_i in P_inter])
        #print('xi2 in {0:.2e}'.format(time()-to))
    # not yet allowed
    if setup['return'] == 'xi':
        if integrate_zlens:
            xi2 = np.sum(z*xi2, axis=1) / intnorm
        return [xi2, meff]
    

    xi2_2 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2_2[i] = power_to_corr_ogata(P_inter_2[i], rvir_range_3d)

    xi2_3 = np.zeros((M_bin_min.size,rvir_range_3d.size))
    for i in xrange(M_bin_min.size):
        xi2_3[i] = power_to_corr_ogata(P_inter_3[i], rvir_range_3d)

    # projected surface density
    # this is the slowest part of the model
    #
    # do we require a double loop here when weighting n(zlens)?
    # perhaps should not integrate over zlens at the power spectrum level
    # but only here -- or even just at the return stage!
    #
    # this avoids the interpolation necessary for better
    # accuracy of the ESD when returning sigma or kappa
    rvir_sigma = rvir_range_2d_i if setup['return'] in ('sigma', 'kappa') \
        else rvir_range_3d_i


    if integrate_zlens:
        surf_dens2 = array(
            [[sigma(xi2_ij, rho_bg_i, rvir_range_3d, rvir_sigma)
              for xi2_ij in xi2_i] for xi2_i, rho_bg_i in zip(xi2, rho_bg)])
        """
        if setup['distances'] == 'proper':
            surf_dens2 = trapz(
                surf_dens2*expand_dims(nz*(1+z)**2, -1), z[:,0], axis=0)
        else:
            surf_dens2 = trapz(surf_dens2*expand_dims(nz, -1), z[:,0], axis=0)
        surf_dens2 = surf_dens2 / trapz(nz, z, axis=0)[:,None]
        """
    else:
        surf_dens2 = array(
            [sigma(xi2_i, rho_i, rvir_range_3d, rvir_sigma)
             for xi2_i, rho_i in zip(xi2, rho_bg)])
             
    # units of Msun/pc^2
    if setup['return'] in ('sigma', 'kappa') and ingredients['pointmass']:
        pointmass = c_pm[1]/(2*np.pi) * array(
            [10**m_pm / rvir_range_2d_i**2 for m_pm in c_pm[0]])
        #print('pointmass =', pointmass.shape)
        surf_dens2 = surf_dens2 + pointmass

    zo = expand_dims(z, -1) if integrate_zlens else z
    if setup['distances'] == 'proper':
        surf_dens2 *= (1+zo)**2
    if setup['return'] == 'kappa':
        surf_dens2 /= sigma_crit(cosmo_model, zo, zs)
    if integrate_zlens:
        # haven't checked the denominator below
        norm = trapz(nz, z, axis=0)
        #if setup['return'] == 'kappa':
            #print('sigma_crit =', sigma_crit(cosmo_model, z, zs).shape)
        surf_dens2 = \
            trapz(surf_dens2 * expand_dims(nz, -1), z[:,0], axis=0) \
            / norm[:,None]
        zw = nz * sigma_crit(cosmo_model, z, zs) \
            if setup['return'] == 'kappa' else nz
        zeff = trapz(zw*z, z, axis=0) / trapz(zw, z, axis=0)
        #print('zeff =', zeff)
        
    # in Msun/pc^2
    if not setup['return'] == 'kappa':
        surf_dens2 /= 1e12
    
    #print('surf_dens2 =', surf_dens2.shape)
    # fill/interpolate nans
    surf_dens2[(surf_dens2 <= 0) | (surf_dens2 >= 1e20)] = np.nan
    for i in range(nbins):
        surf_dens2[i] = fill_nan(surf_dens2[i])
    if setup['return'] in ('kappa', 'sigma'):
        surf_dens2_r = array(
            [UnivariateSpline(rvir_range_2d_i, np.nan_to_num(si), s=0)
             for si in zip(surf_dens2)])
        surf_dens2 = np.array([s_r(rvir_range_2d_i) for s_r in surf_dens2_r])
        return [surf_dens2, meff]

    # excess surface density
    d_surf_dens2 = array(
        [np.nan_to_num(
            d_sigma(surf_dens2_i, rvir_range_3d_i, rvir_range_2d_i))
         for surf_dens2_i in surf_dens2])
    
    out_esd_tot = array(
        [UnivariateSpline(rvir_range_2d_i, np.nan_to_num(d_surf_dens2_i), s=0)
         for d_surf_dens2_i in d_surf_dens2])
    
    out_esd_tot_inter = np.zeros((nbins, rvir_range_2d_i.size))
    for i in range(nbins):
        out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)
    

    # this should be moved to the power spectrum calculation
    if ingredients['pointmass']:
        # the 1e12 here is to convert Mpc^{-2} to pc^{-2} in the ESD
        pointmass = c_pm[1]/(np.pi*1e12) * array(
            [10**m_pm / (rvir_range_2d_i**2) for m_pm in c_pm[0]])
        out_esd_tot_inter = out_esd_tot_inter + pointmass


    # Add other outputs as needed. Total ESD should always be first!
    return [out_esd_tot_inter, meff]
    



    # deal with this!

        
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
                         
    #"""
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
                         
    #"""
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
    #"""
    sur_den2_2_out = _array([UnivariateSpline(rvir_range_3d_i,
                            np.nan_to_num(wp_i), s=0)
                             for wp_i in _izip(w_p)])
        
    sur_den2_2_out_inter = np.zeros((M_bin_min.size, rvir_range_2d_i.size))
    for i in xrange(M_bin_min.size):
        sur_den2_2_out_inter[i] = sur_den2_2_out[i](rvir_range_2d_i)
    #"""
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
    #return [out_esd_tot_inter_2/out_esd_tot_inter]
    #return [k_range, Pg_c, Pg_s, Pg_2h, Pg_k, Pgg_s, Pgg_cs, Pgg_2h, Pgg_k, Pmm_1h, [hmf[0].power, hmf[1].power, hmf[2].power], Pmm]

if __name__ == '__main__':
    print(0)
