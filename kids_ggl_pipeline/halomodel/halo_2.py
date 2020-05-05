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
import os
# disable threading in numpy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
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
    fill_nan, load_hmf, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, sigma_crit,
    power_to_corr_ogata, wp, wp_beta_correction, power_to_sigma, power_to_sigma_ogata)
from .dark_matter import (
    mm_analy, gm_cen_analy, gm_sat_analy, gg_cen_analy,
    gg_sat_analy, gg_cen_sat_analy, two_halo_gm, two_halo_gg, halo_exclusion)
from .covariance import covariance
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

    #cov = covariance(theta, R)
    #quit()
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
    ingredient_gm, ingredient_gg, ingredient_mm, ingredient_mlf = False, False, False, False
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
        if observable.ingredient == 'mm':
            ingredient_mm = True
            observable_mm = observable
            nbins_mm = observable.nbins
            idx_mm = np.s_[nbins:nbins+nbins_mm]
            nbins += nbins_mm
        if observable.ingredient == 'mlf':
            ingredient_mlf = True
            observable_mlf = observable
            hod_observable_mlf = observable.sampling
            nbins_mlf = observable.nbins
            idx_mlf = np.s_[nbins:nbins+nbins_mlf]
            nbins += nbins_mlf

    if setup['return'] in ('wp', 'esd_wp') and not ingredient_gg:
        raise ValueError(
        'If return=wp or return=esd_wp then you must toggle the' \
        ' clustering as an ingredient. Similarly, if return=esd' \
        ' or return=esd_wp then you must toggle the lensing' \
        ' as an ingredient as well.')
    if setup['return'] in ('esd', 'esd_wp') and not ingredient_gm:
        raise ValueError(
        'If return=wp or return=esd_wp then you must toggle the' \
        ' clustering as an ingredient. Similarly, if return=esd' \
        ' or return=esd_wp then you must toggle the lensing' \
        ' as an ingredient as well.')
    
    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta
    
    sigma8, h, omegam, omegab, n, w0, wa, Neff, z = cosmo[:9]
    
    #output = []
    output = np.empty(nbins, dtype=object)
    
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
    #    z = np.array([z])
    #if z.ndim > 1:
    #    z = z.reshape(1,-1).squeeze()
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
    if ingredient_gg:
        rvir_range_2d_i_gg = [r[1:].astype('float64') for r in R[idx_gg]]
    if ingredient_mm:
        rvir_range_2d_i_mm = [r[1:].astype('float64') for r in R[idx_mm]]
    if ingredient_mlf:
        rvir_range_2d_i_mlf = [r[1:].astype('float64') for r in R[idx_mlf]]
    # We might want to move this in the configuration part of the code!
    # Same goes for the bins above
    
    # Calculating halo model
    
    """Calculating halo model"""

    # interpolate selection function to the same grid as redshift and
    # observable to be used in trapz
    if selection.filename == 'None':
        if integrate_zlens:
            completeness = np.ones((z.size,nbins,hod_observable_gm.shape[1]))
        else:
            completeness = np.ones(hod_observable_gm.shape)
    elif integrate_zlens:
        completeness = np.array(
            [[selection.interpolate([zi]*obs.size, obs, method='linear')
              for obs in hod_observable_gm] for zi in z[:,0]])
    else:
        completeness = np.array(
            [selection.interpolate([zi]*obs.size, obs, method='linear')
             for zi, obs in zip(z[:,0], hod_observable_gm)])

    pop_c = np.zeros((nbins,mass_range.size))
    pop_s = np.zeros((nbins,mass_range.size))
    if ingredient_gm:
        if ingredients['centrals']:
            pop_c[idx_gm,:], prob_c_gm = hod.number(
                hod_observable_gm, mass_range, c_mor[0], c_scatter[0],
                c_mor[1:], c_scatter[1:], completeness,
                obs_is_log=observable_gm.is_log)

        if ingredients['satellites']:
            pop_s[idx_gm,:], prob_s_gm = hod.number(
                hod_observable_gm, mass_range, s_mor[0], s_scatter[0],
                s_mor[1:], s_scatter[1:], completeness,
                obs_is_log=observable_gm.is_log)
        
    if ingredient_gg:
        if ingredients['centrals']:
            pop_c[idx_gg,:], prob_c_gg = hod.number(
                hod_observable_gg, mass_range, c_mor[0], c_scatter[0],
                c_mor[1:], c_scatter[1:],
                obs_is_log=observable_gg.is_log)

        if ingredients['satellites']:
            pop_s[idx_gg,:], prob_s_gg = hod.number(
                hod_observable_gg, mass_range, s_mor[0], s_scatter[0],
                s_mor[1:], s_scatter[1:],
                obs_is_log=observable_gg.is_log)
        
    pop_g = pop_c + pop_s
    # note that pop_g already accounts for incompleteness
    dndm = array([hmf_i.dndm for hmf_i in hmf])
    ngal = np.empty(nbins)
    meff = np.empty(nbins)
    if ingredient_gm:
        ngal[idx_gm] = hod.nbar(dndm[idx_gm], pop_g[idx_gm], mass_range)
        meff[idx_gm] = hod.Mh_effective(
            dndm[idx_gm], pop_g[idx_gm], mass_range, return_log=observable_gm.is_log)
    if ingredient_gg:
        ngal[idx_gg] = hod.nbar(dndm[idx_gg], pop_g[idx_gg], mass_range)
        meff[idx_gg] = hod.Mh_effective(
            dndm[idx_gg], pop_g[idx_gg], mass_range, return_log=observable_gg.is_log)
    if ingredient_mm:
        ngal[idx_mm] = np.zeros_like(nbins_mm)
        meff[idx_mm] = np.zeros_like(nbins_mm)
        
    # Luminosity or mass function as an output:
    if ingredient_mlf:
        # Needs independent redshift input!
        z_mlf = z[idx_mlf]
        if z_mlf.size == 1 and nbins_mlf > 1:
            z_mlf = z_mlf*np.ones(nbins_mlf)
        if z_mlf.size != nbins_mlf:
            raise ValueError(
                'Number of redshift bins should be equal to the number of observable bins!')
        hmf_mlf, _rho_mean = load_hmf(z_mlf, setup, cosmo_model, transfer_params)
        dndm_mlf = array([hmf_i.dndm for hmf_i in hmf_mlf])
        if ingredients['centrals']:
            pop_c_mlf = hod.mlf(
                hod_observable_mlf, dndm_mlf, mass_range, c_mor[0], c_scatter[0],
                c_mor[1:], c_scatter[1:],
                obs_is_log=observable_mlf.is_log)

        if ingredients['satellites']:
            pop_s_mlf = hod.mlf(
                hod_observable_mlf, dndm_mlf, mass_range, s_mor[0], s_scatter[0],
                s_mor[1:], s_scatter[1:],
                obs_is_log=observable_mlf.is_log)
        pop_g_mlf = pop_c_mlf + pop_s_mlf
        
        mlf_inter = [UnivariateSpline(hod_i, np.log(ngal_i), s=0, ext=0)
                    for hod_i, ngal_i in zip(hod_observable_mlf, pop_g_mlf)]
        mlf_out = [exp(mlf_i(np.log10(r_i))) for mlf_i, r_i
                    in zip(mlf_inter, rvir_range_2d_i_mlf)]
        output[idx_mlf] = mlf_out
    

    """Power spectra"""

    # damping of the 1h power spectra at small k
    F_k1 = sp.erf(k_range_lin/0.1)
    F_k2 = np.ones_like(k_range_lin)
    #F_k2 = sp.erfc(k_range_lin/10.0)
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
        uk_s = uk_s / expand_dims(uk_s[...,0], -1)
    elif integrate_zlens:
        uk_s = np.ones((nbins,z.size//nbins,mass_range.size,k_range_lin.size))
    else:
        uk_s = np.ones((nbins,mass_range.size,k_range_lin.size))

    # If there is miscentring to be accounted for
    # Only for galaxy-galaxy lensing!
    if ingredients['miscentring']:
        p_off, r_off = c_miscent#[1:]
        uk_c[idx_gm] = uk_c[idx_gm] * nfw.miscenter(
            p_off, r_off, expand_dims(mass_range, -1),
            expand_dims(rvir_range_lin, -1), k_range_lin,
            expand_dims(concentration, -1), uk_c[idx_gm].shape)
    uk_c = uk_c / expand_dims(uk_c[...,0], -1)

    
    # Galaxy - dark matter spectra (for lensing)
    bias = c_twohalo
    bias = array([bias]*k_range_lin.size).T
    
    if not integrate_zlens:
        rho_bg = rho_bg[...,0]
    
    if ingredient_gm:
        if ingredients['twohalo']:
            """
            # unused but not removing as we might want to use it later
            #bias_out = bias.T[0] * array(
                #[TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin, rvir_range_lin_i,
                     #mass_range)[1]
                #for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
                #in zip(rvir_range_lin, hmf, ngal, pop_g)])
            """
            Pgm_2h = F_k2 * bias * array(
                [two_halo_gm(hmf_i, ngal_i, pop_g_i, mass_range)[0]
                for hmf_i, ngal_i, pop_g_i
                in zip(hmf[idx_gm], expand_dims(ngal[idx_gm], -1),
                        expand_dims(pop_g[idx_gm], -2))])
            #print('Pg_2h in {0:.2e} s'.format(time()-ti))
        #elif integrate_zlens:
            #Pg_2h = np.zeros((nbins,z.size//nbins,setup['lnk_bins']))
        else:
            Pgm_2h = np.zeros((nbins_gm,setup['lnk_bins']))

        if ingredients['centrals']:
            Pgm_c = F_k1 * gm_cen_analy(
                dndm[idx_gm], uk_c[idx_gm], rho_bg[idx_gm], pop_c[idx_gm], ngal[idx_gm], mass_range)
        elif integrate_zlens:
            Pgm_c = np.zeros((z.size,nbins_gm,setup['lnk_bins']))
        else:
            Pgm_c = F_k1 * np.zeros((nbins_gm,setup['lnk_bins']))


        if ingredients['satellites']:
            Pgm_s = F_k1 * gm_sat_analy(
                dndm[idx_gm], uk_c[idx_gm], uk_s[idx_gm], rho_bg[idx_gm], pop_s[idx_gm], ngal[idx_gm], mass_range)
        else:
            Pgm_s = F_k1 * np.zeros(Pgm_c.shape)
        
        if ingredients['haloexclusion'] and setup['return'] != 'power':
            Pgm_k_t = Pgm_c + Pgm_s
            Pgm_k = Pgm_c + Pgm_s + Pgm_2h
        else:
            Pgm_k = Pgm_c + Pgm_s + Pgm_2h
    
        # finally integrate over (weight by, really) lens redshift
        if integrate_zlens:
            intnorm = np.sum(nz, axis=0)
            meff[idx_gm] = np.sum(nz*meff[idx_gm], axis=0) / intnorm
    
    # Galaxy - galaxy spectra (for clustering)
    if ingredient_gg:
        if ingredients['twohalo']:
            Pgg_2h = F_k2 * bias * array(
            [two_halo_gg(hmf_i, ngal_i, pop_g_i, mass_range)[0]
            for hmf_i, ngal_i, pop_g_i
            in zip(hmf[idx_gg], expand_dims(ngal[idx_gg], -1),
                    expand_dims(pop_g[idx_gg], -2))])
        else:
            Pgg_2h = F_k2 * np.zeros((nbins_gg,setup['lnk_bins']))
            
        ncen = hod.nbar(dndm[idx_gg], pop_c[idx_gg], mass_range)
        nsat = hod.nbar(dndm[idx_gg], pop_s[idx_gg], mass_range)
    
        if ingredients['centrals']:
            """
            Pgg_c = F_k1 * gg_cen_analy(dndm, ncen, ngal, (nbins,setup['lnk_bins']), mass_range)
            """
            Pgg_c = F_k1 * np.zeros((nbins_gg,setup['lnk_bins']))
        else:
            Pgg_c = F_k1 * np.zeros((nbins_gg,setup['lnk_bins']))
    
        if ingredients['satellites']:
            beta = s_beta
            Pgg_s = F_k1 * gg_sat_analy(dndm[idx_gg], uk_s[idx_gg], pop_s[idx_gg], ngal[idx_gg], beta, mass_range)
        else:
            Pgg_s = F_k1 * np.zeros(Pgg_c.shape)
        
        if ingredients['centrals'] and ingredients['satellites']:
            Pgg_cs = F_k1 * gg_cen_sat_analy(dndm[idx_gg], uk_s[idx_gg], pop_c[idx_gg], pop_s[idx_gg], ngal[idx_gg], mass_range)
        else:
            Pgg_cs = F_k1 * np.zeros(Pgg_c.shape)
        
        if ingredients['haloexclusion'] and setup['return'] != 'power':
            Pgg_k_t = Pgg_c + (2.0 * Pgg_cs) + Pgg_s
            Pgg_k = Pgg_c + (2.0 * Pgg_cs) + Pgg_s + Pgg_2h
        else:
            Pgg_k = Pgg_c + (2.0 * Pgg_cs) + Pgg_s + Pgg_2h
    
    # Matter - matter spectra
    if ingredient_mm:
        if ingredients['twohalo']:
            Pmm_2h = F_k2 * array([hmf_i.power for hmf_i in hmf[idx_mm]])
        else:
            Pmm_2h = np.zeros((nbins_mm,setup['lnk_bins']))
            
        if ingredients['centrals']:
            Pmm_1h = F_k1 * mm_analy(dndm[idx_mm], uk_c[idx_mm], rho_bg[idx_mm], mass_range)
        else:
            Pmm_1h = np.zeros((nbins_mm,setup['lnk_bins']))
          
        #if ingredients['haloexclusion'] and setup['return'] != 'power':
        #    Pmm_k_t = Pmm_1h
        #    Pmm_k = Pmm_1h + Pmm_2h
        #else:
        Pmm_k = Pmm_1h + Pmm_2h
    
    # Outputs
           
    if ingredient_gm:
        # note this doesn't include the point mass! also, we probably
        # need to return k
        if setup['return'] == 'power':
            if integrate_zlens:
                Pgm_k = np.sum(z*Pgm_k, axis=1) / intnorm
            P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                for Pg_k_i in Pgm_k]
        else:
            if integrate_zlens:
                if ingredients['haloexclusion'] and setup['return'] != 'power':
                    P_inter = [[UnivariateSpline(k_range, logPg_ij, s=0, ext=0)
                        for logPg_ij in logPg_i] for logPg_i in np.log(Pgm_k_t)]
                    P_inter_2h = [[UnivariateSpline(k_range, logPg_ij, s=0, ext=0)
                        for logPg_ij in logPg_i] for logPg_i in np.log(Pgm_2h)]
                else:
                    P_inter = [[UnivariateSpline(k_range, logPg_ij, s=0, ext=0)
                        for logPg_ij in logPg_i] for logPg_i in np.log(Pgm_k)]
            else:
                if ingredients['haloexclusion'] and setup['return'] != 'power':
                    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                        for Pg_k_i in Pgm_k_t]
                    P_inter_2h = [UnivariateSpline(k_range, np.log(Pg_2h_i), s=0, ext=0)
                        for Pg_2h_i in Pgm_2h]
                else:
                    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                        for Pg_k_i in Pgm_k]
                   
    if ingredient_gg:
        if ingredients['haloexclusion'] and setup['return'] != 'power':
            P_inter_2 = [UnivariateSpline(k_range, np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in Pgg_k_t]
            P_inter_2_2h = [UnivariateSpline(k_range, np.log(Pgg_2h_i), s=0, ext=0)
                    for Pgg_2h_i in Pgg_2h]
        else:
            P_inter_2 = [UnivariateSpline(k_range, np.log(Pgg_k_i), s=0, ext=0)
                    for Pgg_k_i in Pgg_k]
                    
    if ingredient_mm:
        #if ingredients['haloexclusion'] and setup['return'] != 'power':
        #    P_inter_3 = [UnivariateSpline(k_range, np.log(Pmm_k_i), s=0, ext=0)
        #            for Pmm_k_i in Pmm_k_t]
        #    P_inter_3_2h = [UnivariateSpline(k_range, np.log(Pmm_2h_i), s=0, ext=0)
        #            for Pmm_2h_i in Pmm_2h]
        #else:
        P_inter_3 = [UnivariateSpline(k_range, np.log(Pmm_k_i), s=0, ext=0)
                for Pmm_k_i in Pmm_k]
    
    
    if ingredient_gm:
        if setup['return'] == 'all':
            output[idx_gm] = Pgm_k
        if setup['return'] == 'power':
            Pgm_out = [exp(P_i(np.log(r_i))) for P_i, r_i in zip(P_inter, rvir_range_2d_i_gm)]
            output[idx_gm] = Pgm_out
    if ingredient_gg:
        if setup['return'] == 'all':
            output[idx_gg] = Pgg_k
        if setup['return'] == 'power':
            Pgg_out = [exp(P_i(np.log(r_i))) for P_i, r_i in zip(P_inter_2, rvir_range_2d_i_gg)]
            output[idx_gg] = Pgg_out
    if ingredient_mm:
        if setup['return'] == 'all':
            output[idx_mm] = Pmm_k
        if setup['return'] == 'power':
            Pmm_out = [exp(P_i(np.log(r_i))) for P_i, r_i in zip(P_inter_3, rvir_range_2d_i_mm)]
            output[idx_mm] = Pmm_out
    if setup['return'] == 'power':
        output = list(output)
        output = [output, meff]
        return output
    elif setup['return'] == 'all':
        output.append(k_range_lin)
    else:
        pass
    
    # correlation functions
    if ingredient_gm:
        if integrate_zlens:
            if ingredients['haloexclusion']:
                xi2 = np.array(
                    [[power_to_corr_ogata(P_inter_ji, rvir_range_3d)
                    for P_inter_ji in P_inter_j] for P_inter_j in P_inter])
                xi2_2h = np.array(
                    [[power_to_corr_ogata(P_inter_ji, rvir_range_3d)
                    for P_inter_ji in P_inter_j] for P_inter_j in P_inter_2h])
                xi2 = xi2 + halo_exclusion(xi2_2h, rvir_range_3d, meff[idx_gm], rho_bg[idx_gm], setup['delta'])
            else:
                xi2 = np.array(
                    [[power_to_corr_ogata(P_inter_ji, rvir_range_3d)
                    for P_inter_ji in P_inter_j] for P_inter_j in P_inter])
        else:
            if ingredients['haloexclusion']:
                xi2 = np.array(
                    [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                    for P_inter_i in P_inter])
                xi2_2h = np.array(
                    [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                    for P_inter_i in P_inter_2h])
                xi2 = xi2 + halo_exclusion(xi2_2h, rvir_range_3d, meff[idx_gm], rho_bg[idx_gm], setup['delta'])
            else:
                xi2 = np.array(
                    [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                    for P_inter_i in P_inter])
        # not yet allowed
        if setup['return'] == 'xi':
            if integrate_zlens:
                xi2 = np.sum(z*xi2, axis=1) / intnorm
            xi_out_i = array([UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0) for si in zip(xi2)])
            xi_out = np.array([x_i(r_i) for x_i, r_i in zip(xi_out_i, rvir_range_2d_i_gm)])
            output[idx_gm] = xi_out
        if setup['return'] == 'all':
            if integrate_zlens:
                xi2 = np.sum(z*xi2, axis=1) / intnorm
            output.append(xi2)
    
    if ingredient_gg:
        if ingredients['haloexclusion']:
            xi2_2 = np.array(
                [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                for P_inter_i in P_inter_2])
            xi2_2_2h = np.array(
                [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                for P_inter_i in P_inter_2_2h])
            xi2_2 = xi2_2 + halo_exclusion(xi2_2_2h, rvir_range_3d, meff[idx_gg], rho_bg[idx_gg], setup['delta'])
        else:
            xi2_2 = np.array(
                [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                for P_inter_i in P_inter_2])
        if setup['return'] == 'xi':
            xi_out_i_2 = array([UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0) for si in zip(xi2_2)])
            xi_out_2 = np.array([x_i(r_i) for x_i, r_i in zip(xi_out_i_2, rvir_range_2d_i_gg)])
            output[idx_gg] = xi_out_2
        if setup['return'] == 'all':
            output.append(xi2_2)
        
    if ingredient_mm:
        #if ingredients['haloexclusion']:
        #    xi2_3 = np.array(
        #        [power_to_corr_ogata(P_inter_i, rvir_range_3d)
        #        for P_inter_i in P_inter_3])
        #    xi2_3_2h = np.array(
        #        [power_to_corr_ogata(P_inter_i, rvir_range_3d)
        #        for P_inter_i in P_inter_3_2h])
        #    xi2_3 = xi2_3 + halo_exclusion(xi2_3_2h, rvir_range_3d, meff[idx_mm], rho_bg[idx_mm], setup['delta'])
        #else:
        xi2_3 = np.array(
            [power_to_corr_ogata(P_inter_i, rvir_range_3d)
            for P_inter_i in P_inter_3])
        if setup['return'] == 'xi':
            xi_out_i_3 = array([UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0) for si in zip(xi2_3)])
            xi_out_3 = np.array([x_i(r_i) for x_i, r_i in zip(xi_out_i_3, rvir_range_2d_i_mm)])
            output[idx_mm] = xi_out_3
        if setup['return'] == 'all':
            output.append(xi2_3)
            
    if setup['return'] == 'xi':
        output = list(output)
        output = [output, meff]
        return output
    elif setup['return'] == 'all':
        output.append(rvir_range_3d)
    else:
        pass
    
    # projected surface density
    # this is the slowest part of the model
    #
    # do we require a double loop here when weighting n(zlens)?
    # perhaps should not integrate over zlens at the power spectrum level
    # but only here -- or even just at the return stage!
    #
    # this avoids the interpolation necessary for better
    # accuracy of the ESD when returning sigma or kappa
    #rvir_sigma = rvir_range_2d_i if setup['return'] in ('sigma', 'kappa') \
    #    else rvir_range_3d_i

    if ingredient_gm:
        if integrate_zlens:
            surf_dens2 = array(
                [[sigma(xi2_ij, rho_bg_i, rvir_range_3d, rvir_range_3d_i)
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
                [sigma(xi2_i, rho_i, rvir_range_3d, rvir_range_3d_i)
                for xi2_i, rho_i in zip(xi2, rho_bg)])
             
        # units of Msun/pc^2
        if setup['return'] in ('sigma', 'kappa') and ingredients['pointmass']:
            pointmass = c_pm[1]/(2*np.pi) * array(
                [10**m_pm / r_i**2 for m_pm, r_i in zip(c_pm[0], rvir_range_2d_i_gm)])
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
        
        # in Msun/pc^2
        if not setup['return'] == 'kappa':
            surf_dens2 /= 1e12
    
        # fill/interpolate nans
        surf_dens2[(surf_dens2 <= 0) | (surf_dens2 >= 1e20)] = np.nan
        for i in range(nbins_gm):
            surf_dens2[i] = fill_nan(surf_dens2[i])
        if setup['return'] in ('kappa', 'sigma'):
            surf_dens2_r = array(
                [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(si), s=0)
                for si in surf_dens2])
            surf_dens2 = array([s_r(r_i) for s_r, r_i in zip(surf_dens2_r, rvir_range_2d_i_gm)])
            #return [surf_dens2, meff]
            output[idx_gm] = surf_dens2
        if setup['return'] == 'all':
            output.append(surf_dens2)
            
    if ingredient_gg:
        surf_dens2_2 = array(
            [sigma(xi2_i, rho_i, rvir_range_3d, rvir_range_3d_i)
            for xi2_i, rho_i in zip(xi2_2, rho_bg)])
        
        if setup['distances'] == 'proper':
            surf_dens2_2 *= (1+zo)**2
        
        # in Msun/pc^2
        if not setup['return'] in ('kappa', 'wp', 'esd_wp'):
            surf_dens2_2 /= 1e12
        
        # fill/interpolate nans
        surf_dens2_2[(surf_dens2_2 <= 0) | (surf_dens2_2 >= 1e20)] = np.nan
        for i in range(nbins_gg):
            surf_dens2_2[i] = fill_nan(surf_dens2_2[i])
        if setup['return'] in ('kappa', 'sigma'):
            surf_dens2_2_r = array(
                [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(si), s=0)
                for si in surf_dens2_2])
            surf_dens2_2 = np.array([s_r(r_i) for s_r, r_i in zip(surf_dens2_2_r, rvir_range_2d_i_gg)])
        if setup['return'] in ('kappa', 'sigma'):
            output[idx_gg] = surf_dens2_2
        if setup['return'] in ('wp', 'esd_wp'):
            wp_out_i = array([UnivariateSpline(rvir_range_3d_i, np.nan_to_num(wi/rho_i), s=0)
                        for wi, rho_i in zip(surf_dens2_2, rho_bg)])
            wp_out = [wp_i(r_i) for wp_i, r_i in zip(wp_out_i, rvir_range_2d_i_gg)]
            #output.append(wp_out)
        if setup['return'] == 'all':
            wp_out = surf_dens2_2/expand_dims(rho_bg, -1)
            output.append([surf_dens2_2, wp_out])
    
    
    if ingredient_mm:
        surf_dens2_3 = array([sigma(xi2_i, rho_i, rvir_range_3d, rvir_range_3d_i)
            for xi2_i, rho_i in zip(xi2_3, rho_bg)])
        
        if setup['distances'] == 'proper':
            surf_dens2_3 *= (1+zo)**2
        
        # in Msun/pc^2
        if not setup['return'] == 'kappa':
            surf_dens2_3 /= 1e12
        
        # fill/interpolate nans
        surf_dens2_3[(surf_dens2_3 <= 0) | (surf_dens2_3 >= 1e20)] = np.nan
        for i in range(nbins_mm):
            surf_dens2_3[i] = fill_nan(surf_dens2_3[i])
        if setup['return'] in ('kappa', 'sigma'):
            surf_dens2_3_r = array(
                [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(si), s=0)
                for si in surf_dens2_3])
            surf_dens2_3 = array([s_r(r_i) for s_r, r_i in zip(surf_dens2_3_r, rvir_range_2d_i_mm)])
            #return [surf_dens2_3, meff]
            output[idx_mm] = surf_dens2_3
        if setup['return'] == 'all':
            output.append(surf_dens2_3)
        
    if setup['return'] in ('kappa', 'sigma'):
        output = list(output)
        output = [output, meff]
        return output
    if setup['return'] == ('wp'):
        output = [wp_out, meff]
        return output
    elif setup['return'] == 'all':
        output.append(rvir_range_3d_i)
    elif setup['return'] == 'esd_wp':
        pass
    else:
        pass
    
    
    # excess surface density
    """
    # These two options are not really used for any observable! Keeping in for now.
    
    if ingredients['mm']:
        d_surf_dens2_3 = array(
            [np.nan_to_num(
            d_sigma(surf_dens2_i, rvir_range_3d_i, r_i))
            for surf_dens2_i, r_i in zip(surf_dens2_3, rvir_range_2d_i_mm)])

        out_esd_tot_3 = array(
            [UnivariateSpline(r_i, np.nan_to_num(d_surf_dens2_i), s=0)
            for d_surf_dens2_i, r_i in zip(d_surf_dens2_3, r_i)])
    
        #out_esd_tot_inter_3 = np.zeros((nbins, rvir_range_2d_i_mm[0].size))
        #for i in range(nbins):
        #    out_esd_tot_inter_3[i] = out_esd_tot_3[i](rvir_range_2d_i_mm[i])
        out_esd_tot_inter_3 = [out_esd_tot_3[i](rvir_range_2d_i_mm[i]) for i in range(nbins_mm)]
        output.insert(0, out_esd_tot_inter_3) # This insert makes sure that the ESD's are on the fist place.
    

    if ingredients['gg']:
        d_surf_dens2_2 = array(
                [np.nan_to_num(
                d_sigma(surf_dens2_i, rvir_range_3d_i, r_i))
                for surf_dens2_i, r_i in zip(surf_dens2_2, rvir_range_2d_i_gg)])

        out_esd_tot_2 = array(
            [UnivariateSpline(r_i, np.nan_to_num(d_surf_dens2_i), s=0)
             for d_surf_dens2_i, r_i in zip(d_surf_dens2_2, rvir_range_2d_i)])
        
        #out_esd_tot_inter_2 = np.zeros((nbins, rvir_range_2d_i_gg.size))
        #for i in range(nbins):
        #    out_esd_tot_inter_2[i] = out_esd_tot_2[i](rvir_range_2d_i_gg[i])
        out_esd_tot_inter_2 = [out_esd_tot_2[i](rvir_range_2d_i_gg[i]) for i in range(nbins_gg)]
        output.insert(0, out_esd_tot_inter_2)
    """
        
    if ingredient_gm:
        d_surf_dens2 = array(
            [np.nan_to_num(
                d_sigma(surf_dens2_i, rvir_range_3d_i, r_i))
            for surf_dens2_i, r_i in zip(surf_dens2, rvir_range_2d_i_gm)])
            
        out_esd_tot = array(
            [UnivariateSpline(r_i, np.nan_to_num(d_surf_dens2_i), s=0)
            for d_surf_dens2_i, r_i in zip(d_surf_dens2, rvir_range_2d_i_gm)])
    
        #out_esd_tot_inter = np.zeros((nbins, rvir_range_2d_i.size))
        #for i in range(nbins):
        #    out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)
        out_esd_tot_inter = [out_esd_tot[i](rvir_range_2d_i_gm[i]) for i in range(nbins_gm)]
        # this should be moved to the power spectrum calculation
        if ingredients['pointmass']:
            # the 1e12 here is to convert Mpc^{-2} to pc^{-2} in the ESD
            pointmass = c_pm[1]/(np.pi*1e12) * array(
                [10**m_pm / (r_i**2) for m_pm, r_i in zip(c_pm[0], rvir_range_2d_i_gm)])
            out_esd_tot_inter = [out_esd_tot_inter[i] + pointmass[i] for i in range(nbins_gm)]
        if setup['return'] == 'esd_wp':
            output[idx_gm] = out_esd_tot_inter
            output[idx_gg] = wp_out
            output = list(output)
            output = [output, meff]
            print(output)
        else:
            output = [out_esd_tot_inter, meff]
    
    return output
    

if __name__ == '__main__':
    print(0)
