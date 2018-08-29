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
from numpy import (
    arange, array, exp, iterable, linspace, logspace, newaxis, ones)
from scipy import special as sp
from scipy.integrate import simps, trapz, quad
from scipy.interpolate import interp1d, UnivariateSpline
from itertools import count
if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange
from time import time
from astropy.cosmology import LambdaCDM

from hmf import MassFunction
from hmf import fitting_functions as ff
from hmf import transfer_models as tf

from . import baryons, longdouble_utils as ld, nfw
from .tools import (
    Integrate, Integrate1, extrap1d, extrap2d, fill_nan, gas_concentration,
    star_concentration, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, power_to_corr_ogata,
    wp, wp_beta_correction)
from .dark_matter import (
    DM_mm_spectrum, GM_cen_spectrum, GM_sat_spectrum,
    delta_NFW, MM_analy, GM_cen_analy, GM_sat_analy, GG_cen_analy,
    GG_sat_analy, GG_cen_sat_analy, miscenter, TwoHalo)
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

"""
# Integrals of mass functions with density profiles and population functions.
"""

def f_k(k_x):
    F = sp.erf(k_x/0.1) #0.05!
    return F


#################
##
## Main function
##
#################


def model(theta, R):

    np.seterr(
        divide='ignore', over='ignore', under='ignore', invalid='ignore')

    # new config
    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]

    assert len(observables) == 1, \
        'working with more than one observable is not yet supported.' \
        ' If you would like this feature added please raise an issue.'
    observable = observables[0]

    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter = theta

    sigma8, H0, omegam, omegab, omegav, n, z = cosmo

    # HMF set up parameters
    # all of this can happen before the model is called, to save some
    # time
    k_step = (setup['lnk_max']-setup['lnk_min']) / setup['lnk_bins']
    k_range = arange(setup['lnk_min'], setup['lnk_max'], k_step)
    k_range_lin = exp(k_range)
    # endpoint must be False for mass_range to be equal to hmf.m
    mass_range = 10**linspace(
        setup['logM_min'], setup['logM_max'], setup['logM_bins'],
        endpoint=False)
    mstep = (setup['logM_max'] - setup['logM_min']) / setup['logM_bins']

    # this is the observable section
    nbins = observable.nbins
    # this whole setup thing should be done outside of the model,
    # only once when setting up the sampler basically
    #if not iterable(f):
        #f = array([f]*nbins)
    #if not iterable(fc_nsat):
        #fc_nsat = array([fc_nsat]*nbins)
    if z.size == 1 and nbins > 1:
        z = array(list(z)*nbins)
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter
    z = z[:,newaxis]

    # note that e.g. duffy08_crit takes h as a kwarg but we're not passing it
    # here! This would just be accounted for by allowing a free normalization
    # but it would become very error-prone
    concentration = c_concentration[0](mass_range, *c_concentration[1:])
    concentration_sat = s_concentration[0](mass_range, *s_concentration[1:])

    hod_observable = observable.sampling

    transfer_params = array([])
    for z_i in z:
        transfer_params = np.append(
            transfer_params,
            {'sigma_8': sigma8, 'n': n,
             'lnk_min': setup['lnk_min'], 'lnk_max': setup['lnk_max'],
             'dlnk': k_step, 'z': np.float64(z_i)})

    # Calculation
    # Tinker10 should also be read from theta!
    hmf = array([])
    h = H0 / 100.0
    cosmo_model = LambdaCDM(
        H0=H0, Ob0=omegab, Om0=omegam, Ode0=omegav, Tcmb0=2.725)
    for i in transfer_params:
        hmf_temp = MassFunction(
            Mmin=setup['logM_min'], Mmax=setup['logM_max'], dlog10m=mstep,
            hmf_model=ff.Tinker10, delta_h=setup['delta'],
            delta_wrt=setup['delta_ref'], delta_c=1.686, **i)
        hmf_temp.update(cosmo_model=cosmo_model)
        hmf = np.append(hmf, hmf_temp)

    mass_func = np.zeros((z.size, mass_range.size))
    rho_mean = np.zeros(z.shape)
    rho_crit = np.zeros(z.shape)

    omegab = hmf[0].cosmo.Ob0
    omegac = hmf[0].cosmo.Om0-omegab
    omegav = hmf[0].cosmo.Ode0

    for i in range(z.size):
        mass_func[i] = hmf[i].dndlnm
        rho_mean[i] = hmf[i].mean_density0
        rho_crit[i] = rho_mean[i] / (omegac+omegab)
        #rho_dm[i] = rho_mean[i] * baryons.f_dm(omegab, omegac)
    rho_bg = rho_mean if setup['delta_ref'] == 'mean' else rho_crit

    rvir_range_lin = array([virial_radius(mass_range, rho_i, setup['delta'])
                            for rho_i in rho_bg])
    rvir_range = np.log10(rvir_range_lin)
    rvir_range_3d = logspace(-3.2, 4, 200, endpoint=True)
    rvir_range_3d_i = logspace(-2.5, 1.2, 25, endpoint=True)
    rvir_range_2d_i = R[0][1:]

    """Calculating halo model"""

    # interpolate selection function to the same grid as redshift and
    # observable to be used in trapz
    if selection.filename == 'None':
        completeness = np.ones_like(hod_observable)
    else:
        completeness = np.array(
            [selection.interpolate([zi]*obs.size, obs, method='linear')
             for zi, obs in zip(z[:,0], hod_observable)])

    if ingredients['centrals']:
        pop_c = hod.number(
            hod_observable, mass_range, c_mor[0], c_scatter[0],
            c_mor[1:], c_scatter[1:], completeness,
            obs_is_log=observable.is_log)
    else:
        pop_c = np.zeros((nbins,mass_range.size))

    if ingredients['satellites']:
        pop_s = hod.number(
            hod_observable, mass_range, s_mor[0], s_scatter[0],
            s_mor[1:], s_scatter[1:], completeness,
            obs_is_log=observable.is_log)
    else:
        pop_s = np.zeros((nbins,mass_range.size))

    pop_g = pop_c + pop_s

    # note that pop_g already accounts for incompleteness
    mass_function = array([hmf_i.dndm for hmf_i in hmf])
    ngal = hod.nbar(mass_function, pop_g, mass_range)
    meff = hod.Mh_effective(
        mass_function, pop_g, mass_range, return_log=observable.is_log)

    """Power spectrum"""

    # damping of the 1h power spectra at small k
    F_k1 = f_k(k_range_lin)
    # Fourier Transform of the NFW profile
    if ingredients['centrals']:
        uk_c = array(
            [nfw.uk(k_range_lin, mass_range, rvir_range_lin_i,
                    ci, zi, rho_i, setup['delta'])
             for rvir_range_lin_i, ci, zi, rho_i
             in zip(rvir_range_lin, concentration, z[:,0], rho_bg)])
    else:
        uk_c = np.ones((nbins,k_range_lin.size,mass_range.size))
    # and of the NFW profile of the satellites
    if ingredients['satellites']:
        uk_s = array(
            [nfw.uk(k_range_lin, mass_range, rvir_range_lin_i,
                    ci, zi, rho_i, setup['delta'])
             for rvir_range_lin_i, ci, zi, rho_i
             in zip(rvir_range_lin, concentration, z[:,0], rho_bg)])
        uk_s = uk_s/uk_s[:,0][:,None]
    else:
        uk_s = np.ones((nbins,k_range_lin.size,mass_range.size))

    # If there is miscentring to be accounted for
    if ingredients['miscentring']:
        p_off, r_off = c_miscent[1:]
        if not iterable(p_off):
            p_off = array([p_off]*nbins)
        if not iterable(r_off):
            r_off = array([r_off]*nbins)
        uk_c = uk_c * array(
            [miscenter(p_off_i, r_off_i, mass_range, rvir_range_lin_i,
                       k_range_lin, concentration)
             for rvir_range_lin_i, p_off_i, r_off_i
             in zip(rvir_range_lin, p_off, r_off)])
    uk_c = uk_c / uk_c[:,0][:,None]

    # Galaxy - dark matter spectra (for lensing)
    bias = c_twohalo
    bias = array([bias]*k_range_lin.size).T
    if ingredients['twohalo']:
        Pg_2h = bias * array(
            [TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin, rvir_range_lin_i,
                     mass_range)[0]
             for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
             in zip(rvir_range_lin, hmf, ngal, pop_g)])
        # unused but not removing as we might want to use it later
        #bias_out = bias.T[0] * array(
            #[TwoHalo(hmf_i, ngal_i, pop_g_i, k_range_lin, rvir_range_lin_i,
                     #mass_range)[1]
             #for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
             #in zip(rvir_range_lin, hmf, ngal, pop_g)])
    else:
        Pg_2h = np.zeros((nbins,setup['lnk_bins']))

    if ingredients['centrals']:
        Pg_c = F_k1 * array(
            [GM_cen_analy(hmf_i, uk_c_i, rho_i, pop_c_i, ngal_i, mass_range)
             for rho_i, hmf_i, pop_c_i, ngal_i, uk_c_i
             in zip(rho_bg, hmf, pop_c, ngal, uk_c)])
    else:
        Pg_c = np.zeros((nbins,setup['lnk_bins']))

    if ingredients['satellites']:
        Pg_s = F_k1 * array(
            [GM_sat_analy(hmf_i, uk_c_i, uk_s_i, rho_i, pop_s_i, ngal_i,
                          mass_range)
             for rho_i, hmf_i, pop_s_i, ngal_i, uk_c_i, uk_s_i
             in zip(rho_bg, hmf, pop_s, ngal, uk_c, uk_s)])
    else:
        Pg_s = np.zeros((nbins,setup['lnk_bins']))

    Pg_k = array([(Pg_c_i+Pg_s_i) + Pg_2h_i
                  for Pg_c_i, Pg_s_i, Pg_2h_i in zip(Pg_c, Pg_s, Pg_2h)])
    P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
               for Pg_k_i in zip(Pg_k)]

    # correlation functions
    xi2 = np.zeros((nbins,rvir_range_3d.size))
    for i in range(nbins):
        xi2[i] = power_to_corr_ogata(P_inter[i], rvir_range_3d)

    # projected surface density
    surf_dens2 = array([sigma(xi2_i, rho_i, rvir_range_3d, rvir_range_3d_i)
                        for xi2_i, rho_i in zip(xi2, rho_bg)])
    for i in range(nbins):
        surf_dens2[i][(surf_dens2[i] <= 0.0) \
                         | (surf_dens2[i] >= 1e20)] = np.nan
        surf_dens2[i] = fill_nan(surf_dens2[i])

    # excess surface density
    d_surf_dens2 = array(
        [np.nan_to_num(
            d_sigma(surf_dens2_i, rvir_range_3d_i, rvir_range_2d_i))
         for surf_dens2_i in zip(surf_dens2)]) / 1e12
    out_esd_tot = array(
        [UnivariateSpline(rvir_range_2d_i, np.nan_to_num(d_surf_dens2_i), s=0)
         for d_surf_dens2_i in zip(d_surf_dens2)])
    out_esd_tot_inter = np.zeros((nbins, rvir_range_2d_i.size))
    for i in range(nbins):
        out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)

    if ingredients['pointmass']:
        # the 1e12 here is to convert Mpc^{-2} to pc^{-2} in the ESD
        pointmass = array(
            [c_pm[1]*10**m_pm / (np.pi*rvir_range_2d_i**2) / 1e12
             for m_pm in c_pm[0]])
        out_esd_tot_inter = out_esd_tot_inter + pointmass

    # Add other outputs as needed. Total ESD should always be first!

    if setup['distances'] == 'comoving':
        return [out_esd_tot_inter, meff]
    if setup['distances'] == 'proper':
        out_esd_tot_inter = array(
            [out_esd_tot_inter_i * (1+z_i)**2.0
             for out_esd_tot_inter_i, z_i in zip(out_esd_tot_inter, z)])
        return [out_esd_tot_inter, meff]
    #return out_esd_tot_inter, d_surf_dens3, d_surf_dens4, pointmass, nu(1)


if __name__ == '__main__':
    print(0)
