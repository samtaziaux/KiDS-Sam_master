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
    fill_nan, load_hmf, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, sigma_crit,
    power_to_corr_ogata, wp, wp_beta_correction)
from .dark_matter import (
    mm_analy, gm_cen_analy, gm_sat_analy, gg_cen_analy,
    gg_sat_analy, gg_cen_sat_analy, two_halo_gm)
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

    tstart = time()

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
    F_k1 = sp.erf(k_range_lin/0.1)
    F_k2 = sp.erfc(k_range_lin/1500.0)
    #F_k1 = np.ones_like(k_range_lin)
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
    if ingredients['miscentring']:
        #ti = time()
        p_off, r_off = c_miscent#[1:]
        # these should be implemented as iterables
        #if not iterable(p_off):
            #p_off = array([p_off]*nbins)
        #if not iterable(r_off):
            #r_off = array([r_off]*nbins)
        uk_c = uk_c * nfw.miscenter(
            p_off, r_off, expand_dims(mass_range, -1),
            expand_dims(rvir_range_lin, -1), k_range_lin,
            expand_dims(concentration, -1), uk_c.shape)
        #print('miscentring in {0:.2e} s'.format(time()-ti))
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
        Pgm_2h = F_k2 * bias * array(
            [two_halo_gm(hmf_i, ngal_i, pop_g_i, mass_range)[0]
             for hmf_i, ngal_i, pop_g_i
             in zip(hmf, expand_dims(ngal, -1),
                    expand_dims(pop_g, -2))])
        #print('Pg_2h in {0:.2e} s'.format(time()-ti))
    #elif integrate_zlens:
        #Pg_2h = np.zeros((nbins,z.size//nbins,setup['lnk_bins']))
    else:
        Pgm_2h = np.zeros((nbins,setup['lnk_bins']))
        #if integrate_zlens:
            #Pg_2h = Pg_2h[:,None]
    #print('Pg_2h =', Pg_2h.shape)

    if not integrate_zlens:
        rho_bg = rho_bg[...,0]

    if ingredients['centrals']:
        #ti = time()
        Pgm_c = F_k1 * gm_cen_analy(
            dndm, uk_c, rho_bg, pop_c, ngal, mass_range)
        #print('Pg_c in {0:.2e} s'.format(time()-ti))
    elif integrate_zlens:
        Pgm_c = np.zeros((z.size,nbins,setup['lnk_bins']))
    else:
        Pgm_c = np.zeros((nbins,setup['lnk_bins']))
    #else:
        #Pg_c = np.zeros(Pg_2h.shape)
    #print('Pg_c =', Pg_c.shape)

    if ingredients['satellites']:
        #ti = time()
        Pgm_s = F_k1 * gm_sat_analy(
            dndm, uk_c, uk_s, rho_bg, pop_s, ngal, mass_range)
        #print('Pg_s in {0:.2e} s'.format(time()-ti))
    else:
        Pgm_s = np.zeros(Pgm_c.shape)

    #print('Pg_i =', Pg_c.shape, Pg_s.shape, Pg_2h.shape)
    #print('nan(Pg_i) =', np.isnan(Pg_c).sum(), np.isnan(Pg_s).sum(),
          #np.isnan(Pg_2h).sum())
    Pgm_k = Pgm_c + Pgm_s + Pgm_2h
    #print('Pg_k =', Pg_k.shape, '- nan:', np.isnan(Pg_k).sum())

    # finally integrate over (weight by, really) lens redshift
    if integrate_zlens:
        intnorm = np.sum(nz, axis=0)
        #print('intnorm =', intnorm.shape, nz.shape, meff.shape)
        meff = np.sum(nz*meff, axis=0) / intnorm
    #print('meff =', np.squeeze(meff), meff.shape)

    # not yet allowed
    if setup['return'] == 'power':
        # note this doesn't include the point mass! also, we probably
        # need to return k
        if integrate_zlens:
            Pgm_k = np.sum(z*Pgm_k, axis=1) / intnorm
        P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
            for Pg_k_i in Pgm_k]
    else:
        if integrate_zlens:
            P_inter = [[UnivariateSpline(k_range, logPg_ij, s=0, ext=0)
                    for logPg_ij in logPg_i] for logPg_i in np.log(Pgm_k)]
        else:
            P_inter = [UnivariateSpline(k_range, np.log(Pg_k_i), s=0, ext=0)
                   for Pg_k_i in Pgm_k]
    if setup['return'] == 'power':
        Pgm_out = [exp(P_i(np.log(rvir_range_2d_i))) for P_i in P_inter]
        return [Pgm_out, meff]

    """
    for i, Pk in enumerate(Pg_k):
        plt.loglog(k_range_lin, Pk, label=r'$P_{0}(k)$'.format(i))
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.show()
    """

    """
    if calculate_covariance:
        # this is a single number, in units 1/radians. Need to multiply
        # by the area in each annulus - in radians, of course -- and the
        # number of annuli (i.e., lenses) I presume?
        shape = np.array(
            [covariance.shape_noise(i.cosmo, zi, rho_i, covar, i)
             for i, zi, rho_i in zip(hmf, z, rho_bg)])
        print('shape =', shape)
        return
    """

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
        xi_out_i = array([UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0) for si in zip(xi2)])
        xi_out = np.array([x_i(rvir_range_2d_i) for x_i in xi_out_i])
        return [xi_out, meff]

    debug = False
    if debug:
        np.set_printoptions(formatter={'float': lambda x: format(x, '6.4F')})
        print('R =', repr(np.log10(rvir_range_3d)))
        print('xi =', repr(np.log10(xi2[0])))
        plt.loglog(rvir_range_3d, xi2[0])
        plt.show()

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
    """
    if setup['return'] in ('sigma', 'kappa'):
        surf_dens2 = array(
            [sigma(xi2_i, rho_i, rvir_range_3d, rvir_range_2d_i)
             for xi2_i, rho_i in zip(xi2, rho_bg)])
        if ingredients['pointmass']:
            pointmass = 3*c_pm[1]/(2*np.pi*1e12) * array(
                [10**m_pm / rvir_range_2d_i**2 for m_pm in c_pm[0]])
            surf_dens2 = surf_dens2 + pointmass
    """
    #print('xi2 =', xi2.shape, xi2.size, np.isnan(xi2).sum())
    #print('rvir:', rvir_range_3d_i.shape, rvir_range_2d_i.shape)
    ti = time()
    if integrate_zlens:
        surf_dens2 = array(
            [[sigma(xi2_ij, rho_bg_i, rvir_range_3d, rvir_sigma)
              for xi2_ij in xi2_i] for xi2_i, rho_bg_i in zip(xi2, rho_bg)])
        #print('surf_dens2 pt 1 in {0:.2e} s'.format(time()-ti))
        # integrate lens redshift
        # is this right?
        #ti = time()
        #print('surf_dens2 =', surf_dens2.shape)
        #print('nz =', nz.shape)
        #print('z =', z.shape)
        #print('surf_dens2 in {0:.2e} s'.format(time()-ti))
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
            #[sigma(xi2_i, rho_i, rvir_range_3d, rvir_range_3d_i)
             #for xi2_i, rho_i in zip(xi2, rho_bg)])
            [sigma(xi2_i, rho_i, rvir_range_3d, rvir_sigma)
             for xi2_i, rho_i in zip(xi2, rho_bg)])
    
    #print('surf_dens2 in {0:.2e} s'.format(time()-ti))
    #if setup['return'] == 'kappa':
        #print('sigma_crit =', sigma_crit(cosmo_model, z, zs).T)
    #print('surf_dens2[0] =', surf_dens2[0], surf_dens2.size, np.isnan(surf_dens2).sum())
    #xplot = rvir_range_2d_i if rvir_range_2d_i.size == surf_dens2.shape[-1] \
        #else rvir_range_3d_i
    #for i, x in enumerate(xi2):
        ##plt.loglog(rvir_range_3d, x, label=r'$\xi_{0}(R)$'.format(i))
        ##plt.loglog(xplot, surf_dens2[i]/1e12, label=r'$\Sigma_{0}(R)$'.format(i))
        #plt.loglog(rvir_range_3d, x, label=r'$\xi(r)$')
        #plt.loglog(xplot, surf_dens2[i]/1e12, label=r'$\Sigma(R)$')
    ##for r in rvir_range_2d_i:
        ##plt.axvline(r, ls='-', lw=1, color='0.5')
    #plt.grid()
    #plt.xlabel('R (Mpc)')
    #plt.legend()
    #plt.show()

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
    #ti = time()
    d_surf_dens2 = array(
        [np.nan_to_num(
            d_sigma(surf_dens2_i, rvir_range_3d_i, rvir_range_2d_i))
         for surf_dens2_i in surf_dens2])
    #print('d_surf_dens2 in {0:.2e} s'.format(time()-ti))
    #ti = time()
    out_esd_tot = array(
        [UnivariateSpline(rvir_range_2d_i, np.nan_to_num(d_surf_dens2_i), s=0)
         for d_surf_dens2_i in d_surf_dens2])
    #print('splines in {0:.2e} s'.format(time()-ti))
    #ti = time()
    out_esd_tot_inter = np.zeros((nbins, rvir_range_2d_i.size))
    for i in range(nbins):
        out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)
    #print('esd in {0:.2e} s'.format(time()-ti))

    # this should be moved to the power spectrum calculation
    if ingredients['pointmass']:
        # the 1e12 here is to convert Mpc^{-2} to pc^{-2} in the ESD
        pointmass = c_pm[1]/(np.pi*1e12) * array(
            [10**m_pm / (rvir_range_2d_i**2) for m_pm in c_pm[0]])
        out_esd_tot_inter = out_esd_tot_inter + pointmass

    # this should also probably be moved higher up!
    #if setup['distances'] == 'proper':
        #out_esd_tot_inter = out_esd_tot_inter * (1+z)**2

    # Add other outputs as needed. Total ESD should always be first!
    return [out_esd_tot_inter, meff]
    #return out_esd_tot_inter, d_surf_dens3, d_surf_dens4, pointmass, nu(1)


if __name__ == '__main__':
    print(0)
