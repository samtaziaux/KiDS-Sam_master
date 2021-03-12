#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from numpy import (arange, array, exp, expand_dims, iterable,
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
from astropy.units import Quantity

from hmf import MassFunction
import hmf.mass_function.fitting_functions as ff
import hmf.density_field.transfer_models as tf

from . import baryons, longdouble_utils as ld, nfw
from .tools import (
    fill_nan, load_hmf, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, sigma_crit,
    power_to_corr_ogata, wp, wp_beta_correction, power_to_sigma, power_to_sigma_ogata)
from .dark_matter import (
    mm_analy, gm_cen_analy, gm_sat_analy, gg_cen_analy,
    gg_sat_analy, gg_cen_sat_analy, two_halo_gm, two_halo_gg, halo_exclusion, beta_nl)
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


def model(theta, R):

    # ideally we would move this to somewhere separate later on
    preamble(theta, R)

    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]

    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta

    cosmo_model, sigma8, n_s, z = load_cosmology(cosmo)

    ### a final bit of massaging ###

    if observables.mlf:
        nbins = observables.nbins - observables.mlf.nbins
    else:
        nbins = observables.nbins
    output = np.empty(observables.nbins, dtype=object)

    if ingredients['nzlens']:
        nz = cosmo[9].T
        size_cosmo = 10
    else:
        nz = None
        # hard-coded
        size_cosmo = 9

    if observables.mlf:
        z_mlf = cosmo[-1]
        size_cosmo += 1

    # cheap hack. I'll use this for CMB lensing, but we can
    # also use this to account for difference between shear
    # and reduced shear
    if len(cosmo) == size_cosmo+1:
        zs = cosmo[-1]
    else:
        zs = None

    z = format_z(z, nbins)
    ### load halo mass functions ###

    # Tinker10 should also be read from theta!
    hmf, rho_bg = load_hmf(z, setup, cosmo_model, sigma8, n_s)

    assert np.allclose(setup['mass_range'], hmf[0].m)
    # alias (should probably get rid of it)
    mass_range = setup['mass_range']

    """Calculating halo model"""

    # can probably move this into populations()
    completeness = calculate_completeness(
        observables, selection, ingredients, z)
    pop_c, pop_s = populations(
        observables, ingredients, completeness, mass_range, theta)
    pop_g = pop_c + pop_s

    # note that pop_g already accounts for incompleteness
    dndm = array([hmf_i.dndm for hmf_i in hmf])
    ngal, meff = calculate_ngal(observables, pop_g, dndm, mass_range)

    # Luminosity or mass function as an output:
    if observables.mlf:
        output[observables.mlf.idx] = calculate_mlf(
            z_mlf, observables, ingredients, mass_range, theta, setup, cosmo_model, sigma8, n_s)

    ### Power spectra ###

    uk_c, uk_s = calculate_uk(
        setup, observables, ingredients, z, mass_range, rho_bg,
        c_concentration, s_concentration, c_miscent)

    if not ingredients['nzlens']:
        rho_bg = rho_bg[...,0]

    # spectra that are not required are just dummy variables
    Pgm, Pgg, Pmm = calculate_power_spectra(
            setup, observables, ingredients, hmf,mass_range, dndm, rho_bg,
            c_twohalo, s_beta, pop_g, pop_c, pop_s, uk_c, uk_s, ngal, z)
    if observables.gm:
        Pgm_c, Pgm_s, Pgm_2h = Pgm
        Pgm_1h = Pgm_c + Pgm_s
        Pgm = (Pgm_1h, Pgm_2h) if ingredients['haloexclusion'] \
            else Pgm_1h + Pgm_2h
    if observables.gg:
        ncen = hod.nbar(dndm[observables.gg.idx], pop_c[observables.gg.idx],
                        mass_range)
        nsat = hod.nbar(dndm[observables.gg.idx], pop_s[observables.gg.idx],
                        mass_range)
        Pgg_c, Pgg_s, Pgg_cs, Pgg_2h = Pgg
        Pgg_1h = Pgg_c + 2*Pgg_cs + Pgg_s
        Pgg = (Pgg_1h, Pgg_2h) if ingredients['haloexclusion'] \
            else Pgg_1h + Pgg_2h
    if observables.mm:
        Pmm_1h, Pmm_2h = Pmm
        Pmm = Pmm_1h + Pmm_2h

    ### interpolate power spectra ###

    Pgm_func, Pgg_func, Pmm_func = power_as_interp(
        setup, observables, ingredients, Pgm, Pgg, Pmm)

    if observables.gm:
        if setup['return'] == 'all':
            output[observables.gm.idx] = Pgm_k
        if setup['return'] == 'power':
            Pgm_out = [exp(P_i(logr_i)) for P_i, logr_i
                       in zip(Pgm_func, np.log(observables.gm.R))]
            output[observables.gm.idx] = Pgm_out

    if observables.gg:
        if setup['return'] == 'all':
            output[observables.gg.idx] = Pgg_k
        if setup['return'] == 'power':
            Pgg_out = [exp(P_i(logr_i)) for P_i, logr_i
                       in zip(Pgg_func, np.log(observables.gg.R))]
            output[observables.gg.idx] = Pgg_out

    if observables.mm:
        if setup['return'] == 'all':
            output[observables.mm.idx] = Pmm_k
        if setup['return'] == 'power':
            Pmm_out = [exp(P_i(logr_i)) for P_i, logr_i
                       in zip(Pmm_func, np.log(observables.mm.R))]
            output[observables.mm.idx] = Pmm_out

    if setup['return'] == 'power':
        output = list(output)
        output = [output, meff]
        return output
    elif setup['return'] == 'all':
        output.append(setup['k_range_lin'])
    else:
        pass

    ### correlation functions ###

    xi_gm, xi_gg, xi_mm = calculate_correlations(
        setup, observables, ingredients, Pgm_func, Pgg_func, Pmm_func, rho_bg,
        meff)

    if setup['return'] in ('all', 'xi'):
        output = output_xi(
            setup, observables, ingredients, xi_gm, xi_gg, xi_mm, z, nz)

        if setup['return'] == 'xi':
            output = list(output)
            output = [output, meff]
            return output
        else:
            output.append(rvir_range_3d)

    ### projected surface density ###

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

    # alias

    output, sigma_gm, sigma_gg, sigma_mm = calculate_surface_density(
        setup, observables, ingredients, xi_gm, xi_gg, xi_mm,
        rho_bg, z, nz, c_pm, output, cosmo_model, zs)

    if setup['return'] in ('wp', 'esd_wp'):
        wp_out_i = np.array(
            [UnivariateSpline(setup['rvir_range_3d_interp'], np.nan_to_num(wi/rho_i),
                              s=0)
             for wi, rho_i in zip(sigma_gg, rho_bg)])
        wp_out = [wp_i(r_i) for wp_i, r_i
                  in zip(wp_out_i, observables.gg.R)]
        #output.append(wp_out)
    if setup['return'] == 'all':
        wp_out = sigma_gg/expand_dims(rho_bg, -1)
        output.append([sigma_gg, wp_out])

    if setup['return'] in ('kappa', 'sigma'):
        #output = list(output)
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

    ### excess surface density ###

    if observables.gm:
        esd_gm = calculate_esd(setup, observables, ingredients, sigma_gm, c_pm)
        if setup['return'] == 'esd_wp':
            output[observables.gm.idx] = esd_gm
            # I need to move this to when wp is calculated
            output[observables.gg.idx] = wp_out
            output = list(output)
            output = [output, meff]
        else:
            output = [esd_gm, meff]

    # Finally!
    return output


#############################
##                         ##
##   auxiliary functions   ##
##                         ##
#############################


def calculate_completeness(observables, selection, ingredients, z):
    # interpolate selection function to the same grid as redshift and
    # observable to be used in trapz
    if selection.filename == 'None':
        if ingredients['nzlens']:
            completeness = np.ones(
                (z.size,observables.nbins,observables.sampling.shape[1]))
        else:
            completeness = np.ones(observables.sampling.shape)
    elif ingredients['nzlens']:
        completeness = np.array(
            [[selection.interpolate([zi]*obs.size, obs, method='linear')
              for obs in observables.sampling] for zi in z])
    else:
        completeness = np.array(
            [selection.interpolate([zi]*obs.size, obs, method='linear')
             for zi, obs in zip(z, observables.sampling)])
    # shape ([z.size,]nbins,sampling)
    if ingredients['nzlens']:
        assert completeness.shape \
            == (z.size,observables.nbins,observables.sampling.shape[1])
    else:
        assert completeness.shape == observables.sampling.shape
    return completeness


def calculate_correlations_single(setup, observable, ingredients, Pk_func,
                                  rho_bg, meff):
    """Calculate correlation functions for a single observable

    Note this takes a single observable rather than all observables as
    usual
    """
    if ingredients['haloexclusion']:
        Pk_func, Pk_2h_func = Pk_func
    # how best to test whether there are one or two dimensions?
    if not np.iterable(Pk_func[0]):
        Pk_func = [Pk_func]
        if ingredients['haloexclusion']:
            Pk_2h_func = [Pk_2h_func]
    xi2 = np.array([[power_to_corr_ogata(Pk_func_ij, setup['rvir_range_3d'])
                     for Pk_func_ij in Pk_func_i] for Pk_func_i in Pk_func])
    # only gm and gg because the matter correlation does not know about
    # halos I guess?
    if ingredients['haloexclusion'] and observable.obstype in ('gm','gg'):
        xi2_2h = [[power_to_corr_ogata(Pk_func_ij, setup['rvir_range_3d'])
                   for Pk_func_ij in Pk_func_i] for Pk_func_i in Pk_2h_func]
        xi2 = xi2 + halo_exclusion(
            xi2_2h, setup['rvir_range_3d'], meff[observable.idx],
            rho_bg[observable.idx], setup['delta'])
    if len(xi2) == 1:
        xi2 = xi2[0]
    return xi2


def calculate_correlations(setup, observables, ingredients, Pgm_func,
                           Pgg_func, Pmm_func, rho_bg, meff):
    xi2_gm = None
    xi2_gg = None
    xi2_mm = None
    if observables.gm:
        xi2_gm = calculate_correlations_single(
            setup, observables.gm, ingredients, Pgm_func, rho_bg, meff)
    if observables.gg:
        xi2_gg = calculate_correlations_single(
            setup, observables.gg, ingredients, Pgg_func, rho_bg, meff)
    if observables.mm:
        xi2_mm = calculate_correlations_single(
            setup, observables.mm, ingredients, Pgg_func, rho_bg, meff)
    return xi2_gm, xi2_gg, xi2_mm


def calculate_esd_single(setup, observable, surface_density):
    esd = array(
        [np.nan_to_num(d_sigma(sigma_i, setup['rvir_range_3d_interp'], R_i))
         for sigma_i, R_i in zip(surface_density, observable.R)])
    esd_func = array(
        [UnivariateSpline(R_i, np.nan_to_num(esd_i), s=0)
         for esd_i, R_i in zip(esd, observable.R)])
    esd = [esd_func_i(R_i)
           for esd_func_i, R_i in zip(esd_func, observable.R)]
    return esd


def calculate_esd(setup, observables, ingredients, sigma_gm, c_pm):
    """This is only ever used for gm, but adding the others would
    be trivial if necessary"""
    if observables.gm:
        """
        d_sigma_gm = array(
            [np.nan_to_num(
                d_sigma(sigma_gm_i, rvir_range_3d_i, r_i))
            for sigma_gm_i, r_i in zip(sigma_gm, observables.gm.R)])

        out_esd_tot = array(
            [UnivariateSpline(r_i, np.nan_to_num(d_sigma_gm_i), s=0)
            for d_sigma_gm_i, r_i in zip(d_sigma_gm, observables.gm.R)])

        #out_esd_tot_inter = np.zeros((nbins, rvir_range_2d_i.size))
        #for i in range(nbins):
        #    out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)
        out_esd_tot_inter = [out_esd_tot[i](observables.gm.R[i])
                             for i in range(observables.gm.nbins)]
        """
        esd = calculate_esd_single(setup, observables.gm, sigma_gm)
        if ingredients['pointmass']:
            # the 1e12 here is to convert Mpc^{-2} to pc^{-2} in the ESD
            pointmass = c_pm[1]/(np.pi*1e12) * array(
                [10**m_pm / (r_i**2)
                 for m_pm, r_i in zip(c_pm[0], observables.gm.R)])
            esd = [esd[i] + pointmass[i] for i in range(observables.gm.nbins)]
    return esd


def output_esd_single(output, observable, esd, meff):
    if setup['return'] in ('esd', 'esd_wp'):
        output[observables.gm.idx] = esd


def calculate_mlf(z_mlf, observables, ingredients, mass_range, theta, setup, cosmo_model, sigma8, n_s):
    if z_mlf.size == 1 and observables.mlf.nbins > 1:
        z_mlf = z_mlf*np.ones(observables.mlf.nbins)
    if z_mlf.size != observables.mlf.nbins:
        raise ValueError(
            'Number of redshift bins should be equal to the number of' \
            ' observable bins!')
    hmf_mlf, _rho_mean = load_hmf(z_mlf, setup, cosmo_model, sigma8, n_s)
    dndm_mlf = array([hmf_i.dndm for hmf_i in hmf_mlf])

    c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta[1:]

    pop_c_mlf = np.zeros(observables.mlf.sampling.shape)
    pop_s_mlf = np.zeros(observables.mlf.sampling.shape)

    if ingredients['centrals']:
        pop_c_mlf = hod.mlf(
            observables.mlf.sampling, dndm_mlf, mass_range, c_mor[0],
            c_scatter[0], c_mor[1:], c_scatter[1:],
            obs_is_log=observables.mlf.is_log)

    if ingredients['satellites']:
        pop_s_mlf = hod.mlf(
            observables.mlf.sampling, dndm_mlf, mass_range, s_mor[0],
            s_scatter[0], s_mor[1:], s_scatter[1:],
            obs_is_log=observables.mlf.is_log)
    pop_g_mlf = pop_c_mlf + pop_s_mlf

    mlf_inter = [UnivariateSpline(hod_i, np.log(ngal_i), s=0, ext=0)
                 for hod_i, ngal_i
                 in zip(observables.mlf.sampling,
                        pop_g_mlf*10.0**observables.mlf.sampling)]
    for i, Ri in enumerate(observables.mlf.R):
        Ri = Quantity(Ri, unit='Mpc')
        observables.mlf.R[i] = Ri.to(setup['R_unit']).value
    mlf_out = [exp(mlf_i(np.log10(r_i))) for mlf_i, r_i
               in zip(mlf_inter, observables.mlf.R)]
    return mlf_out


def calculate_ngal(observables, pop_g, dndm, mass_range):
    ngal = np.empty(observables.nbins)
    meff = np.empty(observables.nbins)
    if observables.gm:
        ngal[observables.gm.idx] = hod.nbar(
            dndm[observables.gm.idx], pop_g[observables.gm.idx], mass_range)
        meff[observables.gm.idx] = hod.Mh_effective(
            dndm[observables.gm.idx], pop_g[observables.gm.idx], mass_range,
            return_log=observables.gm.is_log)
    if observables.gg:
        ngal[observables.gg.idx] = hod.nbar(
            dndm[observables.gg.idx], pop_g[observables.gg.idx], mass_range)
        meff[observables.gg.idx] = hod.Mh_effective(
            dndm[observables.gg.idx], pop_g[observables.gg.idx], mass_range,
            return_log=observables.gg.is_log)
    if observables.mm:
        ngal[observables.mm.idx] = np.zeros_like(observables.mm.nbins)
        meff[observables.mm.idx] = np.zeros_like(observables.mm.nbins)
    return ngal, meff


def calculate_Pgg(setup, observables, ingredients, hmf, mass_range, dndm,
                  bias, pop_g, pop_c, pop_s, uk_s, ngal, beta, F_k1, F_k2, Igg):
    
    if ingredients['twohalo']:
        Pgg_2h = F_k2 * bias * array(
        [two_halo_gg(hmf_i, ngal_i, pop_g_i, mass_range)[0]
        for hmf_i, ngal_i, pop_g_i
        in zip(hmf[observables.gg.idx],
                expand_dims(ngal[observables.gg.idx], -1),
                expand_dims(pop_g[observables.gg.idx], -2))])
        if ingredients['bnl']:
            Pgg_2h = (Pgg_2h + array([hmf_i.power for hmf_i in hmf[observables.gg.idx]])*Igg)
    else:
        Pgg_2h = F_k2 * np.zeros((observables.gg.nbins,setup['lnk_bins']))

    #if ingredients['centrals']:
        #Pgg_c = F_k1 * np.zeros((observables.gg.nbins,setup['lnk_bins']))
    #else:
    Pgg_c = F_k1 * np.zeros((observables.gg.nbins,setup['lnk_bins']))

    if ingredients['satellites']:
        Pgg_s = F_k1 * gg_sat_analy(
            dndm[observables.gg.idx], uk_s[observables.gg.idx],
            pop_s[observables.gg.idx], ngal[observables.gg.idx], beta,
            mass_range)
    else:
        Pgg_s = F_k1 * np.zeros(Pgg_c.shape)

    if ingredients['centrals'] and ingredients['satellites']:
        Pgg_cs = F_k1 * gg_cen_sat_analy(
            dndm[observables.gg.idx], uk_s[observables.gg.idx],
            pop_c[observables.gg.idx], pop_s[observables.gg.idx],
            ngal[observables.gg.idx], mass_range)
    else:
        Pgg_cs = F_k1 * np.zeros(Pgg_c.shape)

    return Pgg_c, Pgg_s, Pgg_cs, Pgg_2h


def calculate_Pgm(setup, observables, ingredients, hmf, mass_range, dndm,
                    rho_bg, bias, pop_g, pop_c, pop_s, uk_c, uk_s, ngal,
                    F_k1, F_k2, Igm):
    
    if ingredients['twohalo']:
        """
        # unused but not removing as we might want to use it later
        #bias_out = bias.T[0] * array(
            #[TwoHalo(hmf_i, ngal_i, pop_g_i, setup['k_range_lin'], rvir_range_lin_i,
                 #mass_range)[1]
            #for rvir_range_lin_i, hmf_i, ngal_i, pop_g_i
            #in zip(rvir_range_lin, hmf, ngal, pop_g)])
        """
        Pgm_2h = F_k2 * bias * array(
            [two_halo_gm(hmf_i, ngal_i, pop_g_i, mass_range)[0]
            for hmf_i, ngal_i, pop_g_i
            in zip(hmf[observables.gm.idx],
                   expand_dims(ngal[observables.gm.idx], -1),
                   expand_dims(pop_g[observables.gm.idx], -2))])
        if ingredients['bnl']:
            Pgm_2h = (Pgm_2h + array([hmf_i.power for hmf_i in hmf[observables.gm.idx]])*Igm)
    #elif ingredients['nzlens']:
        #Pg_2h = np.zeros((nbins,z.size//nbins,setup['lnk_bins']))
    else:
        Pgm_2h = np.zeros((observables.gm.nbins,setup['lnk_bins']))

    if ingredients['centrals']:
        Pgm_c = F_k1 * gm_cen_analy(
            dndm[observables.gm.idx], uk_c[observables.gm.idx],
            rho_bg[observables.gm.idx], pop_c[observables.gm.idx],
            ngal[observables.gm.idx], mass_range)
    elif ingredients['nzlens']:
        Pgm_c = np.zeros((z.size,observables.gm.nbins,setup['lnk_bins']))
    else:
        Pgm_c = F_k1 * np.zeros((observables.gm.nbins,setup['lnk_bins']))

    if ingredients['satellites']:
        Pgm_s = F_k1 * gm_sat_analy(
            dndm[observables.gm.idx], uk_c[observables.gm.idx],
            uk_s[observables.gm.idx], rho_bg[observables.gm.idx],
            pop_s[observables.gm.idx], ngal[observables.gm.idx],
            mass_range)
    else:
        Pgm_s = F_k1 * np.zeros(Pgm_c.shape)
    return Pgm_c, Pgm_s, Pgm_2h


def calculate_Pmm(setup, observables, ingredients, hmf, mass_range, dndm,
                  uk_c, F_k1, F_k2, Imm):
    if ingredients['twohalo']:
        Pmm_2h = F_k2 * array([hmf_i.power
                               for hmf_i in hmf[observables.mm.idx]])
    else:
        Pmm_2h = np.zeros((observables.mm.nbins,setup['lnk_bins']))

    if ingredients['centrals']:
        Pmm_1h = F_k1 * mm_analy(
            dndm[observables.mm.idx], uk_c[observables.mm.idx],
            rho_bg[observables.mm.idx], mass_range)
    else:
        Pmm_1h = np.zeros((observables.mm.nbins,setup['lnk_bins']))
    return Pmm_1h, Pmm_2h


def calculate_power_spectra(setup, observables, ingredients, hmf, mass_range,
                            dndm, rho_bg, c_twohalo, s_beta, pop_g, pop_c,
                            pop_s, uk_c, uk_s, ngal, z):
    """Wrapper to calculate gm, gg, and/or mm power spectra"""
    # Galaxy - dark matter spectra (for lensing)
    bias = c_twohalo
    bias = array([bias]*setup['k_range_lin'].size).T
    if setup['delta_ref'] == 'SOCritical':
        bias = bias * omegam

    # damping of the 1h power spectra at small k
    F_k1 = sp.erf(setup['k_range_lin']/0.1)
    F_k2 = np.ones_like(setup['k_range_lin'])
    #F_k2 = sp.erfc(setup['k_range_lin']/10.0)

    output = [[], [], []]

    if ingredients['bnl']:
        from .tools import read_mead_data
        # read in Alex Mead BNL table:
        beta_interp = read_mead_data()
        #import dill as pickle
        #with open('/net/home/fohlen12/dvornik/interpolator_BNL_test_i.npy', 'rb') as dill_file:
        #    beta_interp = pickle.load(dill_file)

    if observables.gm:
        if ingredients['bnl']:
            Igm = array([beta_nl(hmf_i, pop_g_i, mass_range, ngal_i, rho_bg_i,
                        mass_range, beta_interp, setup['k_range_lin'], z_i)
                        for hmf_i, pop_g_i, ngal_i, rho_bg_i, z_i in
                            zip(hmf[observables.gm.idx], pop_g[observables.gm.idx],
                            ngal[observables.gm.idx], rho_bg[observables.gm.idx],
                            z[observables.gm.idx])])
        else:
            Igm = None
        Pgm_c, Pgm_s, Pgm_2h = calculate_Pgm(
            setup, observables, ingredients, hmf, mass_range, dndm, rho_bg,
            bias, pop_g, pop_c, pop_s, uk_c, uk_s, ngal, F_k1, F_k2, Igm)
        if ingredients['haloexclusion'] and setup['return'] != 'power':
            Pgm_k_t = Pgm_c + Pgm_s
            Pgm_k = Pgm_c + Pgm_s + Pgm_2h
        else:
            Pgm_k = Pgm_c + Pgm_s + Pgm_2h
        # finally integrate over (weight by, really) lens redshift
        if ingredients['nzlens']:
            meff[observables.gm.idx] \
                = np.sum(nz*meff[observables.gm.idx], axis=0) \
                    / np.sum(nz, axis=0)
        output[0] = (Pgm_c, Pgm_s, Pgm_2h)

    # Galaxy - galaxy spectra (for clustering)
    if observables.gg:
        if ingredients['bnl']:
            Igg = array([beta_nl(hmf_i, pop_g_i, pop_g_i, ngal_i, ngal_i,
                        mass_range, beta_interp, setup['k_range_lin'], z_i)
                        for hmf_i, pop_g_i, ngal_i, z_i in
                            zip(hmf[observables.gg.idx], pop_g[observables.gg.idx],
                            ngal[observables.gg.idx], z[observables.gg.idx])])
        else:
            Igg = None
        Pgg_c, Pgg_s, Pgg_cs, Pgg_2h = calculate_Pgg(
            setup, observables, ingredients, hmf, mass_range, dndm, bias,
            pop_g, pop_c, pop_s, uk_s, ngal, s_beta, F_k1, F_k2, Igg)

        if ingredients['haloexclusion'] and setup['return'] != 'power':
            Pgg_k_t = Pgg_c + 2*Pgg_cs + Pgg_s
            Pgg_k = Pgg_c + 2*Pgg_cs + Pgg_s + Pgg_2h
        else:
            Pgg_k = Pgg_c + 2*Pgg_cs + Pgg_s + Pgg_2h
        output[1] = (Pgg_c, Pgg_s, Pgg_cs, Pgg_2h)

    # Matter - matter spectra
    if observables.mm:
        Pmm_1h, Pmm_2h = calculate_Pmm(
            setup, observables, ingredients, hmf, mass_range, dndm, uk_c,
            F_k1, F_k2, Imm=None)
        #if ingredients['haloexclusion'] and setup['return'] != 'power':
        #    Pmm_k_t = Pmm_1h
        #    Pmm_k = Pmm_1h + Pmm_2h
        #else:
        #Pmm_k = Pmm_1h + Pmm_2h
        output[2] = (Pmm_1h, Pmm_2h)

    return output


def calculate_surface_density_single(setup, observable, ingredients, xi2,
                                     rho_bg, z, nz, c_pm, output,
                                     cosmo_model, zs):
    if observable.obstype == 'gm' and ingredients['nzlens']:
        surface_density = array(
            [[sigma(xi2_ij, rho_bg_i, setup['rvir_range_3d'],
                    setup['rvir_range_3d_interp'])
            for xi2_ij in xi2_i] for xi2_i, rho_bg_i in zip(xi2, rho_bg)])
        z = expand_dims(z, -1)
    else:
        surface_density = array(
            [sigma(xi2_i, rho_i, setup['rvir_range_3d'],
             setup['rvir_range_3d_interp'])
            for xi2_i, rho_i in zip(xi2, rho_bg)])

    # esd pointmass is added at the end
    if observable.obstype == 'gm' and ingredients['pointmass'] \
            and setup['return'] in ('sigma', 'kappa'):
        pointmass = c_pm[1]/(2*np.pi) * array(
            [10**m_pm / r_i**2
             for m_pm, r_i in zip(c_pm[0], observable.R)])
        surface_density = surface_density + pointmass

    if setup['distances'] == 'proper':
        surface_density *= (1+z)**2

    if observable.obstype == 'gm':
        if setup['return'] == 'kappa':
            surface_density /= sigma_crit(cosmo_model, z, zs)
        if ingredients['nzlens']:
            # haven't checked the denominator below
            norm = trapz(nz, z, axis=0)
            surface_density = \
                trapz(surface_density * expand_dims(nz, -1), z[:,0], axis=0) \
                / norm[:,None]
            zw = nz * sigma_crit(cosmo_model, z, zs) \
                if setup['return'] == 'kappa' else nz
            zeff = trapz(zw*z, z, axis=0) / trapz(zw, z, axis=0)
        # in Msun/pc^2
        if setup['return'] != 'kappa':
            surface_density /= 1e12
    if observable.obstype == 'gg' \
            and setup['return'] not in ('kappa', 'wp', 'esd_wp'):
        surface_density /= 1e12
    if observable.obstype == 'mm' and setup['return'] != 'kappa':
        surface_density /= 1e12

    # fill/interpolate nans
    mask = (surface_density <= 0) | (surface_density >= 1e20)
    surface_density[mask] = np.nan
    for i in range(observable.nbins):
        surface_density[i] = fill_nan(surface_density[i])
    if setup['return'] in ('kappa', 'sigma'):
        surface_density_r = array(
            [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(si), s=0)
             for si in surface_density])
        surface_density = np.array(
            [s_r(r_i) for s_r, r_i in zip(surface_density_r, observable.R)])
        #return [sigma_gm, meff]
        if observable.nbins == 1:
            output[observable.idx.start] = surface_density[0]
        else:
            output[observable.idx] = surface_density

    return output, surface_density


def calculate_surface_density(setup, observables, ingredients,
                              xi_gm, xi_gg, xi_mm, rho_bg, z, nz, c_pm,
                              output, cosmo_model, zs):
    # dummy
    sigma_gm = None
    sigma_gg = None
    sigma_mm = None

    if observables.gm:
        output, sigma_gm = calculate_surface_density_single(
            setup, observables.gm, ingredients, xi_gm, rho_bg, z, nz, c_pm,
            output, cosmo_model, zs)
        if setup['return'] == 'all':
            output.append(sigma_gm)
    if observables.gg:
        output, sigma_gg = calculate_surface_density_single(
            setup, observables.gg, ingredients, xi_gg, rho_bg, z, nz, c_pm,
            output, cosmo_model, zs)
        if setup['return'] == 'all':
            output.append(sigma_gg)
    if observables.mm:
        output, sigma_mm = calculate_surface_density_single(
            setup, observables.mm, ingredients, xi_mm, rho_bg, z, nz, c_pm,
            output, cosmo_model, zs)
        if setup['return'] == 'all':
            output.append(sigma_mm)

    return output, sigma_gm, sigma_gg, sigma_mm


def calculate_uk(setup, observables, ingredients, z, mass_range, rho_bg,
                 c_concentration, s_concentration, c_miscent):
    rvir_range_lin = virial_radius(mass_range, rho_bg, setup['delta'])
    # Fourier Transform of the NFW profile
    if ingredients['centrals']:
        concentration = c_concentration[0](mass_range, *c_concentration[1:])
        uk_c = nfw.uk(
            setup['k_range_lin'], mass_range, rvir_range_lin, concentration,
            rho_bg, setup['delta'])
    elif ingredients['nzlens']:
        uk_c = np.ones((observables.nbins, z.size//observables.nbins,
                        mass_range.size, setup['k_range_lin'].size))
    else:
        uk_c = np.ones((observables.nbins, mass_range.size,
                        setup['k_range_lin'].size))
    # and of the NFW profile of the satellites
    if ingredients['satellites']:
        concentration_sat = s_concentration[0](
            mass_range, *s_concentration[1:])
        uk_s = nfw.uk(
            setup['k_range_lin'], mass_range, rvir_range_lin,
            concentration_sat, rho_bg, setup['delta'])
        uk_s = uk_s / expand_dims(uk_s[...,0], -1)
    elif ingredients['nzlens']:
        uk_s = np.ones((observables.nbins, z.size//observables.nbins,
                        mass_range.size, setup['k_range_lin'].size))
    else:
        uk_s = np.ones((observables.nbins, mass_range.size,
                        setup['k_range_lin'].size))

    # If there is miscentring to be accounted for
    # Only for galaxy-galaxy lensing!
    if ingredients['miscentring']:
        p_off, r_off = c_miscent[1:]
        uk_c[observables.gm.idx] = uk_c[observables.gm.idx] * nfw.miscenter(
            p_off, r_off, expand_dims(mass_range, -1),
            expand_dims(rvir_range_lin, -1), setup['k_range_lin'],
            expand_dims(concentration, -1), uk_c[observables.gm.idx].shape)

    uk_c = uk_c / expand_dims(uk_c[...,0], -1)

    return uk_c, uk_s


def load_cosmology(cosmo):
    sigma8, h, omegam, omegab, n_s, w0, wa, Neff, z = cosmo[:9]
    cosmo_model = Flatw0waCDM(
        H0=100*h, Ob0=omegab, Om0=omegam, Tcmb0=2.725, m_nu=0.06*eV,
        Neff=Neff, w0=w0, wa=wa)
    return cosmo_model, sigma8, n_s, z


def interpolate_xi_single(observable, rvir_range_3d, xi):
    xi_out_interp = array(
        [UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0)
         for si in zip(xi)])
    xi_out = np.array(
        [x_i(r_i) for x_i, r_i in zip(xi_out_interp, observable.R)])
    return xi_out


def output_xi_single(output, setup, observable, xi, nz=None):
    if setup['return'] == 'xi':
        if ingredients['nzlens'] and observable.obstype == 'gm':
            xi = np.sum(nz*xi, axis=1) / np.sum(nz, axis=0)
        xi_out = interpolate_xi_single(observable, setup['rvir_range_3d'], xi)
        output[observable.idx] = xi_out
    if setup['return'] == 'all':
        if ingredients['nzlens']:
            xi = np.sum(nz*xi, axis=1) / np.sum(nz, axis=0)
        output.append(xi)
    return output


def output_xi(output, setup, observables, ingredients, xi_gm, xi_gg, xi_mm,
              nz=None):
    """NOT TESTED"""
    if observables.gm:
        output = output_xi_single(output, setup, observables.gm, xi_gm, nz=nz)
    if observables.gg:
        output = output_xi_single(output, setup, observables.gg, xi_gg)
    if observables.mm:
        output = output_xi_single(output, setup, observables.mm, xi_mm)
    return output


def populations(observables, ingredients, completeness, mass_range, theta):
    pop_c, pop_s = np.zeros((2,observables.nbins,mass_range.size))

    c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta[1:]

    if observables.gm:
        if ingredients['centrals']:
            pop_c[observables.gm.idx,:], prob_c_gm = hod.number(
                observables.gm.sampling, mass_range, c_mor[0], c_scatter[0],
                c_mor[1:], c_scatter[1:], completeness[observables.gm.idx],
                obs_is_log=observables.gm.is_log)
        if ingredients['satellites']:
            pop_s[observables.gm.idx,:], prob_s_gm = hod.number(
                observables.gm.sampling, mass_range, s_mor[0], s_scatter[0],
                s_mor[1:], s_scatter[1:], completeness[observables.gm.idx],
                obs_is_log=observables.gm.is_log)

    if observables.gg:
        if ingredients['centrals']:
            pop_c[observables.gg.idx,:], prob_c_gg = hod.number(
                observables.gg.sampling, mass_range, c_mor[0], c_scatter[0],
                c_mor[1:], c_scatter[1:], completeness[observables.gg.idx],
                obs_is_log=observables.gg.is_log)
        if ingredients['satellites']:
            pop_s[observables.gg.idx,:], prob_s_gg = hod.number(
                observables.gg.sampling, mass_range, s_mor[0], s_scatter[0],
                s_mor[1:], s_scatter[1:], completeness[observables.gg.idx],
                obs_is_log=observables.gg.is_log)
    return pop_c, pop_s


def power_as_interp_single(logk, logPk, s=0, ext=0):
    if len(logPk.shape) == 3:
        logPk_func = [[UnivariateSpline(logk, logPk_ij, s=s, ext=ext)
                       for logPk_ij in logPk_i] for logPk_i in logPk]
    else:
        logPk_func = [UnivariateSpline(logk, logPk_i, s=s, ext=ext)
                      for logPk_i in logPk]
    return logPk_func


def power_as_interp(setup, observables, ingredients, Pgm, Pgg, Pmm, nz=None):
    """Produce UnivariateSplines of log P(log k)"""
    # dummy variables in case any are disabled
    Pgm_func = None
    Pgg_func = None
    Pmm_func = None
    # galaxy-matter
    if observables.gm:
        # in this case Pgm stands for Pgm_1h
        if ingredients['haloexclusion']:
            Pgm, Pgm_2h = Pgm
        if setup['return'] == 'power':
            if ingredients['haloexclusion']:
                Pgm = Pgm + Pgm_2h
            if ingredients['nzlens']:
                Pgm = np.sum(nz*Pgm, axis=1) / np.sum(nz, axis=0)
            Pgm_func = power_as_interp_single(setup['k_range'], np.log(Pgm))
        else:
            Pgm_func = power_as_interp_single(setup['k_range'], np.log(Pgm))
            if ingredients['haloexclusion']:
                Pgm_2h_func = power_as_interp_single(
                    setup['k_range'], np.log(Pgm_2h))
                Pgm_func = (Pgm_func, Pgm_2h_func)
    # galaxy-galaxy
    if observables.gg:
        # in this case Pgg stands for Pgg_1h
        if ingredients['haloexclusion']:
            Pgg, Pgg_2h = Pgg
        Pgg_func = power_as_interp_single(setup['k_range'], np.log(Pgg))
        if ingredients['haloexclusion']:
            Pgg_2h_func = power_as_interp_single(
                setup['k_range'], np.log(Pgg_2h))
            Pgg_func = (Pgg_func, Pgg_2h_func)
    # matter-matter
    if observables.mm:
        Pmm_func = power_as_interp_single(setup['k_range'], np.log(Pgg))

    return Pgm_func, Pgg_func, Pmm_func


def preamble(theta, R):
    """Preamble function

    This function is specified separately in the configuration file
    and is called only once when initializing the sampler module,
    rather than at every step in the MCMC. Include here all variable
    tests, for instance.

    This function does not return anything
    """
    np.seterr(
        divide='ignore', over='ignore', under='ignore', invalid='ignore')

    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]
    cosmo = theta[0]
    sigma8, h, omegam, omegab, n, w0, wa, Neff, z = cosmo[:9]

    if observables.mlf:
        nbins = observables.nbins - observables.mlf.nbins
    else:
        nbins = observables.nbins
    output = np.empty(observables.nbins, dtype=object)

    if ingredients['nzlens']:
        assert len(cosmo) >= 11, \
            'When integrating nzlens, must provide an additional parameter' \
            '"nz", corresponding to the histogram of lens redshifts. See' \
            'demo for an example.'
        nz = cosmo[9].T
        size_cosmo = 10
    else:
        # hard-coded
        size_cosmo = 9
    
    if observables.mlf:
        if len(cosmo) == size_cosmo+1:
            assert len(cosmo) >= len(cosmo), \
                'When using SMF/LF, must provide an additional parameter' \
                '"z_mlf", corresponding to mean redshift values for SMF/LF. See' \
                'demo for an example.'
        z_mlf = cosmo[-1]
        size_cosmo += 1
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

    return


def format_z(z, nbins):
    # if a single value is given for more than one bin, assign same
    # value to all bins
    if z.size == 1 and nbins > 1:
        z = z*np.ones(nbins)
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    z = expand_dims(z, -1)
    return z


if __name__ == '__main__':
    print(0)
