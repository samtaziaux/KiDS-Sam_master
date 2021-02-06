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


def model(theta, R): #, calculate_covariance=False):

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

    if observables.mlf:
        nbins = observables.nbins - observables.mlf.nbins
    else:
        nbins = observables.nbins
    output = np.empty(observables.nbins, dtype=object)

    if ingredients['nzlens']:
        nz = cosmo[9].T
        size_cosmo = 10
    else:
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

    integrate_zlens = ingredients['nzlens']

    # if a single value is given for more than one bin, assign same
    # value to all bins
    if z.size == 1 and nbins > 1:
        z = z*np.ones(nbins)
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    z = expand_dims(z, -1)

    # Tinker10 should also be read from theta!
    hmf, rho_bg = load_hmf(z, setup, cosmo_model, sigma8, n_s)

    assert np.allclose(setup['mass_range'], hmf[0].m)
    # alias
    mass_range = setup['mass_range']

    # same as with redshift
    rho_bg = expand_dims(rho_bg, -1)

    concentration = c_concentration[0](mass_range, *c_concentration[1:])
    if ingredients['satellites']:
        concentration_sat = s_concentration[0](
            mass_range, *s_concentration[1:])

    rvir_range_lin = virial_radius(
        mass_range, rho_bg, setup['delta'])
    # alias
    rvir_range_3d = setup['rvir_range_3d']
    rvir_range_3d_i = setup['rvir_range_3d_interp']

    """Calculating halo model"""

    # can probably move this into populations()
    completeness = calculate_completeness(
        z, observables, selection, ingredients)
    pop_c, pop_s = populations(
        observables, ingredients, completeness, mass_range, theta)
    pop_g = pop_c + pop_s

    # note that pop_g already accounts for incompleteness
    dndm = array([hmf_i.dndm for hmf_i in hmf])
    ngal, meff = calculate_ngal(observables, pop_g, dndm, mass_range)

    # Luminosity or mass function as an output:
    if observables.mlf:
        # Needs independent redshift input!
        #z_mlf = z[observables.mlf.idx]
        if z_mlf.size == 1 and observables.mlf.nbins > 1:
            z_mlf = z_mlf*np.ones(observables.mlf.nbins)
        if z_mlf.size != observables.mlf.nbins:
            raise ValueError(
                'Number of redshift bins should be equal to the number of' \
                ' observable bins!')
        hmf_mlf, _rho_mean = load_hmf(
            z_mlf, setup, cosmo_model, transfer_params)
        dndm_mlf = array([hmf_i.dndm for hmf_i in hmf_mlf])

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
        output[observables.mlf.idx] = mlf_out

    """Power spectra"""

    # damping of the 1h power spectra at small k
    F_k1 = sp.erf(setup['k_range_lin']/0.1)
    F_k2 = np.ones_like(setup['k_range_lin'])
    #F_k2 = sp.erfc(setup['k_range_lin']/10.0)
    # Fourier Transform of the NFW profile
    if ingredients['centrals']:
        uk_c = nfw.uk(
            setup['k_range_lin'], mass_range, rvir_range_lin, concentration,
            rho_bg, setup['delta'])
    elif integrate_zlens:
        uk_c = np.ones((nbins,z.size//nbins,mass_range.size,setup['k_range_lin'].size))
    else:
        uk_c = np.ones((nbins,mass_range.size,setup['k_range_lin'].size))
    # and of the NFW profile of the satellites
    if ingredients['satellites']:
        uk_s = nfw.uk(
            setup['k_range_lin'], mass_range, rvir_range_lin,
            concentration_sat, rho_bg, setup['delta'])
        uk_s = uk_s / expand_dims(uk_s[...,0], -1)
    elif integrate_zlens:
        uk_s = np.ones((nbins,z.size//nbins,mass_range.size,
                        setup['k_range_lin'].size))
    else:
        uk_s = np.ones((nbins,mass_range.size,setup['k_range_lin'].size))

    # If there is miscentring to be accounted for
    # Only for galaxy-galaxy lensing!
    if ingredients['miscentring']:
        p_off, r_off = c_miscent[1:]
        uk_c[observables.gm.idx] = uk_c[observables.gm.idx] * nfw.miscenter(
            p_off, r_off, expand_dims(mass_range, -1),
            expand_dims(rvir_range_lin, -1), setup['k_range_lin'],
            expand_dims(concentration, -1), uk_c[observables.gm.idx].shape)
    uk_c = uk_c / expand_dims(uk_c[...,0], -1)

    """
    # read in Alex Mead BNL table:
    print('Importing BNL pickle...')
    import dill as pickle
    with open('/net/home/fohlen12/dvornik/interpolator_BNL.npy', 'rb') as dill_file:
        beta_interp = pickle.load(dill_file)
    print(beta_interp([0.5, 12.3, 12.8, 1e-1]))
    
    Igm = array([beta_nl(hmf_i, pop_g_i, mass_range, ngal_i, rho_bg_i, mass_range, beta_interp, k_range_lin, z_i) for hmf_i, pop_g_i, ngal_i, rho_bg_i, z_i in zip(hmf[idx_gm], pop_g[idx_gm], ngal[idx_gm], rho_bg[idx_gm], z[idx_gm])])
    Igg = array([beta_nl(hmf_i, pop_g_i, pop_g_i, ngal_i, ngal_i, mass_range, beta_interp, k_range_lin, z_i) for hmf_i, pop_g_i, ngal_i, z_i in zip(hmf[idx_gg], pop_g[idx_gg], ngal[idx_gg], z[idx_gg])])
    """
    # Galaxy - dark matter spectra (for lensing)
    bias = c_twohalo
    bias = array([bias]*setup['k_range_lin'].size).T
    if setup['delta_ref'] == 'SOCritical':
        bias = bias * omegam

    if not integrate_zlens:
        rho_bg = rho_bg[...,0]

    if observables.gm:
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
            #print('Pg_2h in {0:.2e} s'.format(time()-ti))
        #elif integrate_zlens:
            #Pg_2h = np.zeros((nbins,z.size//nbins,setup['lnk_bins']))
        else:
            Pgm_2h = np.zeros((observables.gm.nbins,setup['lnk_bins']))

        if ingredients['centrals']:
            Pgm_c = F_k1 * gm_cen_analy(
                dndm[observables.gm.idx], uk_c[observables.gm.idx],
                rho_bg[observables.gm.idx], pop_c[observables.gm.idx],
                ngal[observables.gm.idx], mass_range)
        elif integrate_zlens:
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

        if ingredients['haloexclusion'] and setup['return'] != 'power':
            Pgm_k_t = Pgm_c + Pgm_s
            Pgm_k = Pgm_c + Pgm_s + Pgm_2h
        else:
            Pgm_k = Pgm_c + Pgm_s + Pgm_2h

        # finally integrate over (weight by, really) lens redshift
        if integrate_zlens:
            intnorm = np.sum(nz, axis=0)
            meff[observables.gm.idx] \
                = np.sum(nz*meff[observables.gm.idx], axis=0) / intnorm

    # Galaxy - galaxy spectra (for clustering)
    if observables.gg:
        if ingredients['twohalo']:
            Pgg_2h = F_k2 * bias * array(
            [two_halo_gg(hmf_i, ngal_i, pop_g_i, mass_range)[0]
            for hmf_i, ngal_i, pop_g_i
            in zip(hmf[observables.gg.idx],
                    expand_dims(ngal[observables.gg.idx], -1),
                    expand_dims(pop_g[observables.gg.idx], -2))])
        else:
            Pgg_2h = F_k2 * np.zeros((observables.gg.nbins,setup['lnk_bins']))

        ncen = hod.nbar(dndm[observables.gg.idx], pop_c[observables.gg.idx],
                        mass_range)
        nsat = hod.nbar(dndm[observables.gg.idx], pop_s[observables.gg.idx],
                        mass_range)

        if ingredients['centrals']:
            """
            Pgg_c = F_k1 * gg_cen_analy(
                dndm, ncen, ngal, (nbins,setup['lnk_bins']), mass_range)
            """
            Pgg_c = F_k1 * np.zeros((observables.gg.nbins,setup['lnk_bins']))
        else:
            Pgg_c = F_k1 * np.zeros((observables.gg.nbins,setup['lnk_bins']))

        if ingredients['satellites']:
            beta = s_beta
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

        if ingredients['haloexclusion'] and setup['return'] != 'power':
            Pgg_k_t = Pgg_c + (2.0 * Pgg_cs) + Pgg_s
            Pgg_k = Pgg_c + (2.0 * Pgg_cs) + Pgg_s + Pgg_2h
        else:
            Pgg_k = Pgg_c + (2.0 * Pgg_cs) + Pgg_s + Pgg_2h

    # Matter - matter spectra
    if observables.mm:
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

        #if ingredients['haloexclusion'] and setup['return'] != 'power':
        #    Pmm_k_t = Pmm_1h
        #    Pmm_k = Pmm_1h + Pmm_2h
        #else:
        Pmm_k = Pmm_1h + Pmm_2h

    # Outputs

    if observables.gm:
        # note this doesn't include the point mass! also, we probably
        # need to return k
        if setup['return'] == 'power':
            if integrate_zlens:
                Pgm_k = np.sum(z*Pgm_k, axis=1) / intnorm
            P_inter = [UnivariateSpline(
                            setup['k_range'], np.log(Pg_k_i), s=0, ext=0)
                       for Pg_k_i in Pgm_k]
        else:
            if integrate_zlens:
                if ingredients['haloexclusion'] and setup['return'] != 'power':
                    P_inter = [[UnivariateSpline(setup['k_range'], logPg_ij,
                                                 s=0, ext=0)
                                for logPg_ij in logPg_i]
                               for logPg_i in np.log(Pgm_k_t)]
                    P_inter_2h = [[UnivariateSpline(setup['k_range'],
                                                    logPg_ij, s=0, ext=0)
                                   for logPg_ij in logPg_i]
                                  for logPg_i in np.log(Pgm_2h)]
                else:
                    P_inter = [[UnivariateSpline(setup['k_range'], logPg_ij,
                                                 s=0, ext=0)
                                for logPg_ij in logPg_i]
                               for logPg_i in np.log(Pgm_k)]
            else:
                if ingredients['haloexclusion'] and setup['return'] != 'power':
                    P_inter = [UnivariateSpline(setup['k_range'], Pg_k_i,
                                                s=0, ext=0)
                               for Pg_k_i in np.log(Pgm_k_t)]
                    P_inter_2h = [UnivariateSpline(setup['k_range'],
                                                   Pg_2h_i, s=0, ext=0)
                                  for Pg_2h_i in np.log(Pgm_2h)]
                else:
                    P_inter = [UnivariateSpline(setup['k_range'],
                                                Pg_k_i, s=0, ext=0)
                               for Pg_k_i in np.log(Pgm_k)]

    if observables.gg:
        if ingredients['haloexclusion'] and setup['return'] != 'power':
            P_inter_2 = [UnivariateSpline(setup['k_range'], Pgg_k_i,
                                          s=0, ext=0)
                         for Pgg_k_i in np.log(Pgg_k_t)]
            P_inter_2_2h = [UnivariateSpline(setup['k_range'], Pgg_2h_i,
                                             s=0, ext=0)
                            for Pgg_2h_i in np.log(Pgg_2h)]
        else:
            P_inter_2 = [UnivariateSpline(setup['k_range'], Pgg_k_i, s=0,
                                          ext=0)
                         for Pgg_k_i in np.log(Pgg_k)]

    if observables.mm:
        #if ingredients['haloexclusion'] and setup['return'] != 'power':
        #    P_inter_3 = [UnivariateSpline(setup['k_range'], np.log(Pmm_k_i), s=0, ext=0)
        #            for Pmm_k_i in Pmm_k_t]
        #    P_inter_3_2h = [UnivariateSpline(setup['k_range'], np.log(Pmm_2h_i), s=0, ext=0)
        #            for Pmm_2h_i in Pmm_2h]
        #else:
        P_inter_3 = [UnivariateSpline(setup['k_range'], Pmm_k_i, s=0, ext=0)
                     for Pmm_k_i in np.log(Pmm_k)]

    if observables.gm:
        if setup['return'] == 'all':
            output[observables.gm.idx] = Pgm_k
        if setup['return'] == 'power':
            Pgm_out = [exp(P_i(np.log(r_i)))
                       for P_i, r_i in zip(P_inter, observables.gm.R)]
            output[observables.gm.idx] = Pgm_out
    if observables.gg:
        if setup['return'] == 'all':
            output[observables.gg.idx] = Pgg_k
        if setup['return'] == 'power':
            Pgg_out = [exp(P_i(np.log(r_i)))
                       for P_i, r_i in zip(P_inter_2, observables.gg.R)]
            output[observables.gg.idx] = Pgg_out
    if observables.mm:
        if setup['return'] == 'all':
            output[observables.mm.idx] = Pmm_k
        if setup['return'] == 'power':
            Pmm_out = [exp(P_i(np.log(r_i)))
                       for P_i, r_i in zip(P_inter_3, observables.mm.R)]
            output[observables.mm.idx] = Pmm_out
    if setup['return'] == 'power':
        output = list(output)
        output = [output, meff]
        return output
    elif setup['return'] == 'all':
        output.append(setup['k_range_lin'])
    else:
        pass

    # correlation functions
    if observables.gm:
        if integrate_zlens:
            if ingredients['haloexclusion']:
                xi2 = np.array(
                    [[power_to_corr_ogata(P_inter_ji, rvir_range_3d)
                      for P_inter_ji in P_inter_j] for P_inter_j in P_inter])
                xi2_2h = np.array(
                    [[power_to_corr_ogata(P_inter_ji, rvir_range_3d)
                      for P_inter_ji in P_inter_j] for P_inter_j in P_inter_2h])
                xi2 = xi2 + halo_exclusion(
                    xi2_2h, rvir_range_3d, meff[observables.gm.idx],
                    rho_bg[observables.gm.idx], setup['delta'])
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
                xi2 = xi2 + halo_exclusion(
                    xi2_2h, rvir_range_3d, meff[observables.gm.idx],
                    rho_bg[observables.gm.idx], setup['delta'])
            else:
                xi2 = np.array(
                    [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                     for P_inter_i in P_inter])
        # not yet allowed
        if setup['return'] == 'xi':
            if integrate_zlens:
                xi2 = np.sum(z*xi2, axis=1) / intnorm
            xi_out_i = array(
                [UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0)
                 for si in zip(xi2)])
            xi_out = np.array(
                [x_i(r_i) for x_i, r_i in zip(xi_out_i, observables.gm.R)])
            output[observables.gm.idx] = xi_out
        if setup['return'] == 'all':
            if integrate_zlens:
                xi2 = np.sum(z*xi2, axis=1) / intnorm
            output.append(xi2)

    if observables.gg:
        if ingredients['haloexclusion']:
            xi2_2 = np.array(
                [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                for P_inter_i in P_inter_2])
            xi2_2_2h = np.array(
                [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                for P_inter_i in P_inter_2_2h])
            xi2_2 = xi2_2 + halo_exclusion(
                xi2_2_2h, rvir_range_3d, meff[observables.gg.idx],
                rho_bg[observables.gg.idx], setup['delta'])
        else:
            xi2_2 = np.array(
                [power_to_corr_ogata(P_inter_i, rvir_range_3d)
                for P_inter_i in P_inter_2])
        if setup['return'] == 'xi':
            xi_out_i_2 = array(
                [UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0)
                 for si in zip(xi2_2)])
            xi_out_2 = np.array(
                [x_i(r_i) for x_i, r_i in zip(xi_out_i_2, observables.gg.R)])
            output[observables.gg.idx] = xi_out_2
        if setup['return'] == 'all':
            output.append(xi2_2)

    if observables.mm:
        #if ingredients['haloexclusion']:
        #    xi2_3 = np.array(
        #        [power_to_corr_ogata(P_inter_i, rvir_range_3d)
        #        for P_inter_i in P_inter_3])
        #    xi2_3_2h = np.array(
        #        [power_to_corr_ogata(P_inter_i, rvir_range_3d)
        #        for P_inter_i in P_inter_3_2h])
        #    xi2_3 = xi2_3 + halo_exclusion(
        #    xi2_3_2h, rvir_range_3d, meff[observables.mm.idx],
        #    rho_bg[observables.mm.idx], setup['delta'])
        #else:
        xi2_3 = np.array(
            [power_to_corr_ogata(P_inter_i, rvir_range_3d)
            for P_inter_i in P_inter_3])
        if setup['return'] == 'xi':
            xi_out_i_3 = np.array(
                [UnivariateSpline(rvir_range_3d, np.nan_to_num(si), s=0)
                 for si in zip(xi2_3)])
            xi_out_3 = np.array(
                [x_i(r_i) for x_i, r_i in zip(xi_out_i_3, observables.mm.R)])
            output[observables.mm.idx] = xi_out_3
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

    if observables.gm:
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
                [10**m_pm / r_i**2
                 for m_pm, r_i in zip(c_pm[0], observables.gm.R)])
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
        for i in range(observables.gm.nbins):
            surf_dens2[i] = fill_nan(surf_dens2[i])
        if setup['return'] in ('kappa', 'sigma'):
            surf_dens2_r = array(
                [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(si), s=0)
                for si in surf_dens2])
            surf_dens2 = np.array(
                [s_r(r_i)
                 for s_r, r_i in zip(surf_dens2_r, observables.gm.R)])
            #return [surf_dens2, meff]
            if observables.gm.nbins == 1:
                output[observables.gm.idx.start] = surf_dens2[0]
            else:
                output[observables.gm.idx] = surf_dens2
        if setup['return'] == 'all':
            output.append(surf_dens2)

    if observables.gg:
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
        for i in range(observables.gg.nbins):
            surf_dens2_2[i] = fill_nan(surf_dens2_2[i])
        if setup['return'] in ('kappa', 'sigma'):
            surf_dens2_2_r = array(
                [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(si), s=0)
                for si in surf_dens2_2])
            surf_dens2_2 = np.array(
                [s_r(r_i) for s_r, r_i
                 in zip(surf_dens2_2_r, observables.gg.R)])
        if setup['return'] in ('kappa', 'sigma'):
            if observables.gg.nbins == 1:
                output[observables.gg.idx.start] = surf_dens2_2[0]
            else:
                output[observables.gg.idx] = surf_dens2_2
        if setup['return'] in ('wp', 'esd_wp'):
            wp_out_i = np.array(
                [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(wi/rho_i),
                                  s=0)
                 for wi, rho_i in zip(surf_dens2_2, rho_bg)])
            wp_out = [wp_i(r_i) for wp_i, r_i
                      in zip(wp_out_i, observables.gg.R)]
            #output.append(wp_out)
        if setup['return'] == 'all':
            wp_out = surf_dens2_2/expand_dims(rho_bg, -1)
            output.append([surf_dens2_2, wp_out])


    if observables.mm:
        surf_dens2_3 = np.array(
            [sigma(xi2_i, rho_i, rvir_range_3d, rvir_range_3d_i)
             for xi2_i, rho_i in zip(xi2_3, rho_bg)])

        if setup['distances'] == 'proper':
            surf_dens2_3 *= (1+zo)**2

        # in Msun/pc^2
        if not setup['return'] == 'kappa':
            surf_dens2_3 /= 1e12

        # fill/interpolate nans
        surf_dens2_3[(surf_dens2_3 <= 0) | (surf_dens2_3 >= 1e20)] = np.nan
        for i in range(observables.mm.nbins):
            surf_dens2_3[i] = fill_nan(surf_dens2_3[i])
        if setup['return'] in ('kappa', 'sigma'):
            surf_dens2_3_r = array(
                [UnivariateSpline(rvir_range_3d_i, np.nan_to_num(si), s=0)
                for si in surf_dens2_3])
            surf_dens2_3 = array(
                [s_r(r_i) for s_r, r_i
                 in zip(surf_dens2_3_r, observables.mm.R)])
            #return [surf_dens2_3, meff]
            if observables.mm.nbins == 1:
                output[observables.mm.idx.start] = surf_dens2_3[0]
            else:
                output[observables.mm.idx] = surf_dens2_3
        if setup['return'] == 'all':
            output.append(surf_dens2_3)

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

    # excess surface density
    """
    # These two options are not really used for any observable! Keeping in for now.

    if ingredients['mm']:
        d_surf_dens2_3 = array(
            [np.nan_to_num(
            d_sigma(surf_dens2_i, rvir_range_3d_i, r_i))
            for surf_dens2_i, r_i in zip(surf_dens2_3, observables.mm.R)])

        out_esd_tot_3 = array(
            [UnivariateSpline(r_i, np.nan_to_num(d_surf_dens2_i), s=0)
            for d_surf_dens2_i, r_i in zip(d_surf_dens2_3, r_i)])

        #out_esd_tot_inter_3 = np.zeros((nbins, observables.mm.R[0].size))
        #for i in range(nbins):
        #    out_esd_tot_inter_3[i] = out_esd_tot_3[i](observables.mm.R[i])
        out_esd_tot_inter_3 = [out_esd_tot_3[i](observables.mm.R[i])
                               for i in range(observables.mm.nbins)]
        # This insert makes sure that the ESD's are on the fist place.
        output.insert(0, out_esd_tot_inter_3)


    if ingredients['gg']:
        d_surf_dens2_2 = array(
                [np.nan_to_num(
                d_sigma(surf_dens2_i, rvir_range_3d_i, r_i))
                for surf_dens2_i, r_i in zip(surf_dens2_2, observables.gg.R)])

        out_esd_tot_2 = array(
            [UnivariateSpline(r_i, np.nan_to_num(d_surf_dens2_i), s=0)
             for d_surf_dens2_i, r_i in zip(d_surf_dens2_2, rvir_range_2d_i)])

        #out_esd_tot_inter_2 = np.zeros((nbins, observables.gg.R.size))
        #for i in range(nbins):
        #    out_esd_tot_inter_2[i] = out_esd_tot_2[i](observables.gg.R[i])
        out_esd_tot_inter_2 = [out_esd_tot_2[i](observables.gg.R[i])
                               for i in range(observables.gg.nbins)]
        output.insert(0, out_esd_tot_inter_2)
    """

    if observables.gm:
        d_surf_dens2 = array(
            [np.nan_to_num(
                d_sigma(surf_dens2_i, rvir_range_3d_i, r_i))
            for surf_dens2_i, r_i in zip(surf_dens2, observables.gm.R)])

        out_esd_tot = array(
            [UnivariateSpline(r_i, np.nan_to_num(d_surf_dens2_i), s=0)
            for d_surf_dens2_i, r_i in zip(d_surf_dens2, observables.gm.R)])

        #out_esd_tot_inter = np.zeros((nbins, rvir_range_2d_i.size))
        #for i in range(nbins):
        #    out_esd_tot_inter[i] = out_esd_tot[i](rvir_range_2d_i)
        out_esd_tot_inter = [out_esd_tot[i](observables.gm.R[i])
                             for i in range(observables.gm.nbins)]
        # this should be moved to the power spectrum calculation
        if ingredients['pointmass']:
            # the 1e12 here is to convert Mpc^{-2} to pc^{-2} in the ESD
            pointmass = c_pm[1]/(np.pi*1e12) * array(
                [10**m_pm / (r_i**2)
                 for m_pm, r_i in zip(c_pm[0], observables.gm.R)])
            out_esd_tot_inter = [out_esd_tot_inter[i] + pointmass[i]
                                 for i in range(observables.gm.nbins)]
        if setup['return'] == 'esd_wp':
            output[observables.gm.idx] = out_esd_tot_inter
            output[observables.gg.idx] = wp_out
            output = list(output)
            output = [output, meff]
        else:
            output = [out_esd_tot_inter, meff]

    return output


## auxiliary functions

def calculate_completeness(z, observables, selection, ingredients):
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


def load_cosmology(cosmo):
    sigma8, h, omegam, omegab, n_s, w0, wa, Neff, z = cosmo[:9]
    cosmo_model = Flatw0waCDM(
        H0=100*h, Ob0=omegab, Om0=omegam, Tcmb0=2.725, m_nu=0.06*eV,
        Neff=Neff, w0=w0, wa=wa)
    return cosmo_model, sigma8, n_s, z


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

if __name__ == '__main__':
    print(0)
