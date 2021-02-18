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
from astropy import units as u
from functools import wraps
from glob import glob
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

# specific to cmbhalo
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo.concentration import concentration
from profiley.nfw import NFW
from profiley.filtering import Filter
from profiley.helpers import lss
import pyccl as ccl

"""
from . import baryons, halo, longdouble_utils as ld, nfw
from .tools import (
    fill_nan, load_hmf, virial_mass, virial_radius)
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, sigma_crit,
    power_to_corr_ogata, wp, wp_beta_correction, power_to_sigma, power_to_sigma_ogata)
from .dark_matter import (
    mm_analy, gm_cen_analy, gm_sat_analy, gg_cen_analy,
    gg_sat_analy, gg_cen_sat_analy, two_halo_gm, two_halo_gg, halo_exclusion, beta_nl)
from .covariance import covariance
"""
from . import halo
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

    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]

    #assert setup['delta_ref'] in ('critical', 'matter')

    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_miscent, c_twohalo, \
        s_concentration, s_mor, s_scatter, s_beta = theta

    #cosmo_model, sigma8, n_s, z = halo.load_cosmology(cosmo)
    sigma8, h, omegam, omegab, n_s, w0, wa, Neff, z = cosmo[:9]
    zs = cosmo[-1]
    # astropy (for profiley)
    cosmo_model, sigma8, n_s, z = halo.load_cosmology(cosmo)
    """
    # colossus (for mass-concentration relation)
    params = dict(Om0=omegam, H0=100*h, ns=n_s, sigma8=sigma8, Ob0=omegab)
    colossus_cosmology.setCosmology('cosmo', params)
    # CCL (for halo mass function, halo bias and matter power spectrum)
    # note that Tcmb and m_nu are hard-coded both in load_cosmology and here
    # this needs to change
    cclcosmo = ccl.Cosmology(
        Omega_c=omegam-omegab, Omega_b=omegab, h=h, A_s=2.1e-9, n_s=n_s,
        T_CMB=2.725, Neff=Neff, m_nu=0.06, w0=w0, wa=wa)
    """
    cclcosmo = define_cosmology(cosmo)
    mdef = ccl.halos.MassDef(setup['delta'], setup['delta_ref'])

    # probably need to be careful with the normalization of the halo bias
    # function in CCL? See KiDS-GGL issue #184

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

    z = halo.format_z(z, nbins)
    print('z =', z.shape)

    # concentration
    profiles = define_profiles(setup, cosmo_model, z, c_concentration)
    print('shape =', profiles.shape)

    if profiles.frame == 'physical':
        arcmin2kpc = cosmo_model.kpc_proper_per_arcmin
    else:
        arcmin2kpc = cosmo_model.kpc_comoving_per_arcmin
    Rfine = (setup['arcmin_fine'] * arcmin2kpc(z)).to(u.Mpc).value.T
    print('Rfine =', Rfine.shape)

    output = {}
    output['kappa.1h.mz.raw'] = np.transpose(
        profiles.convergence(Rfine[:,:,None]), axes=(1,2,0))

    print_output(output)

    output['kappa.1h.mz'] = filter_profiles(
        ingredients, setup['kfilter'], output['kappa.1h.mz.raw'],
        setup['arcmin_fine'], setup['arcmin_bins'])

    print_output(output)

    ti = time()
    output['sigma.2h.raw'] = calculate_sigma_2h(setup, cclcosmo, mdef, z)
    tf = time()
    print(f'calculate_sigma_2h in {tf-ti:.1f} s')

    output['kappa.2h.mz.raw'] = output['sigma.2h.raw'] \
        / profiles.sigma_crit()[:,:,None]

    print_output(output)

    output['kappa.2h.mz'] = filter_profiles(
        ingredients, setup['kfilter'], output['kappa.2h.mz.raw'],
        setup['arcmin_fine'], setup['arcmin_bins'])

    print_output(output)

    # halo mass function weighting
    hmf = ccl.halos.mass_function_from_name('Tinker10')
    hmf = hmf(cclcosmo, mass_def=mdef)
    dndm = np.array([hmf.get_mass_function(cclcosmo, m, ai) for ai in a])
    print('dndm =', dndm.shape)
    output['kappa.1h'] = trapz(
        expand_dims(dndm, -1)*output['kappa.1h.mz'], dndm, axis=1)

    print_output(output)

    return [list(output), np.ones(output.shape[0])]


##################################
##
##  Auxiliary functions
##
##################################

def print_output(output):
    print()
    for key, val in output.items():
        print('   ', key, val.shape)
    print()


def preamble(theta):
    """Preamble to cmbhalo.model

    This function should include e.g., all data type assertions and
    general modifications of parameters from the config file. However
    I still need to figure out the best way to do this latter part
    """
    np.seterr(
    	divide='ignore', over='ignore', under='ignore', invalid='ignore')

    observables, selection, ingredients, params, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]

    assert setup['distances'] == 'comoving'

    #cclcosmo = define_cosmology(params[0])
    setup['bias_func'] = ccl.halos.halo_bias_from_name('Tinker10')

    # read measurement binning scheme
    # need to do it in a smarter (more flexible) way
    dataset_name = 'des_r2'
    filenames = sorted(glob(os.path.join(
        dataset_name, f'{dataset_name}_l*', '*_bin_edges.txt')))
    arcmin_bins = np.array([np.loadtxt(f) for f in filenames])
    setup['arcmin_bins'] = arcmin_bins
    setup['arcmin'] = (arcmin_bins[:,:-1]+arcmin_bins[:,1:]) / 2

    # fine binning for filtering
    setup['arcmin_fine_bins'] = np.linspace(0, 50, 1001) * u.arcmin
    setup['arcmin_fine'] = 0.5 \
        * (setup['arcmin_fine_bins'][:-1]+setup['arcmin_fine_bins'][1:])

    # we might also want to add a function in the configuration
    # functionality to update theta more easily
    theta[1][theta[0].index('setup')] = setup

    return theta


def calculate_sigma_2h(setup, cclcosmo, mdef, z, threads=1):
    a = 1 / (1+z[:,0])
    m = setup['mass_range']
    # matter power spectra
    k = setup['k_range_lin']
    Pk = np.array([ccl.linear_matter_power(cclcosmo, k, ai) for ai in a])
    # halo bias
    bias = setup['bias_func'](cclcosmo, mass_def=mdef)
    bh = np.array([bias.get_halo_bias(cclcosmo, m, ai) for ai in a])

    # correlation function
    #print('correlation functions...')
    Rxi = np.logspace(-2, 2, 100)
    r = setup['rvir_range_3d']
    ti = time()
    lnk = setup['k_range']
    xi_kz = np.array([lss.power2xi(interp1d(lnk, lnPk_i), Rxi)
                      for lnPk_i in np.log(Pk)])
    xi = bh[:,:,None] * xi_kz[:,None]
    #print(f'in {time()-ti:.2f} s')

    #print('distances...')
    #ti = time()
    Dc = ccl.background.comoving_radial_distance(cclcosmo, a)
    #print(f'in {time()-ti:.2f} s')
    bg = 'critical' if 'critical' in setup['delta_ref'].lower() else 'matter'
    #print('rho_m...')
    #ti = time()
    rho_m = ccl.background.rho_x(cclcosmo, 1, bg)
    #print(f'in {time()-ti:.2f} s')
    # note that we assume all arcmin (i.e., for all bins) are the same
    arcmin = setup['arcmin_fine'][0] if len(setup['arcmin_fine'].shape) == 2 \
        else setup['arcmin_fine']
    R = arcmin * Dc[:,None] * (np.pi/180/60)

    # testing one xi2sigma
    #print('xi =', R.shape, Rxi.shape, xi.shape, rho_m)
    #ti = time()
    #sigma_test = lss.xi2sigma(R, Rxi, 

    # finally, surface density
    #print('surface densities...')
    #ti = time()
    sigma_2h = lss.xi2sigma(
        R, Rxi, xi, rho_m, threads=1, full_output=False)
    #print(f'in {time()-ti:.2f} s')
    return sigma_2h

    # only do this when saving in the preamble?
    print('shape =', sigma.shape)
    cosmo_params = {key: val for key, val in params.items()
                    if key in ('h','Omega_m','Omega_c','Omega_b','As','ns')}
    lss.save_profiles(
        'sigma_2h.txt', z[:,0], np.log10(m), theta, sigma, xlabel='z', ylabel='m',
        label='sigma_2h', R_units='arcmin', cosmo_params=cosmo_params)
    return sigma_2h


def define_cosmology(cosmo_params, Tcmb=2.725, m_nu=0, backend='ccl'):
    sigma8, h, omegam, omegab, n_s, w0, wa, Neff, z = cosmo_params[:9]
    if backend == 'ccl':
        cosmo = ccl.Cosmology(
            Omega_c=omegam-omegab, Omega_b=omegab, h=h, A_s=2.1e-9, n_s=n_s,
            T_CMB=Tcmb, Neff=Neff, m_nu=m_nu, w0=w0, wa=wa)
    elif backend == 'astropy':
        cosmo = halo.load_cosmology(cosmo_params)[0]
    # colossus (for mass-concentration relation)
    params = dict(Om0=omegam, H0=100*h, ns=n_s, sigma8=sigma8, Ob0=omegab)
    colossus_cosmology.setCosmology('cosmo', params)
    return cosmo


def define_profiles(setup, cosmo, z, c_concentration):
    model, fc = c_concentration
    ref = setup['delta_ref']
    ref = ref[0] if ref in ('critical', 'matter') \
        else ref[2].lower()
    bg = f"{int(setup['delta'])}{ref}"
    c = fc * np.array([concentration(setup['mass_range'], bg, zi, model=model)
                       for zi in z[:,0]])
    profiles = NFW(
        setup['mass_range'], c, z, cosmo=cosmo, overdensity=setup['delta'],
        background=ref, frame='comoving', z_s=1100)
    return profiles


def filter_profiles(ingredients, filter, profiles, theta_fine, theta_bins,
                    units='arcmin'):
    if ingredients['nzlens']:
        # fix later
        filtered = [np.array(
                        [filter.filter(theta_fine, p_ij, theta_bins, units=units)
                         for p_ij in p_i])
                    for p_i in profiles]
        filtered = np.transpose(filtered, axes=(2,0,1))
    else:
        filtered = [np.array(
                        [filter.filter(theta_fine, prof_mz, theta_bins_z,
                                       units=units)[1]
                         for prof_mz in prof_z])
                    for prof_z, theta_bins_z in zip(profiles, theta_bins)]
        print('filtered =', np.array(filtered).shape)
        #filtered = np.transpose(filtered, axes=(2,0,1))
        filtered = np.array(filtered)
    return filtered
