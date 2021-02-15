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
    # colossus (for mass-concentration relation)
    params = dict(Om0=omegam, H0=100*h, ns=n_s, sigma8=sigma8, Ob0=omegab)
    colossus_cosmology.setCosmology('cosmo', params)
    # CCL (for halo mass function, halo bias and matter power spectrum)
    # note that Tcmb and m_nu are hard-coded both in load_cosmology and here
    # this needs to change
    cclcosmo = ccl.Cosmology(
        Omega_c=omegam-omegab, Omega_b=omegab, h=h, A_s=2.1e-9, n_s=n_s,
        T_CMB=2.725, Neff=Neff, m_nu=0.06, w0=w0, wa=wa)
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
    # move to preamble later
    arcmin_fine_bins = np.linspace(0, 50, 1001) * u.arcmin
    arcmin_fine = (arcmin_fine_bins[:-1]+arcmin_fine_bins[1:]) / 2
    if profiles.frame == 'physical':
        arcmin2kpc = cosmo_model.kpc_proper_per_arcmin
    else:
        arcmin2kpc = cosmo_model.kpc_comoving_per_arcmin
    Rfine = (arcmin_fine * arcmin2kpc(z)).to(u.Mpc).value.T
    print('Rfine =', Rfine.shape)

    output = {}
    output['kappa.1h.mz.raw'] = np.transpose(
        profiles.convergence(Rfine[:,:,None], zs), axes=(1,2,0))

    print_output(output)

    output['kappa.1h.mz'] = filter_profiles(
        ingredients, setup['kfilter'], output['kappa.1h.mz.raw'], arcmin_fine,
        setup['arcmin_bins'])

    print_output(output)

    #output['sigma.2h.raw'] = lss.load_profiles('sigma_2h.txt')[0]

    print_output(output)

    return


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

    # read measurement binning scheme
    # need to do it in a smarter (more flexible) way
    dataset_name = 'des_r2'
    filenames = sorted(glob(os.path.join(
        dataset_name, f'{dataset_name}_l*', '*_bin_edges.txt')))
    arcmin_bins = np.array([np.loadtxt(f) for f in filenames])
    setup['arcmin_bins'] = arcmin_bins

    # we might also want to add a function in the configuration
    # functionality to update theta more easily
    theta[1][theta[0].index('setup')] = setup

    return theta


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
        background=ref, frame=setup['distances'])
    return profiles
