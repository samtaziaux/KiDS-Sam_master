#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cluster lensing halo model

A halo model tailored for galaxy cluster measurements - that is, when
the lens sample is composed only of central galaxies, with a satellite
fraction set to exactly zero. This assumptions allows us to simplify
much of the halo model implementation, and therefore warrants its own
module.

For easy exchange with ``halo.model``, the satellite parameters are
retained in the configuration file, but they are ignored completely in
the model. Be sure to fix all of them in the configuration or they
will appear in the output file as free parameters!

NOTES
-----
* Only 'gm' observables allowed for now
"""
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
from .lens import d_sigma
from .miscentring import miscentring
from .. import hod
from ..helpers import io

# used only in testing phase
from plott.plotutils import colorscale, savefig, update_rcParams
update_rcParams()


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


debug = True

def model(theta, R):
    """CMB halo modeling

    Note that R here is in arcmin
    """

    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]

    retvalues = setup['return'] + setup['return_extra']
    # alias
    Mh = setup['mass_range']

    #assert setup['delta_ref'] in ('critical', 'matter')

    # satellite parameters in the config file are ignored;
    # they may or may not be present
    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_mis, c_twohalo = theta[:7]
        #s_concentration, s_mor, s_scatter, s_beta = theta

    zs = cosmo[-1]
    # astropy (for profiley)
    cosmo_model, sigma8, n_s, z = halo.load_cosmology(cosmo)
    # CCL (for 2-halo things)
    cclcosmo = define_cosmology(cosmo)
    mdef = ccl.halos.MassDef(setup['delta'], setup['delta_ref'])

    # probably need to be careful with the normalization of the halo bias
    # function in CCL? See KiDS-GGL issue #184

    ### a final bit of massaging ###

    # leaving this for later
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
    if debug:
        print('z =', z.shape)

    profiles = define_profiles(setup, cosmo_model, z, c_concentration)
    #if debug:
        #print('shape =', profiles.shape)

    # convert radii if necessary - this must be done here because it
    # depends on cosmology
    if setup['R_unit'] == 'arcmin':
        if setup['distances'] in ('physical', 'proper'):
            arcmin2kpc = cosmo_model.kpc_proper_per_arcmin
        else:
            arcmin2kpc = cosmo_model.kpc_comoving_per_arcmin
        arcmin2Mpc = arcmin2kpc(z).to(u.Mpc/u.arcmin).value
        if debug:
            print('R =', R, 'arcmin')
        # remember that the first element is a zero we added (and should remove)
        # for nfw_stack
        R = np.array(R[:,1:] * arcmin2Mpc, dtype=float)
        if debug:
            print('R =', R, 'Mpc', R.shape, R.dtype)
        if setup['kfilter']:
            setup['Rfine'] \
                = (setup['bin_centers_fine']*arcmin2Mpc).T
            if debug:
                # only for the prints below
                af = setup['bin_centers_fine']
                rf = setup['Rfine']
                print('bin_centers_fine =', af[0], af[-1], af.shape)
                print('Rfine =', rf[0], rf[-1], rf.shape)
        else:
            setup['Rfine'] = R.T
    elif setup['kfilter']:
        setup['Rfine'] = setup['bin_centers_fine']
    else:
        setup['Rfine'] = R.T

    output = {}

    ### halo mass function ###

    hmf = ccl.halos.mass_function_from_name('Tinker10')
    hmf = hmf(cclcosmo, mass_def=mdef)
    dndm = np.array(
        [hmf.get_mass_function(cclcosmo, Mh, ai)
         for ai in 1/(1+z[:,0])])
    if debug:
        print('dndm =', dndm.shape)

    ### selection function and mass-observable relation ###

    output['pop_c'], output['pop_s'] = halo.populations(
        observables, ingredients, selection, Mh, z, theta)
    output['pop_g'] = output['pop_c'] + output['pop_s']
    # note that pop_g already accounts for incompleteness
    ngal, logMh_eff = halo.calculate_ngal(
        observables, output['pop_g'], dndm, Mh)
    info(output, 'pop_g')

    mass_norm = trapz(dndm*output['pop_g'], Mh, axis=1)
    if debug:
        print('mass_norm =', mass_norm.shape)

    # radii for initial calculation
    Rx = setup['Rfine'] if setup['kfilter'] else R.T

    ### 1-halo term ###

    # for now only includes sigma, kappa, esd (gg_1h=0)
    qfuncs = ([p.projected for p in profiles],
              [p.convergence for p in profiles],
              [p.projected_excess for p in profiles])
    for profile_name, funcs in zip(('sigma', 'kappa', 'esd'), qfuncs):
        if profile_name in retvalues:
            key = f'{profile_name}.1h'
            print(f'\n *** {key} ***')
            # note that this will probably not work if the function is
            # not analytical
            print('shape =', profiles[0].shape)
            print(profiles[0])
            output[key] = np.array(
                [func(Rx[:,i,None]) for i, func in enumerate(funcs)])
            print(f'{key} =', output[key].shape)
            #plot_profile_mz(Rx.T, np.transpose(prof, axes=(1,2,0)),
                            #z, mx=Mh, z0=0.46, m0=1e14,
                            #output=f'profiles_{profile_name}1h.png')

            # miscentering (1h only for now)
            # c_mis[1] corresponds to f_mis
            if ingredients['miscentring']:
                #prof_mis = miscentering(
                    #profiles, Rx, *c_mis[2:], dist=c_mis[0])
                offkey = f'{key}.off'
                p_mis = miscentring(c_mis[0], *c_mis[2:])
                print('p_mis =', p_mis.shape)
                print('Rx =', Rx.shape)
                print('c_mis[2] =', c_mis[2])
                print('profiles[0].shape =', profiles[0].shape)
                ti = time()
                output[offkey] = np.array(
                    [p.offset(f, Rx_i, c_mis[2], weights=p_mis_i)
                     for p, f, Rx_i, p_mis_i
                     in zip(profiles, funcs, Rx.T, p_mis)])
                #output[offkey] = profiles.offset(func, Rx.T
                print(f'offset in {time()-ti:.2f} s')
                print(f'{offkey}: {output[offkey].shape}')
                # rename well-centered profile
                output[f'{key}.cent'] = output.pop(key)
                # total 1h profile
                output[key] = (1-c_mis[1])*output[f'{key}.off'] \
                    + c_mis[1]*output[f'{key}.cent']

            if setup['kfilter'] and (f'{key}.cent' in retvalues
                    or f'{key}.off' in retvalues):
                output[offkey] = np.transpose(
                    output[offkey], axes=(0,2,1))
                output[offkey] = filter_profiles(
                    ingredients, setup['kfilter'], output[offkey],
                    setup['bin_centers_fine'], setup['bin_edges'])

            # dndm weighting
            output[key] = trapz(
                (dndm*output['pop_g'])[:,None] * output[key], Mh, axis=2) \
                / mass_norm[:,None]
            print(f'weighted {key}: {output[key].shape}')

    #ti = time()
    if 'kappa.1h.mz' in output:
        output['kappa.1h'] = trapz(
            dndm * output['pop_g'] * output['kappa.1h.mz'], Mh, axis=2) \
            / mass_norm

        output['kappa'] = output['kappa.1h']

    if ingredients['twohalo']:
        output = calculate_twohalo(
            output, setup, ingredients, cclcosmo, mdef, dndm, z, profiles,
            mass_norm)

        if 'kappa' in retvalues or 'kappa.2h' in retvalues:
            #print(R.shape, output['kappa.1h.mz'].shape, output['kappa.2h.mz'].shape)
            plot_profile_mz(R, np.transpose(output['kappa.2h.mz'], axes=(1,2,0)),
                            z, yscale='linear',
                            mx=Mh, z0=0.46, xscale='linear',
                            output='profiles_kappa2h_filtered.png')
            plot_profile_mz(R, np.transpose(output['kappa.2h.mz']+output['kappa.1h.mz'], axes=(1,2,0)),
                            z, yscale='linear',
                            mx=Mh, z0=0.46, xscale='linear',
                            output='profiles_kappa1h2h_filtered.png')

            output['kappa'] = output['kappa'] + output['kappa.2h']
        #output['kappa.quadpy'] = \
            #output['kappa.1h'] + output['kappa.2h.quadpy']

        print_output(output)

        if setup['kfilter']:
            output['sigma'] = output['sigma.1h.mz.raw'] + output['sigma.2h.raw']
            plot_profile_mz(setup['Rfine'].T, output['sigma'], z[:,0],
                            mx=Mh, z0=0.46, m0=1e14,
                            output='profiles_sigma_1h2h.png')
            output['kappa.mz.raw'] = output['kappa.1h.mz.raw'] \
                + output['kappa.2h.mz.raw']
            plot_profile_mz(setup['Rfine'].T, output['kappa.mz.raw'], z[:,0],
                            mx=Mh, z0=0.46, m0=1e14,
                            output='profiles_kappa_1h2h.png')
    #plot_profile_mz(Rxi, out['xi'], z[:,0], mx=m, z0=0.46, m0=1e14,
                    #output='profiles_xi.png')

    # for now
    output['kappa'] = output['kappa'].T
    info(output, 'kappa')

    return [list(output['kappa']), logMh_eff]


##################################
##
##  Auxiliary functions
##
##################################

def info(output, key):
    if not debug:
        return
    print()
    print(key)
    if key not in output:
        print(f'key {key} not in output: {list(output.keys())}')
        return
    if output[key].shape[-1] > 100:
        print('Last 100 elements:')
        print(output[key][-1][-100:])
    else:
        print(output[key][-1])
    print(output[key].shape)
    print()

def print_output(output):
    if not debug:
        return
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

    for obs in observables:
        assert obs.obstype == 'gm', \
            "only 'gm' observables allowed in ``cluster.model``. Please" \
            " raise an issue on github if you would like to see other" \
            " observable types implemented"

    if setup['kfilter'] and not setup['R_unit'] == 'arcmin':
        msg = "R_unit must be set to 'arcmin' if applying a k-space filter"
        raise ValueError(msg)
    assert setup['distances'] in ('comoving', 'physical', 'proper')

    # for now
    setup['return'] = [setup['return']]

    #cclcosmo = define_cosmology(params[0])
    setup['bias_func'] = ccl.halos.halo_bias_from_name('Tinker10')

    # read measurement binning scheme if filtering
    if setup['kfilter']:
        if debug:
            print('bin_edges before exclude =', setup['bin_edges'])
        # remove excluded points from bin edges
        bin_edges = []
        for file in setup['bin_edges'][0]:
            bin_edges.append(np.loadtxt(file, usecols=[0]))
            be = [[bin_edges[-1][i] for i in range(len(bin_edges[-1])-1)
                   if i not in setup['exclude']][0]]
            bin_edges[-1] = be \
                + [bin_edges[-1][i+1] for i in range(len(bin_edges[-1])-1)
                   if i not in setup['exclude']]
        bin_edges = [np.array(b, dtype=float) for b in bin_edges]
        if debug:
            print('bin_edges =', bin_edges)
        setup['bin_edges'] = bin_edges

        # fine binning for filtering
        setup['bin_edges_fine'] \
            = np.linspace(0, setup['bin_sampling_max'], setup['bin_samples'])
        #if setup['R_unit'] == 'arcmin':
            #setup['bin_edges_fine'] *= u.arcmin
        setup['bin_centers_fine'] = 0.5 \
            * (setup['bin_edges_fine'][:-1]+setup['bin_edges_fine'][1:])

    # we might also want to add a function in the configuration
    # functionality to update theta more easily
    theta[1][theta[0].index('setup')] = setup

    return theta


def calculate_sigma_2h(setup, cclcosmo, mdef, z, threads=1):
    # for tests
    out = {}
    a = 1 / (1+z[:,0])
    m = setup['mass_range']
    # matter power spectra
    k = setup['k_range_lin']
    Pk = np.array([ccl.linear_matter_power(cclcosmo, k, ai) for ai in a])
    # halo bias
    bias = setup['bias_func'](cclcosmo, mass_def=mdef)
    bh = np.array([bias.get_halo_bias(cclcosmo, m, ai) for ai in a])
    Pgm = bh[:,:,None] * Pk[:,None]
    out['Pk'] = Pk
    info(out, 'Pk')
    out['Pgm'] = Pgm
    info(out, 'Pgm')

    # correlation function
    if debug:
        print('correlation functions...')
    #Rxi = np.logspace(-4, 3, 200)
    # I have no idea why, but for quadpy to work in xi2sigma the upper
    # bound on this *must* be 2
    Rxi = np.logspace(-3, 6, 2000)
    #Rxi = np.logspace(-3, 4, 120)
    #Rxi = setup['rvir_range_3d']
    if debug:
        print('Rxi =', Rxi[0], Rxi[-1], Rxi.shape)
        print('xi_kz...')
    lnk = setup['k_range']
    ti = time()
    xi_z = np.array([lss.power2xi(interp1d(lnk, lnPk_i), Rxi)
                     for lnPk_i in np.log(Pk)])
    if debug:
        print(f'in {time()-ti:.2f} s')
    out['xi_z'] = xi_z
    info(out, 'xi_z')
    out['xi'] = bh[:,:,None] * xi_z[:,None]
    info(out, 'xi')

    plot_profile_mz(Rxi, out['xi'], z[:,0], mx=m, z0=0.46, m0=1e14,
                    output='profiles_xi.png')

    bg = 'critical' if 'critical' in setup['delta_ref'].lower() else 'matter'
    if debug:
        print('rho_m...')
        ti = time()
    # in Msum/Mpc^3
    rho_m = ccl.background.rho_x(cclcosmo, 1, bg, is_comoving=True)
    if debug:
        print(f'in {time()-ti:.2f} s')
        print('rho_m =', rho_m)

    #print('Rfine =', setup['Rfine'], setup['Rfine'].shape)
    # finally, surface density
    # quad_vec
    """
    if debug:
        print('surface densities...')
        ti = time()
    sigma_2h_z1 = lss.xi2sigma(
        setup['Rfine'].T, Rxi, xi_z, rho_m, threads=1, full_output=False,
        integrator='scipy-vec', run_test=False)
    if debug:
        print(f'in {time()-ti:.2f} s')
        print('sigma_2h_z1 =', sigma_2h_z1.shape)
    plot_profile_mz(setup['Rfine'].T, bh[:,:,None]*sigma_2h_z1[:,None], z[:,0],
                    mx=m,
                    z0=0.46, m0=1e14, output='profiles_sigma2hz_quadvec.png')
    """
    """
    if debug:
        print('surface densities...')
        ti = time()
    # quadpy
    sigma_2h_z2 = lss.xi2sigma(
        setup['Rfine'].T, Rxi, xi_z, rho_m, threads=1, full_output=False,
        integrator='quadpy', run_test=False)
    if debug:
        print(f'in {time()-ti:.2f} s')
        print('sigma_2h_z2 =', sigma_2h_z2.shape)
    plot_profile_mz(setup['Rfine'].T, bh[:,:,None]*sigma_2h_z2[:,None], z[:,0],
                    mx=m,
                    z0=0.46, m0=1e14, output='profiles_sigma2hz_quadpy.png')
    """
    if debug:
        print('surface densities...')
        ti = time()
    #R2h = setup['Rfine'].T if setup['kfilter'] else setup['
    sigma_2h_z = lss.xi2sigma(
        setup['Rfine'].T, Rxi, xi_z, rho_m, threads=1, full_output=False,
        integrator='scipy', run_test=False, verbose=2*debug)
    plot_profile_mz(setup['Rfine'].T, bh[:,:,None]*sigma_2h_z[:,None], z[:,0],
                    mx=m,
                    z0=0.46, m0=1e14, output='profiles_sigma2hz_quad.png')
    if debug:
        print(f'in {time()-ti:.2f} s')
        print('sigma_2h_z =', sigma_2h_z.shape)
        """
        ti = time()
        sigma_2h = lss.xi2sigma(
            #setup['Rfine'].T, Rxi, xi_z, rho_m, threads=1, full_output=False,
            setup['Rfine'].T, Rxi, xi, rho_m, threads=1, full_output=False,
            integrator='scipy-vec', run_test=False)
        print(f'in {time()-ti:.2f} s')
        sigma_2h_1 = bh[:,:,None] * sigma_2h_z[:,None]
        s_ = np.s_[:,:10]
        y = np.allclose(sigma_2h[s_], sigma_2h_1[s_])
        print(f'are both sigma_2h the same? {y}')
        """

    return bh[:,:,None] * sigma_2h_z[:,None]


def calculate_twohalo(output, setup, ingredients, cclcosmo, mdef, dndm, z,
                      profiles, mass_norm):
    #ti = time()
    output['sigma.2h.raw'] = calculate_sigma_2h(setup, cclcosmo, mdef, z)
    #tf = time()
    #print(f'calculate_sigma_2h in {tf-ti:.1f} s')

    info(output, 'sigma.2h.raw')
    plot_profile_mz(setup['Rfine'].T, output['sigma.2h.raw'], z,
                    mx=setup['mass_range'], z0=0.46,
                    output='profiles_sigma.png')

    #######################
    ## could it be the units are not consistent here?? I think we're good
    #######################

    if 'esd' in retvalues or 'esd.2h' in retvalues:
        output['esd.2h.mz.raw'] = halo.calculate_esd(
            setup, observables, ingredients, output['sigma.2h.raw'], c_pm)
        info(output, 'esd.2h.mz.raw')
        plot_profile_mz(R, output['esd.2h.mz.raw'], z,
                        mx=setup['mass_range'], z0=0.46,
                        output='profiles_esd2h.png')

    if 'kappa' in retvalues or 'kappa.2h' in retvalues:
        output['kappa.2h.mz.raw'] = output['sigma.2h.raw'] \
            / profiles.sigma_crit()[:,:,None]

        info(output, 'kappa.2h.mz.raw')
        plot_profile_mz(setup['Rfine'].T, output['kappa.2h.mz.raw'], z,
                        mx=setup['mass_range'], z0=0.46,
                        output='profiles_kappa.png')

    # it seems now that kappa.2h.mz.raw is OK but something happens when
    # we apply the filter?

    #print_output(output)

    if setup['kfilter']:
        output['kappa.2h.mz'] = filter_profiles(
            ingredients, setup['kfilter'], output['kappa.2h.mz.raw'],
            setup['bin_centers_fine'], setup['bin_edges'])
        #output['kappa.2h.mz.quadpy'] = filter_profiles(
            #ingredients, setup['kfilter'], output['kappa.2h.mz.raw.quadpy'],
            #setup['bin_centers_fine'], setup['arcmin_bins'])
    else:
        output['kappa.2h.mz'] = np.transpose(
            output.pop('kappa.2h.mz.raw'), axes=(2,0,1))

    info(output, 'kappa.2h.mz')
    #R = (setup['bin_edges'][:,1:]+setup['bin_edges'][:,:-1])/2
    #print(R.shape)
    #plot_profile_mz(R.T,
                    #output['kappa.2h.mz'], z[:,0], mx=setup['mass_range'],
                    #z0=0.46, output='profiles_kappa2h_filtered.png')

    #print_output(output)

    #print('mass integrals...')
    output['kappa.2h'] = trapz(
        dndm * output['kappa.2h.mz'], setup['mass_range'], axis=2) \
        / mass_norm
    #output['kappa.2h'] = output['kappa.2h'].T
    #print(f'in {time()-ti:.2f} s')

    info(output, 'kappa.1h')
    info(output, 'kappa.2h')

    #print_output(output)

    """
    kerr = output['kappa.2h.quadpy'] - output['kappa.2h']
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for i in range(3):
        axes[0].plot(setup['arcmin'][i], output['kappa.2h'][i], f'C{i}-')
        axes[0].plot(setup['arcmin'][i], output['kappa.2h.quadpy'][i], f'C{i}--')
        axes[1].plot(setup['arcmin'][i], kerr[i], f'C{i}-', label=str(i))
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel(r'$\theta$ (arcmin)')
    axes[0].set_ylabel(r'$\kappa(\theta)$')
    axes[1].set_ylabel(r'$\kappa_\mathrm{quadpy}/\kappa_\mathrm{scipy}$')
    fig.tight_layout()
    plt.savefig('quadpy_test_kappa.png')
    plt.close()
    """

    return output


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
    #profiles = NFW(
        #setup['mass_range'], c, z, cosmo=cosmo, overdensity=setup['delta'],
        #background=ref, frame=setup['distances'], z_s=1100)
    # need to loop through for the miscentring calculations
    profiles = [NFW(setup['mass_range'], ci, zi, cosmo=cosmo,
                    overdensity=setup['delta'], background=ref,
                    frame=setup['distances'], z_s=1100)
                for ci, zi in zip(c, z[:,0])]
    return profiles


def filter_profiles(ingredients, filter, profiles, theta_fine, theta_bins,
                    units='arcmin'):
    if debug:
        ti = time()
    if ingredients['nzlens']:
        # fix later
        filtered = [np.array(
                        [filter.filter(theta_fine, p_ij, theta_bins,
                                       units=units)
                         for p_ij in p_i])
                    for p_i in profiles]
        filtered = np.transpose(filtered, axes=(2,0,1))
    else:
        #if debug:
            #print('in filter_profiles')
            #print('theta_fine =', theta_fine)
            #print('theta_bins =', theta_bins)
        filtered = [np.array(
                        [filter.filter(theta_fine, prof_mz, theta_bins_z,
                                       units=units)[1]
                         for prof_mz in prof_z])
                    for prof_z, theta_bins_z in zip(profiles, theta_bins)]
        if debug:
            print('filtered =', np.array(filtered).shape)
        filtered = np.transpose(filtered, axes=(2,0,1))
        #filtered = np.array(filtered)
    if debug:
        print(f'filtered in {time()-ti:.2f} s')
    return filtered


def miscentering1(profiles, R, Rmis, tau, Rcl, dist='gamma'):
    if debug:
        print('*** in miscentering ***')
        print('profiles =', profiles)
        print('R =', R, R.shape, type(R[0][0]))
        print('Rmis =', Rmis, Rmis.shape, type(Rmis[0]))
        # this does many more calculations than necessary but will do for now
        # (and also have to do the funny indexing)
        print(profiles.offset_convergence(R[0], Rmis))
    kappa_mis = np.array(
        [profiles.offset_convergence(Ri, Rmis)[:,:,i]
         for i, Ri in  enumerate(R)])
    if debug:
        print('kappa_mis =', kappa_mis.shape)
    if dist == 'gamma':
        p_mis = Rmis/(tau*Rcl[:,None])**2 * np.exp(-Rmis/(tau*Rcl[:,None]))
    if debug:
        print('p_mis =', p_mis.shape)
    # average over Rmis
    kappa_mis = trapz(kappa_mis*p_mis[:,:,None,None], Rmis, axis=1) \
        / trapz(p_mis, Rmis, axis=1)[:,None,None]
    if debug:
        print('kappa_mis =', kappa_mis.shape)
        print('*** end ***')
    return np.transpose(kappa_mis, axes=(1,0,2))


def plot_profile_mz(R, profile, z, mx=None, z0=1, m0=1e14, xscale='log',
                    yscale='log', ylabel='', output='', close=False, fs=14,
                    figaxes=None):
    """
    Parameters
    ----------
    R must have shape ([N,]M), where N is the number of redshift bins and
        M is the number of radial bins
    profile must have shape (N,P,M) where P is the number of mass bins
    """
    if not debug:
        return
    z = np.squeeze(z)
    if len(R.shape) == 1:
        R = np.array([R for i in range(z.size)])
    if mx is None:
        lnm = np.log(m)
    else:
        lnm = np.log(mx)
    logm = np.log10(np.exp(lnm))
    idx_z = np.argmin(np.abs(z-z0))
    idx_m = np.argmin(np.abs(lnm-np.log(m0)))
    if figaxes is None:
        fig, axes = plt.subplots(figsize=(14,5), ncols=2)
    else:
        fig, axes = figaxes
    # varying redshift
    axes[0].set_title('log10(m)={0:.2f}'.format(logm[idx_m]), fontsize=fs)
    colors, cmap = colorscale(z)
    for i in range(z.size):
        axes[0].plot(R[i], profile[i,idx_m], color=colors[i])
    if figaxes is None:
        plt.colorbar(cmap, label='Redshift', ax=axes[0])
    # varying mass
    axes[1].set_title('z={0:.2f}'.format(z[idx_z]), fontsize=fs)
    colors, cmap = colorscale(logm)
    for i in range(lnm.size):
        axes[1].plot(R[idx_z], profile[idx_z,i], color=colors[i])
    if figaxes is None:
        plt.colorbar(cmap, label='log10 m', ax=axes[1])
    for ax in axes:
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if ylabel:
            ax.set_ylabel(ylabel)
#     axes[1].set_x
    if output:
        savefig(output, fig=fig, close=close)
    else:
        fig.tight_layout()
    return fig, axes
