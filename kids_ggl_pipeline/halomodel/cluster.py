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

Andrej Dvornik, Cristóbal Sifón, 2014-2021
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
# disable threading in numpy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from astropy import units as u
from astropy.cosmology import Flatw0waCDM
from astropy.units import Quantity
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from time import time

if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange

# specific to the `cluster` model
import colossus
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo.concentration import concentration
from profiley.nfw import NFW
from profiley.filtering import Filter
from profiley.helpers import lss
import pyccl as ccl

# local
from . import halo
from .lens import d_sigma
from .miscentring import miscentring
from .tools import load_hmf
from .. import hod
from ..helpers import io
from ..helpers._debugging import plot_profile_mz
# debugging
from ..helpers._debugging import import_icecream
ic = import_icecream()


#################
##
## Main function
##
#################


debug = ('--debug' in sys.argv)


def model(theta, R):
    """Cluster lensing halo modeling"""

    observables, selection, ingredients, theta, setup \
        = [theta[1][theta[0].index(name)]
           for name in ('observables', 'selection', 'ingredients',
                        'parameters', 'setup')]

    retvalues = setup['return'] + setup['return_extra']
    # alias
    Mh = setup['mass_range']

    # satellite parameters in the config file are ignored;
    # they may or may not be present
    cosmo, \
        c_pm, c_concentration, c_mor, c_scatter, c_mis, c_twohalo = theta[:7]

    # this order is fixed in helpers.configuration.core.CosmoSection.__init__
    Om0, Ob0, h, sigma8, n_s, m_nu, Neff, w0, wa, Tcmb0, z, nz, z_mlf, zs \
        = cosmo

    # astropy (for profiley)
    cosmo_model = Flatw0waCDM(
        H0=100*h, Ob0=Ob0, Om0=Om0, Tcmb0=Tcmb0, m_nu=[0,0,m_nu]*u.eV,
        Neff=Neff, w0=w0, wa=wa)
    # CCL (for 2-halo things and mass function)
    if setup['backend'] == 'ccl':
        cclcosmo = define_cosmology(cosmo, Tcmb=Tcmb0, m_nu=m_nu)
        mdef = ccl.halos.MassDef(setup['delta'], setup['delta_ref'])

    # probably need to be careful with the normalization of the halo bias
    # function in CCL? See KiDS-GGL issue #184

    ### a final bit of massaging ###

    # leaving this for later
    if observables.mlf:
        nbins = observables.nbins - observables.mlf.nbins
    else:
        nbins = observables.nbins
    output_array = np.empty(observables.nbins, dtype=object)
    #output_array = []

    z = halo.format_z(z, nbins)
    ic(z.shape)

    profiles = define_profiles(setup, cosmo_model, z, c_concentration)

    # convert radii if necessary - this must be done here because it
    # depends on cosmology
    if setup['R_unit'] == 'arcmin':
        if setup['distances'] in ('physical', 'proper'):
            arcmin2kpc = cosmo_model.kpc_proper_per_arcmin
        else:
            arcmin2kpc = cosmo_model.kpc_comoving_per_arcmin
        arcmin2Mpc = arcmin2kpc(z).to(u.Mpc/u.arcmin).value
        # remember that the first element is a zero we added (and should remove)
        # for nfw_stack. I *think* we don't need it in the halo model at all
        R = np.array(R[:,1:] * arcmin2Mpc, dtype=float)
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
        setup['Rfine'] = R.T[1:]

    output = {}

    ### halo mass function ###

    #hmf, dndm = define_hmf(setup, cosmo, cclcosmo)
    if setup['backend'] == 'ccl':# or debug:
        if debug:
            delta_ref = str(setup['delta_ref'])
            if delta_ref not in ('matter', 'critical'):
                setup['delta_ref'] = 'critical' if delta_ref == 'SOCritical' \
                    else 'matter'
            ti = time()
            if setup['backend'] != 'ccl':
                cclcosmo = define_cosmology(cosmo, Tcmb=Tcmb0, m_nu=m_nu)
                mdef = ccl.halos.MassDef(setup['delta'], setup['delta_ref'])
        hmf = ccl.halos.mass_function_from_name('Tinker10')
        hmf = hmf(cclcosmo, mass_def=mdef)
        # this is actually dndlog10m
        dndm = np.array(
            [hmf.get_mass_function(cclcosmo, Mh, ai)
             for ai in 1/(1+z[:,0])])
        dndm = dndm / (Mh/np.log(10))
        if debug:
            print(f'ccl in {time()-ti:.2f} s')
            setup['delta_ref'] = str(delta_ref)
    if setup['backend'] == 'hmf' or debug:
        if debug:
            delta_ref = str(setup['delta_ref'])
            if delta_ref in ('matter', 'critical'):
                setup['delta_ref'] = 'SOMean' if delta_ref == 'matter' \
                    else 'SOCritical'
            ti = time()
        hmf_hmf, rho_bg = load_hmf(z, setup, cosmo_model, sigma8, n_s)
        dndm_hmf = np.array([hmf_i.dndm
                             for hmf_i in hmf_hmf])
        if debug:
            print(f'hmf in {time()-ti:.2f} s')
            setup['delta_ref'] = str(delta_ref)
    if debug:
        ic(dndm)
        print('dndm =', dndm / np.max(dndm, axis=1)[:,None])
        print('dndm_hmf =', dndm_hmf / np.max(dndm_hmf, axis=1)[:,None])
        print()
        print('dndm_hmf/dndm - 1 =', dndm_hmf/dndm - 1)
        #sys.exit()
    if setup['backend'] == 'hmf':
        hmf = hmf_hmf
        dndm = dndm_hmf

    ### selection function and mass-observable relation ###

    output['pop_c'], output['pop_s'] = halo.populations(
        observables, ingredients, selection, Mh, z, theta)
    output['pop_g'] = output['pop_c'] + output['pop_s']
    # note that pop_g already accounts for incompleteness
    ngal, logMh_eff = halo.calculate_ngal(
        observables, output['pop_g'], dndm, Mh)
    info(output, 'pop_g')
    if debug:
        print('ngal =', ngal)#/ngal.max())
        print('logMh_eff =', logMh_eff.shape)

    mass_norm = trapz(dndm*output['pop_g'], Mh, axis=1)
    if debug:
        print('mass_norm =', mass_norm.shape)

    # radii for initial calculation
    Rx = setup['Rfine'] if setup['kfilter'] else R.T[1:]
    if debug:
        print('Rx =', Rx.shape, Rx.dtype)

    ### 1-halo term ###

    # for now only includes sigma, kappa, esd (gg_1h=0)
    qfuncs = ([p.projected for p in profiles],
              [p.convergence for p in profiles],
              [p.projected_excess for p in profiles])
    for profile_name, funcs in zip(('sigma', 'kappa', 'esd'), qfuncs):
        if profile_name not in retvalues:
            continue
        key1h = f'{profile_name}.1h'
        key2h = f'{profile_name}.2h'
        if debug:
            print(f'\n *** {key1h} ***')
        # note that this will probably not work if the function is
        # not analytical (should do a list comp in general)
        if debug:
            print('shape =', profiles[0].shape)
            print(profiles[0])
        output[key1h] = np.array(
            [func(Rx[:,i,None]) for i, func in enumerate(funcs)])
        info(output, key1h)
        if debug:
            print(f'{key1h} =', output[key1h][0,:,-1], output[key1h].shape)
            #plot_profile_mz(Rx.T, np.transpose(prof, axes=(1,2,0)),
                            #z, mx=Mh, z0=0.46, m0=1e14,
                            #output=f'profiles_{profile_name}1h.png')

        # miscentering (1h only for now)
        # c_mis[1] corresponds to f_mis
        if ingredients['miscentring']:
            #prof_mis = miscentering(
                #profiles, Rx, *c_mis[2:], dist=c_mis[0])
            offkey = f'{key1h}.off'
            p_mis = miscentring(c_mis[0], *c_mis[2:])
            if p_mis.ndim == 1:
                p_mis = np.ones((z.size,p_mis.size)) * p_mis
            if debug:
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
            if debug:
                print(f'offset in {time()-ti:.2f} s')
                print(f'{offkey}: {output[offkey].shape}')
            # rename well-centered profile
            output[f'{key1h}.cent'] = output.pop(key1h)
            # total 1h profile
            output[key1h] = (1-c_mis[1])*output[f'{key1h}.off'] \
                + c_mis[1]*output[f'{key1h}.cent']

        output[profile_name] = 1 * output[key1h]

        if ingredients['twohalo']:
            # this only does kappa for now
            output = calculate_twohalo(
                output, setup, observables, ingredients, cclcosmo, mdef,
                dndm, z, profiles, c_pm, c_twohalo, mass_norm)

            #if profile_name in retvalues or key2h in retvalues:
                #plot_profile_mz(R, np.transpose(output['kappa.2h.mz'], axes=(1,2,0)),
                                #z, yscale='linear',
                                #mx=Mh, z0=0.46, xscale='linear',
                                #output='profiles_kappa2h_filtered.png')
                #plot_profile_mz(R, np.transpose(output['kappa.2h.mz']+output['kappa.1h.mz'], axes=(1,2,0)),
                                #z, yscale='linear',
                                #mx=Mh, z0=0.46, xscale='linear',
                                #output='profiles_kappa1h2h_filtered.png')

            print_output(output)
            if debug:
                print(profile_name, key2h)
            ic(profile_name)
            ic(output[profile_name].shape)
            ic(output[profile_name][:,:,-1])
            ic(key2h)
            ic(output[key2h][:,:,-1])
            output[profile_name] += output[key2h]
            ic(profile_name)
            ic(profile_name)
            ic(output[profile_name][:,:,-1])
            ic(output[profile_name][:,:,-1]/output[key1h][:,:,-1])
            if debug:
                #sys.exit()
                pass

        # later allow returning 1h and 2h separately (this will mean
        # filtering the profiles twice!)
        if setup['kfilter']:
            for key in (profile_name, key1h, key2h):
                if key not in retvalues:
                    continue
                output[key] = np.transpose(
                    output[key], axes=(0,2,1))
                output[key] = filter_profiles(
                    ingredients, setup['kfilter'], output[key],
                    setup['bin_centers_fine'], setup['bin_edges'])

        print_output(output)

        # dndm weighting
        for key in (profile_name, key1h, key2h):
            if key not in retvalues:
                continue
            output[key] = trapz(
                (dndm*output['pop_g'])[:,None] * output[profile_name],
                Mh, axis=2) / mass_norm[:,None]
            if debug:
                print(f'weighted {key}: {output[key].shape}')

    print_output(output)

    # for now
    #output['kappa'] = output['kappa'].T
    #info(output, 'kappa')

    if debug:
        print(type(output['esd']), type(list(output['esd'])))
        print('esd_unit =', setup['esd_unit'])
    # will work on a more general version later
    # for now we only accept one observable?
    if 'kappa' in setup['return']:
        #output_array[observables.gm.idx] = list(output['kappa'])
        output_array = output['kappa']
    elif 'esd' in setup['return']:
        if setup['esd_unit'] != 'Msun/Mpc^2':
            if setup['esd_unit'] == 'Msun/pc^2':
                output['esd'] /= 1e12
        #print('idx =', observables.gm.idx)
        #output_array[observables.gm.idx] = list(output['esd'])
        output_array = output['esd']

    #print(type(output_array[observables.gm.idx]))

    #return [model_output, logMh_eff]
    return [output_array, logMh_eff]


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
    """Preamble to cluster.model

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

    if setup['backend'] == 'ccl':
        if setup['delta_ref'] not in ('critical', 'matter'):
            if setup['delta_ref'] == 'SOCritical':
                setup['delta_ref'] = 'critical'
            else:
                setup['delta_ref'] = 'matter'
    else:
        if setup['delta_ref'] == 'critical': setup['delta_ref'] = 'SOCritical'
        elif setup['delta_ref'] == 'matter': setup['delta_ref'] = 'SOMean'

    # for now
    setup['return'] = [setup['return']]

    if setup['backend'] == 'ccl':
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


def calculate_sigma_2h(setup, cclcosmo, mdef, z, A_2h, threads=1):
    # for tests
    out = {}
    a = 1 / (1+z[:,0])
    m = setup['mass_range']
    # matter power spectra
    k = setup['k_range_lin']
    Pk = np.array(
        [ccl.linear_matter_power(cclcosmo, k, ai) for ai in a])
    # halo bias
    bias = setup['bias_func'](cclcosmo, mass_def=mdef)
    # A_2h can only be a constant in the current implementation
    bh = A_2h * np.array([bias.get_halo_bias(cclcosmo, m, ai) for ai in a])
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

    if debug:
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
    sigma_2h_z = lss.xi2sigma(
        setup['R2h'], Rxi, xi_z, rho_m, threads=1, full_output=False,
        integrator='scipy', run_test=False, verbose=2*debug)
    # are we sure we need this? If so, should move to xi2sigma, probably
    if setup['distances'] == 'comoving':
        sigma_2h_z = (1 + z)**2 * sigma_2h_z
    if debug:
        plot_profile_mz(setup['R2h'], bh[:,:,None]*sigma_2h_z[:,None], z[:,0],
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


def calculate_twohalo(output, setup, observables, ingredients, cclcosmo, mdef,
                      dndm, z, profiles, c_pm, A_2h, mass_norm):
    # I'm sort of assuming that the kfilter will always be for kappa
    setup['R2h'] = setup['Rfine'].T if setup['kfilter'] \
        else setup['rvir_range_3d_interp']
    if debug:
        print('R2h =', setup['R2h'].shape)

    #ti = time()
    output['sigma.2h.raw'] = calculate_sigma_2h(
        setup, cclcosmo, mdef, z, A_2h)
    #tf = time()
    #print(f'calculate_sigma_2h in {tf-ti:.1f} s')

    info(output, 'sigma.2h.raw')
    if debug:
        plot_profile_mz(setup['R2h'], output['sigma.2h.raw'], z,
                        mx=setup['mass_range'], z0=0.46,
                        output='profiles_sigma.png')

    #######################
    ## could it be the units are not consistent here?? I think we're good
    #######################

    # I should probably be able to do this after averaging over mass
    # and redshift
    retvalues = setup['return'] + setup['return_extra']
    if 'esd' in retvalues or 'esd.2h' in retvalues:
        output['esd.2h.raw'] = np.transpose(
            [halo.calculate_esd(
                setup, observables, ingredients, sigma_2h_m, c_pm)
             for sigma_2h_m in output['sigma.2h.raw'].swapaxes(0,1)],
             #for sigma_2h_m in output['sigma.2h.raw']],
            #axes=(1,0,2))
            axes=(1,0,2))
        if debug:
            info(output, 'esd.2h.raw')
            plot_profile_mz(observables.gm.R[0], output['esd.2h.raw'], z,
                            mx=setup['mass_range'], z0=0.46,
                            output='profiles_esd2h.png')

    if 'kappa' in retvalues or 'kappa.2h' in retvalues:
        output['kappa.2h.mz.raw'] = output['sigma.2h.raw'] \
            / profiles.sigma_crit()[:,:,None]

        if debug:
            info(output, 'kappa.2h.mz.raw')
            plot_profile_mz(setup['R2h'], output['kappa.2h.mz.raw'], z,
                            mx=setup['mass_range'], z0=0.46,
                            output='profiles_kappa.png')

    # it seems now that kappa.2h.mz.raw is OK but something happens when
    # we apply the filter?

    #print_output(output)

    # there's a problem here: if kfilter then the name is kappa.2h.mz
    # but if not the name is kappa.2h
    if setup['kfilter']:
        output['kappa.2h'] = filter_profiles(
            ingredients, setup['kfilter'], output['kappa.2h.raw'],
            setup['bin_centers_fine'], setup['bin_edges'])
        #output['kappa.2h.mz.quadpy'] = filter_profiles(
            #ingredients, setup['kfilter'], output['kappa.2h.mz.raw.quadpy'],
            #setup['bin_centers_fine'], setup['arcmin_bins'])
    else:
        for obs in ('kappa', 'esd'):
            if obs in retvalues or f'{obs}.2h' in retvalues:
                output[f'{obs}.2h'] = np.transpose(
                    output.pop(f'{obs}.2h.raw'), axes=(0,2,1))

                info(output, f'{obs}.2h')

    #R = (setup['bin_edges'][:,1:]+setup['bin_edges'][:,:-1])/2
    #print(R.shape)
    #plot_profile_mz(R.T,
                    #output['kappa.2h.mz'], z[:,0], mx=setup['mass_range'],
                    #z0=0.46, output='profiles_kappa2h_filtered.png')

    #print_output(output)

    #print('mass integrals...')
    """
    for obs in ('kappa', 'esd'):
        if obs in retvalues or f'{obs}.2h' in retvalues:
            output[f'{obs}.2h'] = trapz(
                dndm * output[f'{obs}.2h'], setup['mass_range'], axis=2) \
                / mass_norm
            #output[f'{obs}.2h'] = output[f'{obs}.2h'].T
            #print(f'in {time()-ti:.2f} s')

            info(output, f'{obs}.1h')
            info(output, f'{obs}.2h')
    """

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


def define_cosmology(cosmo_params, Tcmb=2.725, m_nu=0.0, backend='ccl'):
    # this order is fixed in helpers.configuration.core.CosmoSection.__init__
    Om0, Ob0, h, sigma8, n_s, m_nu, Neff, w0, wa, Tcmb0, z, nz, z_mlf, zs \
        = cosmo_params
    if backend == 'ccl':
        cosmo = ccl.Cosmology(
            Omega_c=Om0-Ob0, Omega_b=Ob0, h=h, sigma8=sigma8, n_s=n_s,
            T_CMB=Tcmb0, Neff=Neff, m_nu=m_nu, w0=w0, wa=wa)
    elif backend == 'astropy':
        cosmo = halo.load_cosmology(cosmo_params)[0]
    # colossus (for mass-concentration relation)
    params = dict(Om0=Ob0, H0=100*h, ns=n_s, sigma8=sigma8, Ob0=Ob0)
    colossus_cosmology.setCosmology('cosmo', params)
    return cosmo


def define_hmf(setup, cosmo, mdef):
    return hmf, dndm

def define_profiles(setup, cosmo, z, c_concentration):
    ref = setup['delta_ref']
    # convert between hmf and CCL conventions
    ref = ref[0] if ref in ('critical', 'matter') \
        else ref[2].lower()
    # are we using a KiDS-GGL function or a COLOSSUS function?
    if callable(c_concentration[0]):
        c = c_concentration[0](setup['mass_range'], *c_concentration[1:])
    else:
        model, fc = c_concentration
        bg = f"{int(setup['delta'])}{ref}"
        ic.enable()
        ic(setup['mass_range'])
        ic(bg)
        ic(z)
        c = fc * np.array([concentration(setup['mass_range'], bg, zi,
                                         model=model)
                           for zi in z[:,0]])
        ic(c)
        if not debug: ic.disable()
    # need to loop through for the miscentring calculations
    profiles = [NFW(setup['mass_range'], ci, zi, cosmo=cosmo,
                    overdensity=setup['delta'], background=ref,
                    frame=setup['distances'], z_s=1100)
                for ci, zi in zip(c, z[:,0])]
    if debug:
        #uk = np.array([p.fourier(setup['k_range_lin']) for p in profiles])
        #print('uk_c =', uk, uk.shape)
        #p = [prof.profile
        pass
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
        filtered = np.transpose(filtered, axes=(0,2,1))
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
        filtered = np.transpose(filtered, axes=(0,2,1))
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

