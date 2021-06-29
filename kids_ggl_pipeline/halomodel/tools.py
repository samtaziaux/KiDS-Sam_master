#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
from hmf import MassFunction
import hmf.mass_function.fitting_functions as ff
import numpy as np
import matplotlib.pyplot as pl
import scipy
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp
import dill as pickle
import copy


"""
# Mathematical tools.
"""

def Integrate(func_in, x_array, **kwargs):
    """Simpson integration on fixed spaced data"""
    func_in = np.nan_to_num(func_in)
    result = trapz(func_in, x_array, **kwargs)
    return result


def Integrate1(func_in, x_array): # Gauss - Legendre quadrature!

    import scipy.integrate as intg

    #func_in = np.nan_to_num(func_in)

    c = interp1d(np.log(x_array), np.log(func_in), kind='slinear', \
                 bounds_error=False, fill_value=0.0)

    integ = lambda x: np.exp(c(np.log(x)))
    result = intg.fixed_quad(integ, x_array[0], x_array[-1])

    #print ("Integrating over given function.")
    return result[0]


def extrap1d(x, y, step_size, method):

    y_out = np.ones(len(x))


    # Step for stars at n = 76 = 0.006, for DM 0.003 and for gas same as for stars
    #~ step = len(x)*0.005#len(x)*0.0035#0.005!
    step = len(x)*step_size

    xi = np.log10(x)
    yi = np.log10(y)

    xs = xi
    ys = yi

    minarg = np.argmin(np.nan_to_num(np.gradient(yi)))
    grad = np.min(np.nan_to_num(np.gradient(yi)))

    if step < 3:
        step = 3

    #~ print np.exp(xi[minarg])
    #~ print np.exp(xi[minarg] - 0.04)

    if method == 1:
        yslice = ys[(minarg-step):(minarg):1]
        xslice = np.array([xs[(minarg-step):(minarg):1], np.ones(step)])

        w = np.linalg.lstsq(xslice.T, yslice)[0]

    elif method == 2:
        yslice = ys[(minarg-step):(minarg):1]
        xslice = xs[(minarg-step):(minarg):1]

        w = np.polyfit(xslice, yslice, 2)

    elif method == 3:
        #~ yslice = y[(minarg-20):(minarg):1] #60
        #~ xslice = x[(minarg-20):(minarg):1]
        yslice = y[(minarg-20):(minarg):1] #60
        xslice = x[(minarg-20):(minarg):1]

        from scipy.optimize import curve_fit
        def func(x, a, b, c):
            return a * (1.0+(x/b))**(-2.0)

        popt, pcov = curve_fit(func, xslice, yslice, p0=(y[0], x[minarg], 0.0))
        #print popt

    for i in range(len(x)):

        if i > minarg: #100! #65, 125
        #~ if i+start > minarg: #100! #65, 125
        #if i+50 > (np.where(grad < grad_val)[0][0]):

            """
            # Both procedures work via same reasoning, second one actually
            # fits the slope from all the points, so more precise!
            """

            if method == 1:
                y_out[i] = 10.0**(ys[minarg] + (xi[i] - xi[minarg])*(w[0]))

            elif method == 2:
                y_out[i] = 10.0**(w[2] + (xi[i])*(w[1]) + ((xi[i])**2.0)*(w[0]))

            elif method == 3:
                y_out[i] = (func(x[i], popt[0], popt[1], popt[2]))

        else:
            y_out[i] = 10.0**(yi[i])

    return y_out


def extrap2d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def fill_nan(a):
    not_nan = np.logical_not(np.isnan(a))
    indices = np.arange(len(a))
    if not_nan.sum() == 0:
        return a
    else:
        return np.interp(indices, indices[not_nan], a[not_nan])


def clone_git(repository, out_path):
    # keep imports local
    import os
    import git
    """
    Clone and download all contents at server_path
    """
    print('Cloning git repository: {0}'.format(repository))
    git.Git(out_path).clone(repository)


def read_mead_data():
    # keep imports local
    import os
    import shutil
    from scipy.interpolate import RegularGridInterpolator
    
    
    path0 = os.getcwd()
    path = os.path.join(path0, 'BNL_data')
    
    if os.path.exists(os.path.join(path0, 'interpolator_BNL.npy')):
        with open(os.path.join(path0, 'interpolator_BNL.npy'), 'rb') as dill_file:
            beta_interp = pickle.load(dill_file)
        return beta_interp
       
    else:
        if not os.path.exists(path):
            os.makedirs(path)
    
        print('Using non-linear halo bias correction from Mead at al. 2020. Warning, this is slow!')
        if os.path.exists(os.path.join(path, 'BNL/data/MDR1_redshifts.csv')):
            pass
        else:
            clone_git('https://github.com/alexander-mead/BNL', path)
        
        data_dir = os.path.join(path, 'BNL/data/BNL_M200/M512')
        data_dir_folded = os.path.join(path, 'BNL/data/BNL_M200_folded/M512')
        
        snaps = np.genfromtxt(os.path.join(path, 'BNL/data/MDR1_redshifts.csv'), delimiter=',', skip_header=1)[:,[1,2]]
        snaps = snaps[snaps[:,1] >= 52] # takes data up to z=1, might be removed later
        
        file_num = 0
        for i,snap in enumerate(np.flip(snaps[:,1])):
            if os.path.exists(os.path.join(data_dir, 'MDR1_rockstar_%s_bnl.dat'%int(snap))) and os.path.exists(os.path.join(data_dir_folded, 'MDR1_rockstar_%s_bnl.dat'%int(snap))):
                file_num += 1
        
        mass_dim = 12
        mass_interp = np.linspace(11, 16, mass_dim)
        
        out_array = np.empty((file_num * mass_dim * mass_dim * 29, 5))
        out_array_tmp = np.empty((8 * 8 * 29, 4))
        index_array = np.empty((file_num * mass_dim * mass_dim * 29, 4), dtype=int)
        index_array_tmp = np.empty((8 * 8 * 29, 3), dtype=int)
        id = 0
        i_tmp = 0
    
        for i,snap in enumerate(np.flip(snaps[:,1])):
            if os.path.exists(os.path.join(data_dir, 'MDR1_rockstar_%s_bnl.dat'%int(snap))) and os.path.exists(os.path.join(data_dir_folded, 'MDR1_rockstar_%s_bnl.dat'%int(snap))):
                dat = np.genfromtxt(os.path.join(data_dir, 'MDR1_rockstar_%s_bnl.dat'%int(snap)), usecols=(0,1))
                mass = np.genfromtxt(os.path.join(data_dir, 'MDR1_rockstar_%s_binstats.dat'%int(snap)))

                dat_folded = np.genfromtxt(os.path.join(data_dir_folded, 'MDR1_rockstar_%s_bnl.dat'%int(snap)), usecols=(0,1))
                data_del = dat[dat[:,0]<=0.61]
                split_dat = np.split(data_del, len(data_del)/24)
    
                data_fold_del = dat_folded[dat_folded[:,0]>0.61]
                split_dat_folded = np.split(data_fold_del, len(data_fold_del)/5)

                combined = [np.vstack((split_dat[i], split_dat_folded[i])) for i in np.arange(0, len(split_dat),1)]
                combined_array = np.asarray(np.concatenate(combined))
            
                idx = 0
                for m1, mass1 in enumerate(mass[:,2]):
                    for m2, mass2 in enumerate(mass[:,2]):
                        for k, k_val in enumerate(np.unique(combined_array[:,0])):
                            out_array_tmp[idx,:] = np.array([mass1, mass2, k_val, combined_array[idx,1]])
                            index_array_tmp[idx,:] = np.array([m1, m2, k])
                            idx += 1
                        
                data_tmp = np.zeros((8, 8, 29))
                data_tmp[index_array_tmp[:,0], index_array_tmp[:,1], index_array_tmp[:,2]] = out_array_tmp[:,3]
                interp = RegularGridInterpolator((mass[:,2], mass[:,2], np.unique(combined_array[:,0])), data_tmp, fill_value=1.0, bounds_error=False)
                idx = 0
                for m1, mass1 in enumerate(mass_interp):
                    for m2, mass2 in enumerate(mass_interp):
                        for k, k_val in enumerate(np.unique(combined_array[:,0])):
                            out_array[id,:] = np.array([snaps[i,0], mass1, mass2, k_val, interp([mass1, mass2, k_val])])
                            index_array[id,:] = np.array([i_tmp, m1, m2, k])
                            idx += 1
                            id += 1
                i_tmp += 1
            else:
                continue
        
        print('Interpolating...')
        x = out_array[:,:4]
        data = out_array[:,4]
        data = np.zeros((file_num, mass_dim, mass_dim, 29))
        data[index_array[:,0], index_array[:,1], index_array[:,2], index_array[:,3]] = out_array[:,4]
        my_interpolating_function = RegularGridInterpolator([np.unique(x[:,0]), mass_interp, mass_interp, np.unique(x[:,3])], data, fill_value=1.0, bounds_error=False)
        with open(os.path.join(path0, 'interpolator_BNL.npy'), 'wb') as dill_file:
            pickle.dump(my_interpolating_function, dill_file)
        print('Cleaning the temporary data...')
        shutil.rmtree(path)
        print('Importing BNL pickle...')
        return my_interpolating_function



def load_hmf(z, setup, cosmo_model, sigma8, n_s):
    transfer_params = \
        {'sigma_8': sigma8, 'n': n_s, 'lnk_min': setup['lnk_min'],
         'lnk_max': setup['lnk_max'], 'dlnk': setup['k_step']}
    hmf = []
    dndm = []
    power = []
    nu = []
    m = []
    nu_scaled = []
    fsigma_scaled = []
    rho_mean = np.zeros(z.shape[0])
    #rho_mean_z = np.zeros(z.shape[0])
    
    hmf_init = MassFunction(
            Mmin=setup['logM_min'], Mmax=setup['logM_max'],
            dlog10m=setup['mstep'],
            hmf_model=ff.Tinker10, mdef_model=setup['delta_ref'],
            mdef_params={'overdensity':setup['delta']}, disable_mass_conversion=True, delta_c=1.686,
            cosmo_model=cosmo_model, z=0.0,
            transfer_model=setup['transfer'], **transfer_params)

    for i, zi in enumerate(z):
        hmf_init.update(z=zi)
        dndm.append(hmf_init.dndm)
        power.append(hmf_init.power)
        nu.append(hmf_init.nu)
        m.append(hmf_init.m)
        rho_mean[i] = hmf_init.mean_density0
        #rho_mean_z[i] = hmf[i].mean_density # Add to return
        
    for i, zi in enumerate(z):
        # rescaling mass range for galaxy bias
        min_0 = 2.0
        hmf_init.update(z=zi, Mmin=min_0, Mmax=setup['logM_max'], dlog10m=(setup['logM_max']-min_0)/500.0)
        nu_scaled.append(hmf_init.nu)
        fsigma_scaled.append(hmf_init.fsigma)
        hmf_init.update(z=zi, Mmin=setup['logM_min'], Mmax=setup['logM_max'],
            dlog10m=setup['mstep']) # reset
        
    rho_bg = rho_mean / cosmo_model.Om0 if setup['delta_ref'] == 'SOCritical' \
        else rho_mean
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    rho_bg = np.expand_dims(rho_bg, -1)
    return hmf, rho_bg, np.array(dndm), np.array(power), np.array(nu), np.array(m), np.array(nu_scaled), np.array(fsigma_scaled)


def load_hmf_cov(z, setup, cosmo_model, sigma8, n_s):
    # For the covariance wee keep the old method of calling the halo mass function,
    # as adding all the instances of the hmf to the return is not feasible
    # Also, speed is not necessary a concern for this part.
    transfer_params = \
        {'sigma_8': sigma8, 'n': n_s, 'lnk_min': setup['lnk_min'],
         'lnk_max': setup['lnk_max'], 'dlnk': setup['k_step']}
    hmf = []
    nu = []
    nu_scaled = []
    fsigma_scaled = []
    rho_mean = np.zeros(z.shape[0])
    #rho_mean_z = np.zeros(z.shape[0])
    for i, zi in enumerate(z):
        hmf.append(MassFunction(
            Mmin=setup['logM_min'], Mmax=setup['logM_max'],
            dlog10m=setup['mstep'],
            hmf_model=ff.Tinker10, mdef_model=setup['delta_ref'],
            mdef_params={'overdensity':setup['delta']}, disable_mass_conversion=True, delta_c=1.686,
            cosmo_model=cosmo_model, z=zi,
            transfer_model=setup['transfer'], **transfer_params)
            )
        rho_mean[i] = hmf[i].mean_density0
        nu.append(hmf[i].nu)
        #rho_mean_z[i] = hmf[i].mean_density # Add to return
        
    for i, zi in enumerate(z):
        # rescaling mass range for galaxy bias
        min_0 = 2.0
        hmf[i].update(z=zi, Mmin=min_0, Mmax=setup['logM_max'], dlog10m = (setup['logM_max']-min_0)/500.0)
        nu_scaled.append(hmf[i].nu)
        fsigma_scaled.append(hmf[i].fsigma)
        hmf[i].update(z=zi, Mmin=setup['logM_min'], Mmax=setup['logM_max'],
            dlog10m=setup['mstep']) # reset
        
    rho_bg = rho_mean / cosmo_model.Om0 if setup['delta_ref'] == 'SOCritical' \
        else rho_mean
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    rho_bg = np.expand_dims(rho_bg, -1)
    return hmf, rho_bg, np.array(nu), np.array(nu_scaled), np.array(fsigma_scaled)


def virial_mass(r, rho_mean, delta_halo):
    """
    Returns the virial mass of a given halo radius
    """

    return 4.0 * np.pi * r ** 3.0 * rho_mean * delta_halo / 3.0

def virial_radius(m, rho_mean, delta_halo):
    """
    Returns the virial radius of a given halo mass
    """

    return ((3.0 * m) / (4.0 * np.pi * rho_mean * delta_halo)) ** (1.0 / 3.0)


if __name__ == '__main__':
    main()




