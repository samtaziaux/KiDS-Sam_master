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


def download_directory(repository, server_path, out_path):
    # keep imports local
    import os
    from github import Github, GithubException
    """
    Download all contents at server_path
    """
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    g = Github()
    repository = g.get_repo(repository)
    contents = repository.get_dir_contents(server_path)

    for content in contents:
        print('Processing %s' % content.path)
        if content.type == 'dir':
            download_directory(repository, content.path)
        else:
            try:
                path = content.path
                file_content = repository.get_contents(path)
                file_data = file_content.decoded_content
                file_out = open(out_path + '/' + content.name, 'wb')
                file_out.write(file_data)
                file_out.close()
            except (GithubException, IOError) as exc:
                print('Error processing %s: %s', content.path, exc)


def read_mead_data():
    # keep imports local
    import dill as pickle
    import os
    from scipy.interpolate import LinearNDInterpolator
    
    path0 = os.getcwd()
    path = os.path.join(path0, 'BNL_data')

    download_directory('alexander-mead/BNL', '/data/BNL/M512', path)
    
    file_num = (len([name for name in os.listdir(path)]) - 1) // 2
    print(file_num)
    
    # read in the snapshot number and corresponding redshift, will also loop over snap number
    snaps = np.genfromtxt(path + '/MDR1_redshifts.csv', delimiter=',', skip_header=1)[:,[2,3]]
    out_array = np.empty((file_num * 8 * 8 * 25, 5))
    id = 0
    for i,snap in enumerate(np.flip(snaps[:,0])):
        try:
            dat = np.genfromtxt(path+'/MDR1_rockstar_%s_bnl.dat'%int(snap))
            mass = np.genfromtxt(path+'/MDR1_rockstar_%s_binstats.dat'%int(snap))
            idx = 0
            for m1, mass1 in enumerate(mass[:,2]):
                for m2, mass2 in enumerate(mass[:,2]):
                    for k, k_val in enumerate(np.unique(dat[:,0])):
                        out_array[id,:] = np.array([snaps[i,1], mass1, mass2, k_val, dat[idx,1]])
                        idx += 1
                        id += 1
        except:
            pass
    
    print('Interpolating')
    x = out_array[:,:4]
    data = out_array[:,4]
    my_interpolating_function = LinearNDInterpolator(x, data, fill_value=1.0)
    with open(path0+'/interpolator_BNL.npy', 'wb') as dill_file:
        pickle.dump(my_interpolating_function, dill_file)


def load_hmf(z, setup, cosmo_model, sigma8, n_s):
    transfer_params = \
        {'sigma_8': sigma8, 'n': n_s, 'lnk_min': setup['lnk_min'],
         'lnk_max': setup['lnk_max'], 'dlnk': setup['k_step']}
    hmf = []
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
        #rho_mean_z[i] = hmf[i].mean_density # Add to return
    rho_bg = rho_mean / cosmo_model.Om0 if setup['delta_ref'] == 'SOCritical' \
        else rho_mean
    # this is in case redshift is used in the concentration or
    # scaling relation or scatter (where the new dimension will
    # be occupied by mass)
    rho_bg = np.expand_dims(rho_bg, -1)
    return hmf, rho_bg


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




