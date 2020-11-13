#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  lens.py
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

from astropy import units as u
from quadpy.line_segment import integrate_adaptive
from numba import float64, jit
#from progressbar import *
import numpy as np
from numpy import exp, log, log10, pi
import matplotlib.pyplot as pl
import multiprocessing as multi
import scipy
from scipy.integrate import cumtrapz, quad, romberg, simps, trapz
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, \
                              UnivariateSpline
import scipy.special as sp
import sys
from time import time

if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange

from . import hankel


"""
# Fourier transform from P to correlation function!
"""

def multi_proc(a,b, que, power_func, R):
    outdict = {}

    r = R[a:b:1]

    # Call power_to_corr function.

    corr = power_to_corr(power_func, r)

    # Write in dictionary, so the result can be read outside of function.

    outdict = np.column_stack((r, corr))
    que.put(outdict)

    return


def power_to_corr_multi(power_func, R):

    #print ('Calculating correlation function.')

    nprocs = multi.cpu_count()#8 # Match the number of cores!
    q1 = multi.Queue()
    procs = []
    chunk = int(np.ceil(len(R)/float(nprocs)))

    for j in range(nprocs):

        work = multi.Process(target=multi_proc, args=((j*chunk), \
                                            ((j+1)*chunk), q1, power_func, R))
        procs.append(work)
        work.start()

    result = np.array([]).reshape(0, 2)

    for j in range(nprocs):
        result = np.vstack([result, np.array(q1.get())])

    result = result[np.argsort(result[:, 0])]
    #print np.array(result)[:,1]

    #print ('Done. \n')
    return result[:,1]


def power_to_corr(power_func, R):
    """
    Calculate the correlation function given a power spectrum

    Parameters
    ----------
    power_func : callable
        A callable function which returns the natural log of power given lnk

    R : array_like
        The values of separation/scale to calculate the correlation at.

    """

    #print ('Calculating correlation function.')

    if not np.iterable(R):
        R = [R]

    corr = np.zeros_like(R)

    # the number of steps to fit into a half-period at high-k.
    # 6 is better than 1e-4.
    minsteps = 8

    # set min_k, 1e-6 should be good enough
    mink = 1e-6

    temp_min_k = 1.0

    for i, r in enumerate(R):
        # getting maxk here is the important part. It must be a half multiple of
        # pi/r to be at a "zero", it must be >1 AND it must have a number of half
        # cycles > 38 (for 1E-5 precision).

        min_k = (2.0 * np.ceil((temp_min_k * r / np.pi - 1.0) / 2.0) + 0.5) * np.pi / r
        maxk = max(501.5 * np.pi / r, min_k)

        # Now we calculate the requisite number of steps to have a good dk at hi-k.
        nk = np.int(np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * r)))))
        #nk = 10000

        lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
        P = power_func(lnk)
        integ = np.exp(P) * (np.exp(lnk) ** 2.0) * np.sin(np.exp(lnk) * r) / r
        corr[i] = (0.5 / (np.pi ** 2.0)) * simps(integ, dx=dlnk)

    return corr


def power_to_corr_ogata(power_func, R):
    result = np.zeros(R.shape)
    h = hankel.SphericalHankelTransform(0,10000,0.00001)
    for i in range(result.size):
        integ = lambda x: exp(power_func(log(x/R[i]))) * \
                          (x**2.0) / (2.0*pi**2.0)
        result[i] = h.transform(integ)[0]
    #print('result =', result.shape)
    #integ2 = lambda X: \
        #exp(power_func(log(X/R[:,None]))) * (X**2) / (2*pi**2)
    #integ2 = lambda X: \
        #exp(power_func(log(X[:,None]/R))) * (X[**2) / (2*pi**2)
    #result2 = h.transform(integ2)
    #for i, r in enumerate(result2):
        #print(i, r, r.shape)
    #print('result2[0] =', result2[0])
    #print('result2 =', np.array(result2).shape)
    return result / R**3


"""
# Lensing calculations from correlation function, i.e. surface density, 
# excess surface density, tangential shear ...
"""


def sigma(corr, rho_mean, r_i, r_x):
    debug = False
    if debug:
        np.set_printoptions(4)
        print('x =', np.array2string(r_i, separator=', '))
        print('f =', np.array2string(corr, separator=', '))
        print()
        print('logx =', np.array2string(log10(r_i), separator=', '))
        print('logf =', np.array2string(log10(1+corr), separator=', '))
        print('x0 =', r_x[0])
        print('rx =', np.array2string(r_x, separator=', '))
        to = time()
    c = UnivariateSpline(log10(r_i), log10(1+corr), s=0, ext=0)
    if debug:
        print('interpolated in {0:.2e} s'.format(time()-to))
    integ = lambda x, rxi: (10**c(log10(rxi/x))-1) / \
                           (x*x * (1-x*x)**0.5)
    #integ = lambda x, rxi: 10**c(log10(rxi/x)) / \
    #                        (x*x * (1-x*x)**0.5)
    #@jit(float64(float64, float64, float64, float64), nopython=False)#, cache=True)
    #def integ(x, rxi, r_i, corr):
        #c = UnivariateSpline(_log10(r_i), _log10(1+corr), s=0, ext=0)
        #return 10**c(_log10(rxi/x)) / (x**2*(1-x**2)**0.5)
    #s = np.array([quad(integ, 0, 1, args=(rxi,r_i,corr), full_output=1)[0]
                  #for rxi in r_x])
    #@jit(float64(float64, float64), nopython=False)#, cache=True)
    #def integ(x, rxi):
        #return 10**c(_log10(rxi/x)) / (x**2*(1-x**2)**0.5)
    to = time()
    s = np.array([quad(integ, 0, 1, args=(rxi,), full_output=1)[0] for rxi in r_x])
    if debug:
        print('integrated in {0:.2e} s'.format(time()-to))
        print('s =', s)
    """
    print('s =', s, s.shape)
    xint = np.logspace(-10, 0, 2000)
    print('xint =', xint.shape, r_x.shape)
    print('integ =', integ(xint, r_x[:,None]).shape)
    s2 = np.array([romberg(integ, 0, 1, args=(rxi,))
                   for rxi in r_x])
    print('s2 =', s2, s2.shape)
    print(np.allclose(s,s2))
    """
    """
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.loglog(r_x, s, label='quad')
    plt.loglog(r_x, s2, label='simps')
    plt.legend()
    plt.subplot(122)
    plt.plot(r_x, s2/s-1)
    plt.show()
    """
    return 2.0 * rho_mean * s * r_x


def d_sigma(sigma, r_i, r_x):
    # This is correct way to do it!
    s = np.zeros(len(r_x))
    err = np.zeros(len(r_x))
    c = UnivariateSpline(np.log10(r_i), np.log10(sigma),s=0, ext=0, k=1)
    x_int = np.linspace(0.0, 1.0, 1000, endpoint=True)
    for i, rxi in enumerate(r_x):
        integ = lambda x: 10.0**(c(np.log10(x*rxi)))*x # for int from 0 to 1
        s[i] = cumtrapz(np.nan_to_num(integ(x_int)), x_int, initial=None)[-1]
    # Not subtracting sigma because the range is different!
    # Subtracting interpolated function!
    d_sig = (2.0*s - 10.0**(c(np.log10(r_x))))
    return d_sig


def wp(corr, r_i, r_x):
    _log10 = log10
    #c = UnivariateSpline(_log10(r_i), _log10(1+corr), s=0, ext=0)
    #integ = lambda x, rxi: 10.0**c(_log10(rxi/x)) / (x**2 * (1-x**2)**0.5)
    
    c = UnivariateSpline(log10(r_i), log10(1+corr), s=0, ext=0)
    integ = lambda x, rxi: (10**c(log10(rxi/x))-1) / \
                           (x*x * (1-x*x)**0.5)
    
    s = np.array([quad(integ, 0, 1, args=(rxi,), full_output=1)[0] for rxi in r_x])
    return 2.0 * s * r_x


def wp_beta_correction(corr, r_i, r_x, omegam, bias):
    # See Cacciato et al. 2012, equations 21 - 28
    # Gives the same resutls as wp above. :/
    
    c = UnivariateSpline(log10(r_i), corr, s=0, ext=1)
    
    beta = omegam**0.6 / bias
    
    leg_0 = lambda x: 1.0
    leg_2 = lambda x: 0.5 * (3.0 * x**2.0 - 1.0)
    leg_4 = lambda x: (1.0/8.0) * (35.0 * x**4.0 - 30.0 * x**2.0 + 3.0)
    
    J_3 = np.empty(len(r_i))
    J_5 = np.empty(len(r_i))
    for i in range(len(r_i)):
        
        x_int = np.linspace(0.0, r_i[i], 10000, endpoint=True)
        
        int_j3 = lambda x: c(log10(x))*x**2.0
        int_j5 = lambda x: c(log10(x))*x**4.0
    
        J_3[i] = (1.0/r_i[i]**3.0) * intg.cumtrapz((int_j3(x_int)), x_int, initial=None)[-1]
        J_5[i] = (1.0/r_i[i]**5.0) * intg.cumtrapz((int_j5(x_int)), x_int, initial=None)[-1]
    
    J_3_interpolated = UnivariateSpline(log10(r_i), J_3, s=0, ext=1)
    J_5_interpolated = UnivariateSpline(log10(r_i), J_5, s=0, ext=1)
    
    xi_0 = lambda x: (1.0 + (2.0/3.0) * beta + (1.0/5.0) * beta**2.0) * c(log10(x))
    xi_2 = lambda x: ((4.0/3.0) * beta + (4.0/7.0) * beta**2.0) * (c(log10(x)) - 3.0*J_3_interpolated(log10(x)))
    xi_4 = lambda x: ((8.0/35.0) * beta**2.0) * (c(log10(x)) + (15.0/2.0)*J_3_interpolated(log10(x)) - (35.0/2.0)*J_5_interpolated(log10(x)))
    
    int_1 = np.zeros(len(r_x))
    int_2 = np.zeros(len(r_x))
    int_3 = np.zeros(len(r_x))
    
    for i in range(len(r_x)):
        x_int = np.linspace(0.0, 100.0, 10000, endpoint=True)
        
        int_1[i] = intg.cumtrapz(np.nan_to_num(xi_0(np.sqrt(r_x[i]**2.0 + x_int**2.0)) * leg_0(x_int/np.sqrt(r_x[i]**2.0 + x_int**2.0))), x_int, initial=None)[-1]
        int_2[i] = intg.cumtrapz(np.nan_to_num(xi_2(np.sqrt(r_x[i]**2.0 + x_int**2.0)) * leg_2(x_int/np.sqrt(r_x[i]**2.0 + x_int**2.0))), x_int, initial=None)[-1]
        int_3[i] = intg.cumtrapz(np.nan_to_num(xi_4(np.sqrt(r_x[i]**2.0 + x_int**2.0)) * leg_4(x_int/np.sqrt(r_x[i]**2.0 + x_int**2.0))), x_int, initial=None)[-1]
    
    w_p = 2.0 * (int_1 + int_2 + int_3)
    
    return w_p


def sigma_crit(cosmo, zl, zs):
    """Critical surface density in Msun/pc^2"""
    # To be consistent in halo model, this needs to be in comoving, thus (1+zl)^2
    beta = (1+zl)**2.0 * cosmo.angular_diameter_distance_z1z2(zl, zs) \
        * cosmo.angular_diameter_distance(zl) \
        / cosmo.angular_diameter_distance(zs)
    # the first factor is c^2/(4*pi*G) in Msun/Mpc
    return 1.6629165e18 / beta.to(u.Mpc).value


def power_to_sigma(power_func, R, order, rho_mean):
    """
    Calculate the projected correlation function given a power spectrum
    using the properties of FAH cycle

    Parameters
    ----------
    power_func : callable
        A callable function which returns the natural log of power given lnk

    R : array_like
        The values of separation/scale to calculate the correlation at.
    
    order : int
        Bessel function order: 0 for clustering, 2 for galaxy-galaxy lensing
    
    rho_mean : float
        Mean density of the universe

    """

    #print ('Calculating correlation function.')

    if not np.iterable(R):
        R = [R]

    sigma = np.zeros_like(R)

    # the number of steps to fit into a half-period at high-k.
    # 6 is better than 1e-4.
    minsteps = 8

    # set min_k, 1e-6 should be good enough
    mink = 1e-6

    temp_min_k = 1.0

    for i, r in enumerate(R):
        # getting maxk here is the important part. It must be a half multiple of
        # pi/r to be at a "zero", it must be >1 AND it must have a number of half
        # cycles > 38 (for 1E-5 precision).

        min_k = (2.0 * np.ceil((temp_min_k * r / np.pi - 1.0) / 2.0) + 0.5) * np.pi / r
        maxk = max(501.5 * np.pi / r, min_k)

        # Now we calculate the requisite number of steps to have a good dk at hi-k.
        nk = np.int(np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * r)))))
        #nk = 10000

        lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
        P = power_func(lnk)
        integ = np.exp(lnk)**2.0 * sp.jv(order, np.exp(lnk) * r) * np.exp(P)
        
        sigma[i] = (rho_mean / (2.0*np.pi)) * simps(integ, dx=dlnk)

    return sigma
    

def power_to_sigma_ogata(power_func, R, order, rho_mean):
    result = np.zeros(R.shape)
    h = hankel.HankelTransform(order,10000,0.00001)
    for i in range(result.size):
        integ = lambda x: exp(power_func(log(x/R[i]))) * x
        result[i] = h.transform(integ)[0]
        
    return (rho_mean * result) / (2.0 * pi * R**2)


if __name__ == '__main__':
    main()

