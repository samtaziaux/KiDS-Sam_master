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
import time
#from progressbar import *
import numpy as np
import matplotlib.pyplot as pl
import scipy
import multiprocessing as multi
from itertools import izip
from numpy import exp, log, log10, pi
from scipy.integrate import quad, romberg, simps, trapz
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, \
                              UnivariateSpline
import scipy.special as sp
from time import time

import hankel


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

    import scipy.integrate as intg

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
        nk = np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * r))))
        #nk = 10000

        lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
        P = power_func(lnk)
        integ = np.exp(P) * (np.exp(lnk) ** 2.0) * np.sin(np.exp(lnk) * r) / r
        corr[i] = (0.5 / (np.pi ** 2.0)) * intg.simps(integ, dx=dlnk)

    return corr


def power_to_corr_ogata(power_func, R):
    result = np.zeros(R.shape)
    h = hankel.SphericalHankelTransform(0,10000,0.00001)
    for i in xrange(result.size):
        integ = lambda x: exp(power_func(log(x/R[i]))) * \
                          (x**2.0) / (2.0*pi**2.0)
        result[i] = h.transform(integ)[0]
    return result / R**3


"""
# Lensing calculations from correlation function, i.e. surface density, 
# excess surface density, tangential shear ...
"""


def sigma(corr, rho_mean, r_i, r_x):
    _log10 = log10
    c = UnivariateSpline(_log10(r_i), _log10(1+corr), s=0, ext=0)

    integ = lambda x, rxi: 10**c(_log10(rxi/x)) / \
                           (x**2 * (1-x**2)**0.5)
    s = np.array([quad(integ, 0, 1, args=(rxi,), full_output=1)[0]
                  for rxi in r_x])
    #s2 = np.array([romberg(integ, 0, 1, args=(rxi,))
                   #for rxi in r_x])
    return 2.0 * rho_mean * s * r_x


def d_sigma(sigma, r_i, r_x):

    # This is correct way to do it!

    #print ('Calculating excess surface density.')

    import scipy.integrate as intg

    s = np.zeros(len(r_x))
    err = np.zeros(len(r_x))

    c = UnivariateSpline(np.log10(r_i), np.log10(sigma),s=0, ext=0, k=1)
    x_int = np.linspace(0.0, 1.0, 1000, endpoint=True)

    for i in xrange(len(r_x)):

        integ = lambda x: 10.0**(c(np.log10(x*r_x[i])))*x # for int from 0 to 1

        #s[i], err[i] = intg.quad(integ, 0.0, 1.0)
        s[i] = intg.cumtrapz(np.nan_to_num(integ(x_int)), x_int, initial=None)[-1]
    
    # Not subtracting sigma because the range is different!
    # Subtracting interpolated function!
    d_sig = ((2.0)*s - 10.0**(c(np.log10(r_x)))) 

    return d_sig


def wp(corr, r_i, r_x):
    _log10 = log10
    c = UnivariateSpline(_log10(r_i), _log10(1+corr), s=0, ext=0)
    
    integ = lambda x, rxi: 10.0**c(_log10(rxi/x)) / (x**2 * (1-x**2)**0.5)
    s = np.array([quad(integ, 0, 1, args=(rxi,), full_output=1)[0] for rxi in r_x])
        
    return 2.0 * s * r_x


def wp_beta_correction(corr, r_i, r_x, omegam, bias):
    # See Cacciato et al. 2012, equations 21 - 28
    # Gives the same resutls as wp above. :/
    
    import scipy.integrate as intg
    
    _log10 = log10
    c = UnivariateSpline(_log10(r_i), corr, s=0, ext=1)
    
    beta = omegam**0.6 / bias
    
    leg_0 = lambda x: 1.0
    leg_2 = lambda x: 0.5 * (3.0 * x**2.0 - 1.0)
    leg_4 = lambda x: (1.0/8.0) * (35.0 * x**4.0 - 30.0 * x**2.0 + 3.0)
    
    J_3 = np.empty(len(r_i))
    J_5 = np.empty(len(r_i))
    for i in xrange(len(r_i)):
        
        x_int = np.linspace(0.0, r_i[i], 10000, endpoint=True)
        
        int_j3 = lambda x: c(_log10(x))*x**2.0
        int_j5 = lambda x: c(_log10(x))*x**4.0
    
        J_3[i] = (1.0/r_i[i]**3.0) * intg.cumtrapz((int_j3(x_int)), x_int, initial=None)[-1]
        J_5[i] = (1.0/r_i[i]**5.0) * intg.cumtrapz((int_j5(x_int)), x_int, initial=None)[-1]
    
    J_3_interpolated = UnivariateSpline(_log10(r_i), J_3, s=0, ext=1)
    J_5_interpolated = UnivariateSpline(_log10(r_i), J_5, s=0, ext=1)
    
    xi_0 = lambda x: (1.0 + (2.0/3.0) * beta + (1.0/5.0) * beta**2.0) * c(_log10(x))
    xi_2 = lambda x: ((4.0/3.0) * beta + (4.0/7.0) * beta**2.0) * (c(_log10(x)) - 3.0*J_3_interpolated(_log10(x)))
    xi_4 = lambda x: ((8.0/35.0) * beta**2.0) * (c(_log10(x)) + (15.0/2.0)*J_3_interpolated(_log10(x)) - (35.0/2.0)*J_5_interpolated(_log10(x)))
    
    int_1 = np.zeros(len(r_x))
    int_2 = np.zeros(len(r_x))
    int_3 = np.zeros(len(r_x))
    
    for i in xrange(len(r_x)):
        x_int = np.linspace(0.0, 100.0, 10000, endpoint=True)
        
        int_1[i] = intg.cumtrapz(np.nan_to_num(xi_0(np.sqrt(r_x[i]**2.0 + x_int**2.0)) * leg_0(x_int/np.sqrt(r_x[i]**2.0 + x_int**2.0))), x_int, initial=None)[-1]
        int_2[i] = intg.cumtrapz(np.nan_to_num(xi_2(np.sqrt(r_x[i]**2.0 + x_int**2.0)) * leg_2(x_int/np.sqrt(r_x[i]**2.0 + x_int**2.0))), x_int, initial=None)[-1]
        int_3[i] = intg.cumtrapz(np.nan_to_num(xi_4(np.sqrt(r_x[i]**2.0 + x_int**2.0)) * leg_4(x_int/np.sqrt(r_x[i]**2.0 + x_int**2.0))), x_int, initial=None)[-1]
    
    w_p = 2.0 * (int_1 + int_2 + int_3)
    
    return w_p



if __name__ == '__main__':
	main()

