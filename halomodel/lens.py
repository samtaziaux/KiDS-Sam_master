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
from progressbar import *
import numpy as np
import matplotlib.pyplot as pl
import scipy
import multiprocessing as multi
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp


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
			
		work = multi.Process(target=multi_proc, args=((j*chunk), ((j+1)*chunk), q1, power_func, R))
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

	corr = np.zeros_like(R)#, dtype=np.float128)

	# the number of steps to fit into a half-period at high-k. 6 is better than 1e-4.
	minsteps = 8.0

    # set min_k, 1e-6 should be good enough
	mink = 1.0e-6

	temp_min_k = 1.0
	
	#widgets = ['Corr: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ', ETA()]
	#pbar = ProgressBar(widgets=widgets, maxval=len(R)).start()
	
	for i, r in enumerate(R):
		# getting maxk here is the important part. It must be a half multiple of
		# pi/r to be at a "zero", it must be >1 AND it must have a number of half
		# cycles > 38 (for 1E-5 precision).

		min_k = (2.0 * np.ceil((temp_min_k * r / np.pi - 1.0) / 2.0) + 0.5) * np.pi / r
		maxk = max(501.5 * np.pi / r, min_k)

        # Now we calculate the requisite number of steps to have a good dk at hi-k.
		#nk = np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * r))))
		nk = 10000
		
		lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
		
		P = power_func(lnk)
		
		integ = np.exp(P) * (np.exp(lnk) ** 2.0) * np.sin(np.exp(lnk) * r) / r
		corr[i] = (1.0 / (2.0 * np.pi ** 2.0)) * intg.simps(integ, dx=dlnk)
		
		#pbar.update(i+1)

		#print i, r, corr[i]
	
	#pbar.finish()
	#print ('Done. \n')
	
	return corr
	

"""
# Lensing calculations from correlation function, i.e. surface density, excess surface density, tangential shear ...
"""


def sigma(corr, rho_mean, r_i, r_x):
	
    """
    EXT is either 0 for extrapolation, 1 for zeros and 3 for boudary value! Default is 0.
    """
	
    #print ('Calculating projected surface density.')
    
    import scipy.integrate as intg
	
    s = np.ones(len(r_x))
    err = np.ones(len(r_x))

    c = scipy.interpolate.UnivariateSpline(np.log(r_i), np.log(corr), s=0)
    x_int = np.linspace(0.0, 1.0, 100, endpoint=False)
    
    #widgets = ['pSigma: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=len(r_x)).start()
	
    for i in range(len(r_x)):
        integ = lambda x: (1.0 + np.exp(c(np.log(r_x[i]/x))))/((x**2.0) * ((1.0 - (x**2.0))**0.5)) # for int from 0 to
        s[i], err[i] = intg.quad(integ, 0.0, 1.0)
        #s[i] = intg.cumtrapz(np.nan_to_num(integ(x_int)), x_int, initial=None)[-1]
        
        #pbar.update(i+1)
		
    sig = 2.0 * rho_mean * s * r_x
    #pbar.finish()
    #print ('Done. \n')
    #quit()
    return sig
	
	
def d_sigma(sigma, r_i, r_x):
	
    # This is correct way to do it!
	
    #print ('Calculating excess surface density.')
	
    import scipy.integrate as intg
	
    s = np.ones(len(r_x))
    err = np.ones(len(r_x))
	
    c = scipy.interpolate.UnivariateSpline(np.log(r_i), np.log(sigma),s=0)
    x_int = np.linspace(0.0, 1.0, 100)
	
    #widgets = ['dSigma: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=len(r_x)).start()
	
    for i in range(len(r_x)):
		
        integ = lambda x: np.exp(c(np.log(x*r_x[i])))*x # for int from 0 to 1
		
        #s[i], err[i] = intg.quad(integ, 0.0, 1.0)
        s[i] = intg.cumtrapz(np.nan_to_num(integ(x_int)), x_int, initial=None)[-1]
		
        #pbar.update(i+1)
	
    d_sig = ((2.0)*s - np.exp(c(np.log(r_x)))) #Not subtracting sigma because the range is different! Subtracting interpolated function!
    
    #pbar.finish()
    #print ('Done. \n')
	
    return d_sig
	
	
	

if __name__ == '__main__':
	main()

