#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  halo.py
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

# Halo profile code
# Andrej Dvornik, 2014/2015

import time
import multiprocessing as multi
from progressbar import *
import numpy as np
import mpmath as mp
import longdouble_utils as ld
import matplotlib.pyplot as pl
import scipy
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp
import sys
sys.path.insert(0, '/home/dvornik/MajorProject/pylib/lib/python2.7/site-packages/')
import hmf.tools as ht
from hmf import MassFunction

import baryons
from tools import Integrate, Integrate1, extrap1d, extrap2d, fill_nan, gas_concentration, star_concentration
from lens import power_to_corr, power_to_corr_multi, sigma, d_sigma
from dark_matter import NFW, NFW_Dc, NFW_f, Con, DM_mm_spectrum, GM_cen_spectrum, GM_sat_spectrum, delta_NFW, GM_cen_analy, GM_sat_analy
from cmf import *


"""
#-------- Declaring functions ----------
"""
	
	
"""
# --------------- Actual halo functions and stuff ------------------
"""


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

@memoize
def Mass_Function(M_min, M_max, step, k_min, k_max, k_step, name, **cosmology_params):
	
	m = MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=step, mf_fit=name, delta_h=200.0, delta_wrt='mean', cut_fit=False, z2=None, nz=None, delta_c=1.686, **cosmology_params)
	
	return m


"""
# Components of density profile from Mohammed and Seljak 2014
"""

def T_n(n, rho_mean, z, M, R, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	
	np.seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')
    
	"""
	Takes some global variables! Be carefull if you remove or split some stuff to different container! 
	"""
	n = np.float64(n)
	
	if len(M.shape) == 0:
		T = np.ones(1)
		M = np.array([M])
		R = np.array([R])
	else:
		T = np.ones(len(M), dtype=np.longdouble)
		
	if profile == 'dm':
		for i in range(len(M)):
		
			i_range = np.linspace(0,R[i],100)

			Ti = (mp.mpf(4.0 * np.pi)/(M[i] * (mp.factorial(2.0*n + 1.0)))) * mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * (NFW(rho_mean, Con(z, M[i]), i_range)), i_range))
			
			T[i] = ld.string2longdouble(str(Ti))
	
	elif profile == 'gas':
		for i in range(len(M)):
		
			i_range = np.linspace(0,R[i],100)
	
			Ti = (mp.mpf(4.0 * np.pi)/(M[i] * (mp.factorial(2.0*n + 1.0)))) * mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * (baryons.u_g(np.float64(i_range), slope, r_char[i], omegab, omegac, M[i])), i_range))
		
			T[i] = ld.string2longdouble(str(Ti))
		
	elif profile == 'stars':
		for i in range(len(M)):
		
			i_range = np.linspace(0,R[i],100)
	
			Ti = (mp.mpf(4.0 * np.pi)/(M[i] * (mp.factorial(2.0*n + 1.0)))) * mp.mpf(Integrate((i_range**(2.0 * (1.0 + n))) * (baryons.u_s(np.float64(i_range), slope, r_char[i], h_mass, M[i], sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)), i_range))
			
			T[i] = ld.string2longdouble(str(Ti))

	return T
	

"""
# Integrals of mass functions with density profiles and population functions.
"""

def f_k(k_x):

	F = sp.erf(k_x/0.1) #0.05!
	
	return F

	
def multi_proc_T(a,b, que, n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	outdict = {}
	
	r = np.arange(a, b, 1)

	T = np.ones((len(r), len(m_x)), dtype=np.longdouble)
		
	for i in range(len(r)):
		T[i,:] = T_n(r[i], rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)
	
	# Write in dictionary, so the result can be read outside of function.
		
	outdict = np.column_stack((r, T))
	que.put(outdict)
	
	return


def T_table_multi(n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	
	n = (n+4)/2
	
	nprocs = multi.cpu_count() # Match the number of cores!
	q1 = multi.Queue()
	procs = []
	chunk = int(np.ceil(n/float(nprocs)))
	
    #widgets = ['Calculating T: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=nprocs).start()
	
	for j in range(nprocs):
			
		work = multi.Process(target=multi_proc_T, args=((j*chunk), ((j+1)*chunk), q1, n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2))
		procs.append(work)
		work.start()

	result = np.array([]).reshape(0, len(m_x)+1)
	
	for j in range(nprocs):
		result = np.vstack([result, np.array(q1.get())])
	
    #pbar.finish()
	result = result[np.argsort(result[:, 0])]

	return np.delete(result, 0, 1)


def T_table(n, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2):
	
	"""
	Calculates all the T integrals and saves them into a array, so that the calling of them is fast for all other purposes.
	"""
	
	n = n+2
	
	T = np.ones((n/2, len(m_x)))
	
    #widgets = ['Calculating T: ', Percentage(), ' ', Bar(marker='-',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=n/2).start()
	
	for i in range(0, n/2, 1):
		T[i,:] = T_n(i, rho_mean, z, m_x, r_x, h_mass, profile, omegab, omegac, slope, r_char, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)
        #pbar.update(i+1)
	
    #pbar.finish()
		
	return T
	
	
def n_gal(z, mass_function, population, x, r_x): # Calculates average number of galaxies!

	integrand = mass_function.dndlnm*population/x
	
	n = Integrate(integrand, x)

	return n
	
	
def eff_mass(z, mass_func, population, m_x):
	
	integ1 = mass_func.dndlnm*population
	integ2 = mass_func.dndm*population
	
	mass = Integrate(integ1, m_x)/Integrate(integ2, m_x)
	
	return mass
	

	
"""	
# Some bias functions
"""

def Bias(hmf, r_x):
	# PS bias - analytic
		
	bias = 1+(hmf.nu-1)/(hmf.growth*hmf.delta_c)
	
    #print ("Bias OK.")
	return bias
	
	
def Bias_Tinker10(hmf, r_x):
	# Tinker 2010 bias - empirical
	
    nu = np.sqrt(hmf.nu)
    y = np.log10(hmf.delta_halo)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4

	# print y, A, a, B, b, C, c
    return 1 - A * nu ** a / (nu ** a + hmf.delta_c ** a) + B * nu ** b + C * nu ** c	
	
	
"""
# Two halo term for matter-galaxy specta! For matter-matter it is only P_lin!
"""

def TwoHalo(mass_func, norm, population, k_x, r_x, m_x): # This is ok!
	
	P2 = (np.exp(mass_func.power)/norm)*(Integrate((mass_func.dndlnm*population*Bias_Tinker10(mass_func,r_x)/m_x),m_x))
	
    #print ("Two halo term calculated.")
	
	return P2

def model(theta, R, h=1, Om=0.315, Ol=0.685, rmax=2):
    #start = time.time()
    from itertools import count, izip
    np.seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')
    
    
    # HMF set up parameters - for now fixed and not setable from config file.
    
    
    expansion = 100
    expansion_stars = 160
    
    n_bins = 10000
    
    M_min = 8.5
    M_max = 18.5#15.5
    step = (M_max-M_min)/100 # or n_bins
    
    k_min = -6.0 #ln!!! not log10!
    k_max = 9.0 #ln!!! not log10!
    k_step = (k_max-k_min)/n_bins
    
    k_range = np.arange(k_min, k_max, k_step)
    k_range_lin = np.exp(k_range)
    
    M_star_min = 8.5 # Halo mass bin range
    M_star_max = 18.5#15.5
    step_star = (M_star_max-M_star_min)/100
    
    mass_range = np.arange(M_min,M_max,step)
    mass_range_lin = 10.0 ** (mass_range)
    
    mass_star_log = np.arange(M_star_min,M_star_max,step_star, dtype=np.longdouble)
    mass_star = 10.0 ** (mass_star_log)
    
    
    # Setting parameters from config file
    
    
    omegab = 0.0455 # To be replaced by theta
    omegac = 0.226 # To be replaced by theta
    omegav = 0.726 # To be replaced by theta
    
    sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, alpha_star, beta_gas, r_t0, r_c0, z, M_bin_min1, M_bin_min2, M_bin_min3, M_bin_min4, M_bin_min5, M_bin_min6, M_bin_max1, M_bin_max2, M_bin_max3, M_bin_max4, M_bin_max5, M_bin_max6, smth1, smth2 = theta
    
    M_bin_min = np.log10([M_bin_min1, M_bin_min2, M_bin_min3, M_bin_min4, M_bin_min5, M_bin_min6]) # Expanded according to number of bins!
    M_bin_max = np.log10([M_bin_max1, M_bin_max2, M_bin_max3, M_bin_max4, M_bin_max5, M_bin_max6]) # Expanded according to number of bins!
    
    hod_mass = [(10.0 ** (np.arange(Mi, Mx, (Mx - Mi)/100, dtype=np.longdouble))) for Mi, Mx in izip(M_bin_min, M_bin_max)]
    
    r_t0 = r_t0*np.ones(100)
    r_c0 = r_c0*np.ones(100)
    
    H0 = 70.4 # To be replaced by theta
    rho_crit = 2.7763458 * (10.0**11.0) # in M_sun*h^2/Mpc^3 # To be called from nfw_utils!
    
    cosmology_params = {"sigma_8": 0.81, "H0": 70.4,"omegab": 0.0455, "omegac": 0.226, "omegav": 0.728, "n": 0.967, "lnk_min": k_min ,"lnk_max": k_max, "dlnk": k_step, "transfer_fit": "CAMB", "z":z} # Values to be read in from theta
    
    # Calculation
    
    
    hmf = Mass_Function(M_min, M_max, step, k_min, k_max, k_step, "Tinker10", **cosmology_params) # Tinker10 should also be read from theta!
    
    mass_func = hmf.dndlnm
    #power = hmf.power
    rho_mean = rho_crit*(hmf.omegac+hmf.omegab)
    rho_mean_int = Integrate(mass_func, mass_range_lin)
    
    
    
    radius_range_lin = ht.mass_to_radius(mass_range_lin, rho_mean)/((200)**(1.0/3.0))
    radius_range = np.log10(radius_range_lin)
    radius_range_3d = 10.0 ** np.arange(-3.0, 3.0, (3.0 - (-3.0))/(50))
    
    radius_range_3d_i = 10.0 ** np.arange(-3.0, 1.2, (1.2 - (-3.0))/(50))
    radius_range_2d_i = R[0] #10.0 ** np.arange(-2.5, 1.0, (1.0 - (-2.5))/(50)) # THIS IS R!
    radius_range_2d_i = radius_range_2d_i[1:]
    
    # Calculating halo model
    
    
    ngal = np.array([n_gal(z, hmf, ngm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) , mass_range_lin, radius_range_lin) for i in izip(hod_mass)])
    rho_dm = baryons.rhoDM(hmf, mass_range_lin, omegab, omegac)
    #rho_stars = np.array([baryons.rhoSTARS(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    #rho_gas = np.array([baryons.rhoGAS(hmf, rho_crit, omegab, omegac, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])[:,0]
    #F = np.array([baryons.rhoGAS(hmf, rho_crit, omegab, omegac, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])[:,1]
    
    norm2 = rho_mean_int/rho_mean
    
    effective_mass = np.array([eff_mass(z, hmf, ngm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2), mass_range_lin) for i in izip(hod_mass)])
    effective_mass_dm = np.array([eff_mass(z, hmf, ncm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2), mass_range_lin)*baryons.f_dm(omegab, omegac) for i in izip(hod_mass)])
    #effective_mass2 = effective_mass*(omegab/(omegac+omegab))
    #effective_mass_bar = np.array([effective_mass2*(baryons.f_stars(i[0], effective_mass2, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2)) for i in izip(hod_mass)])
    
    T_dm = np.array([T_table(expansion, rho_dm, z, mass_range_lin, radius_range_lin, i[0], "dm", omegab, omegac, 0, 0, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    #T_stars = np.array([T_table_multi(expansion_stars, rho_mean, z, mass_range_lin, radius_range_lin, i[0], "stars", omegab, omegac, alpha_star, r_t0, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    #T_gas = np.array([T_table_multi(expansion, rho_mean, z, mass_range_lin, radius_range_lin, i[0], "gas", omegab, omegac, beta_gas, r_c0, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    T_tot = np.array([T_dm[i][0:1:1,:] for i in range(len(M_bin_min))])# + T_stars[i][0:1:1,:] + T_gas[i][0:1:1,:] for i in range(len(M_bin_min))])
    
    F_k1 = f_k(k_range_lin)
    
    pop_c = np.array([ncm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    pop_s = np.array([nsm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    pop_g = np.array([ngm(hmf, i[0], mass_range_lin, sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2) for i in izip(hod_mass)])
    
    # Galaxy - dark matter spectra
    
    Pg_2h = np.array([TwoHalo(hmf, ngal_i, pop_g_i, k_range_lin, radius_range_lin, mass_range_lin) for ngal_i, pop_g_i in izip(ngal, pop_g)])
    
    Pg_c = np.array([F_k1 * GM_cen_spectrum(hmf, z, rho_dm, rho_mean, expansion, pop_c_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_tot_i) for pop_c_i, ngal_i, T_dm_i, T_tot_i in izip(pop_c, ngal, T_dm, T_tot)])
    Pg_s = np.array([F_k1 * GM_sat_spectrum(hmf, z, rho_dm, rho_mean, expansion, pop_s_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_tot_i) for pop_s_i, ngal_i, T_dm_i, T_tot_i in izip(pop_s, ngal, T_dm, T_tot)])
    
    # Galaxy - stars spectra
    
    #Ps_c = np.array([F_k1 * baryons.GS_cen_spectrum(hmf, z, rho_stars_i, rho_mean, expansion_stars, pop_c_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_stars_i, T_tot_i) for rho_stars_i, pop_c_i, ngal_i, T_stars_i, T_tot_i in izip(rho_stars, pop_c, ngal, T_stars, T_tot)])
    #Ps_s = np.array([F_k1 * baryons.GS_sat_spectrum(hmf, z, rho_stars_i, rho_mean, expansion, pop_s_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_stars_i, T_tot_i) for rho_stars_i, pop_s_i, ngal_i, T_dm_i, T_stars_i, T_tot_i in izip(rho_stars, pop_s, ngal, T_dm, T_stars, T_tot)])
    
    # Galaxy - gas spectra
    
    #Pgas_c = np.array([F_k1 * baryons.GGas_cen_spectrum(hmf, z, F_i, rho_gas_i, rho_mean, expansion, pop_c_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_gas_i, T_tot_i) for F_i, rho_gas_i, pop_c_i, ngal_i, T_gas_i, T_tot_i in izip(F, rho_gas, pop_c, ngal, T_gas, T_tot)])
    #Pgas_s = np.array([F_k1 * baryons.GGas_sat_spectrum(hmf, z, F_i, rho_gas_i, rho_mean, expansion, pop_s_i, ngal_i, k_range_lin, radius_range_lin, mass_range_lin, T_dm_i, T_gas_i, T_tot_i) for F_i, rho_gas_i, pop_s_i, ngal_i, T_dm_i, T_gas_i, T_tot_i in izip(F, rho_gas, pop_s, ngal, T_dm, T_gas, T_tot)])
    
    # Combined (all) by type
    
    Pg_k_dm = np.array([(ngal_i*rho_dm*(Pg_c_i + Pg_s_i + Pg_2h_i*rho_mean/rho_dm))/(rho_mean*ngal_i) for ngal_i, Pg_c_i, Pg_s_i, Pg_2h_i in izip(ngal, Pg_c, Pg_s, Pg_2h)])
    
    #Pg_k_s = np.array([(ngal_i*rho_stars_i*(Ps_c_i + Ps_s_i))/(rho_mean*ngal_i) for ngal_i, rho_stars_i, Ps_c_i, Ps_s_i in izip(ngal, rho_stars, Ps_c, Ps_s)])
    #Pg_k_g = np.array([(ngal_i*rho_gas_i*(Pgas_c_i + Pgas_s_i))/(rho_mean*ngal_i) for ngal_i, rho_gas_i, Pgas_c_i, Pgas_s_i in izip(ngal, rho_gas, Pgas_c, Pgas_s)])
    
    #Pg_k = np.array([(ngal_i*rho_dm*(Pg_c_i + Pg_s_i + Pg_2h_i*rho_mean/rho_dm) + ngal_i*rho_stars_i*(Ps_c_i + Ps_s_i) + ngal_i*rho_gas_i*(Pgas_c_i + Pgas_s_i))/(rho_mean*ngal_i) for ngal_i, Pg_c_i, Pg_s_i, Pg_2h_i, rho_stars_i, Ps_c_i, Ps_s_i, rho_gas_i, Pgas_c_i, Pgas_s_i in izip(ngal, Pg_c, Pg_s, Pg_2h, rho_stars, Ps_c, Ps_s, rho_gas, Pgas_c, Pgas_s)])  # all components
    
    # Normalized sattelites and centrals for sigma and d_sigma
    
    #Pg_c2 = np.array([(rho_dm/rho_mean)*Pg_c_i for Pg_c_i in izip(Pg_c)])
    #Pg_s2 = np.array([(rho_dm/rho_mean)*Pg_s_i for Pg_s_i in izip(Pg_s)])
     
    #Ps_c2 = np.array([(rho_stars_i/rho_mean)*Ps_c_i for rho_stars_i, Ps_c_i in izip(rho_stars, Ps_c)])
    #Ps_s2 = np.array([(rho_stars_i/rho_mean)*Ps_s_i for rho_stars_i, Ps_s_i in izip(rho_stars, Ps_s)])
     
    #Pgas_c2 =  np.array([(rho_gas_i/rho_mean)*Pgas_c_i for rho_gas_i, Pgas_c_i in izip(rho_gas, Pgas_c)])
    #Pgas_s2 =  np.array([(rho_gas_i/rho_mean)*Pgas_s_i for rho_gas_i, Pgas_s_i in izip(rho_gas, Pgas_s)])
    
    lnPg_k = np.array([np.log(Pg_k_i) for Pg_k_i in izip(Pg_k_dm)]) # Total
    
    P_inter2 = [scipy.interpolate.UnivariateSpline(k_range, np.nan_to_num(lnPg_k_i), s=0) for lnPg_k_i in izip(lnPg_k)]
    
    #pl.plot(k_range_lin, Pg_k_dm[0], '-k', linewidth=2, alpha=1)
    #pl.plot(k_range_lin, Pg_c[0], '-r', linewidth=2, alpha=1)
    #pl.plot(k_range_lin, Pg_s[0], '-g', linewidth=2, alpha=1)
    #pl.plot(k_range_lin, Pg_2h[0], '-b', linewidth=2, alpha=1)
    #pl.yscale('log')
    #pl.xscale('log')
    #pl.tight_layout()
    #pl.show()

    
    """
    # Correlation functions
    """
    
    xi2 = np.zeros((len(M_bin_min), len(radius_range_3d)))
    for i in range(len(M_bin_min)):
        xi2[i,:] = power_to_corr(P_inter2[i], radius_range_3d)
        xi2[xi2 <= 0.0] = np.nan
        xi2[i,:] = fill_nan(xi2[i,:])
    """
    # Projected surface density
    """
    
    sur_den2 = np.array([np.nan_to_num(sigma(xi2_i, rho_mean, radius_range_3d, radius_range_3d_i)) for xi2_i in izip(xi2)])
    for i in range(len(M_bin_min)):
        sur_den2[i][sur_den2[i] <= 0.0] = np.nan
        sur_den2[i][sur_den2[i] >= 10.0**20.0] = np.nan
        sur_den2[i] = fill_nan(sur_den2[i])
    """
    # Excess surface density
    """
    
    d_sur_den2 = np.array([np.nan_to_num(d_sigma(sur_den2_i, radius_range_3d_i, radius_range_3d_i)) for sur_den2_i in izip(sur_den2)])/10.0**12.0
    for i in range(len(M_bin_min)):
        d_sur_den2[i][d_sur_den2[i] <= 0.0] = np.nan
        d_sur_den2[i][d_sur_den2[i] >= 10.0**20.0] = np.nan
        d_sur_den2[i] = fill_nan(d_sur_den2[i])
    
    out_esd_tot = np.array([scipy.interpolate.UnivariateSpline(radius_range_3d_i, np.nan_to_num(d_sur_den2_i), s=0) for d_sur_den2_i in izip(d_sur_den2)])
    out_esd_tot_inter = np.zeros((len(M_bin_min), len(radius_range_2d_i)))
    for i in range(len(M_bin_min)):
        out_esd_tot_inter[i] = out_esd_tot[i](radius_range_2d_i)

    #pl.plot(radius_range_2d_i, out_esd_tot_inter[0], '-r', linewidth=2, alpha=1)
    #pl.plot(radius_range_2d_i, out_esd_tot_inter[1], '-g', linewidth=2, alpha=1)
    #pl.plot(radius_range_2d_i, out_esd_tot_inter[2], '-b', linewidth=2, alpha=1)
    #pl.plot(radius_range_2d_i, out_esd_tot_inter[3], '-y', linewidth=2, alpha=1)
    #pl.plot(radius_range_2d_i, out_esd_tot_inter[4], '-c', linewidth=2, alpha=1)
    #pl.plot(radius_range_2d_i, out_esd_tot_inter[5], '-k', linewidth=2, alpha=1)
    #pl.yscale('log')
    #pl.xscale('log')
    #pl.tight_layout()
    #pl.show()
    
    #pl.plot(mass_range_lin, pop_c[0], '-r', linewidth=1, alpha=1)
    #pl.plot(mass_range_lin, pop_s[0], '-g', linewidth=1, alpha=1)
    #pl.plot(mass_range_lin, pop_g[0], '-b', linewidth=1, alpha=1)

    #pl.plot(mass_range_lin, pop_c[3], '--r', linewidth=1, alpha=1)
    #pl.plot(mass_range_lin, pop_s[3], '--g', linewidth=1, alpha=1)
    #pl.plot(mass_range_lin, pop_g[3], '--b', linewidth=1, alpha=1)

    #pl.plot(mass_range_lin, pop_c[5], ':r', linewidth=1, alpha=1)
    #pl.plot(mass_range_lin, pop_s[5], ':g', linewidth=1, alpha=1)
    #pl.plot(mass_range_lin, pop_g[5], ':b', linewidth=1, alpha=1)


    #pl.xlim([mass_range_lin[0],mass_range_lin[-1]])
    #pl.ylim([0.001,10**3])
    #pl.yscale('log')
    #pl.xscale('log')
    #pl.show()

    #print np.nan_to_num(out_esd_tot_inter)
    #print effective_mass
    #print sigma_c, alpha_s, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, alpha_star, beta_gas, z
    #end = time.time()
    #print end-start

    return [np.nan_to_num(out_esd_tot_inter), effective_mass, 0] # Add other outputs as needed. Total ESD should always be first!


	
if __name__ == '__main__':

    print 0
