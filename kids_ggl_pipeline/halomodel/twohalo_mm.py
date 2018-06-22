from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
import multiprocessing as multi
import numpy as np
import mpmath as mp
import matplotlib.pyplot as pl
import scipy
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp
from hmf import MassFunction
from hmf import fitting_functions as ff
from hmf import transfer_models as tf
from astropy.cosmology import LambdaCDM

from . import longdouble_utils as ld
from .lens import (
    power_to_corr, power_to_corr_multi, sigma, d_sigma, power_to_corr_ogata)


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


def dsigma_mm(sigma_8, h, omegab_h2, omegam, omegav, n, z, R):
    
    # This function provides one 2-halo matter-matter power spectrum, it's correlation function and projected surface density and corresponding lensing signal, as it says in the return statement.
    # For the input it takes usual cosmological parameters, and numpy array containing radii at which to calculate the ESD. ESD is in M_sun/(h^-1 pc^2).
    
    
    ##########################################################
    # Not tested, but it should work! If not, let me know... #
    ##########################################################
    
    np.seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')
    
    
    # HMF set up parameters - for now fixed and not setable from config file.
    
    n_bins = 10000
    
    M_min = 5.0
    M_max = 16.0
    step = (M_max-M_min)/100
    
    k_min = -13.0 #ln!!! not log10!
    k_max = 17.0 #ln!!! not log10!
    k_step = (k_max-k_min)/n_bins
    
    k_range = np.arange(k_min, k_max, k_step)
    k_range_lin = np.exp(k_range)
    
    mass_range = np.arange(M_min,M_max,step)
    mass_range_lin = (10.0 ** (mass_range))
    
    # Setting parameters from config file
    

    h = H0/100.0
    cosmo_model = LambdaCDM(H0=H0, Ob0=omegab/h**2.0, Om0=omegam, Ode0=omegav, Tcmb0=2.725)

    transfer_params = {'sigma_8': sigma_8, 'n': n, 'lnk_min': k_min ,'lnk_max': k_max, 'dlnk': k_step, 'z':z}
    # Calculation
    
    
    hmf = MassFunction(Mmin=M_min, Mmax=M_max, dlog10m=step, hmf_model=ff.Tinker10, delta_h=200.0, delta_wrt='mean', delta_c=1.686, **transfer_params)
    hmf.update(cosmo_model=cosmo_model)
    
    rho_crit = hmf.mean_dens_z/(hmf.omegac+hmf.omegab)
    rho_mean = hmf.mean_dens_z
    
    radius_range_3d = 10.0 ** np.linspace(-4.0, 4.0, 1000, endpoint=True)
    
    radius_range_3d_i = 10.0 ** np.linspace(-2.5, 1.5, 25, endpoint=True)
    radius_range_2d_i = R
    
    p_2h = hmf.power
    
    xi = power_to_corr_ogata(scipy.interpolate.UnivariateSpline(k_range, p_2h, s=0, ext=0), radius_range_3d)

    sur_den = sigma(xi, rho_mean, radius_range_3d, radius_range_3d_i)
    
    d_sur_den = np.nan_to_num(d_sigma(sur_den, radius_range_3d_i, radius_range_2d_i))/10.0**12.0
    
    out_esd = scipy.interpolate.UnivariateSpline(radius_range_2d_i, np.nan_to_num(d_sur_den), s=0)
    
    out_esd_inter = out_esd(radius_range_2d_i)

    return out_esd_inter, p_2h, xi, sur_den


if __name__ == '__main__':
    
    print(0)
        

