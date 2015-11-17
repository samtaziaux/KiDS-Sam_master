#!/usr/bin/python

"This modules will compute the 2-halo term of the halo model."

import pyfits
import numpy as np
import sys
import os
from astropy import constants as const, units as u
import glob
import gc

# Van den Bosch 2002

def calc_u(x):
    u = 64.087 * (1 + 1.074*x**0.3 - 1.581*x**0.4 + 0.954*x**0.5 - 0.185*x**0.6)**-10
    return u


def calc_S(M, Omega_m, Omega_b, sigma_8, h):
    
    c0 = 3.904e-4
    Gamma = Omega_m * h * np.exp( -Omega_b * (1 + np.sqrt(2*h)/Omega_m ) )
    
    var_S = (calc_u( (c0*Gamma) / (Omega_m**(1/3)) * M**(1/3) ))**2 * ( sigma_8**2 / (calc_u( 32*Gamma ))**2 )
    
    return var_S
    
    
def Bias_Tinker10(var_S):
    # Tinker 2010 bias - empirical
    
    sigma = np.sqrt(var_S)

    delta_c = 1.686
    delta_halo = 200.

    y = np.log10(delta_halo)
    nu = delta_c/sigma

    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4

    bias = 1 - A * nu ** a / (nu ** a + delta_c ** a) + B * nu ** b + C * nu ** c

    print y, A, B, C, a, b, c

    return bias


def bias(M, Omega_m, omegab_h2, sigma_8, h):
    
    Omega_b = 0.02205/h**2

    var_S = calc_S(M, Omega_m, Omega_b, sigma_8, h)

#    print 'variance S:', var_S
    
    bias = Bias_Tinker10(var_S)
    
#    print 'Bias:', bias
    
#main()
