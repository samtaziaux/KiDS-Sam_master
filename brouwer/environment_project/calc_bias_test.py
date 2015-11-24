#!/usr/bin/python

"This modules will compute the 2-halo term of the halo model."

import pyfits
import numpy as np
import sys
import os
from astropy import constants as const, units as u
import glob
import gc

import matplotlib.pyplot as plt

sys.path.insert(0, '../../halomodel/')
import twohalo_mm

# Van den Bosch 2002
def calc_S(M, Omega_m, Omega_b, sigma_8, h):
    
    c0 = 3.904e-4
    Gamma = Omega_m * h * np.exp( -Omega_b * (1. + np.sqrt(2.*h)/Omega_m ) )
    
    var_S = (calc_u( (c0*Gamma) / (Omega_m**(1./3.)) * M**(1./3.) ))**2. * ( sigma_8**2. / (calc_u( 32.*Gamma ))**2. )
    
    print 'u(M):', (c0*Gamma) / (Omega_m**(1./3.)), M**(1./3.)
    print 'M:', M
    
    print 'variance S:', var_S
    
    return var_S

def calc_u(x):
    print 'x:', x
    u = 64.087 * (1. + 1.074*x**0.3 - 1.581*x**0.4 + 0.954*x**0.5 - 0.185*x**0.6)**-10.
    
    print 'u:', u
    
    return u
    
    
def Bias_Tinker10(var_S):
    # Tinker 2010 bias - empirical
    
    sigma = np.sqrt(var_S)

    delta_c = 1.686
    delta_halo = 200.

    y = np.log10(delta_halo)
    nu = delta_c/sigma

    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4.)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4. / y) ** 4.)
    c = 2.4

    bias = 1. - A * nu ** a / (nu ** a + delta_c ** a) + B * nu ** b + C * nu ** c

    #print y, A, B, C, a, b, c

    return bias


def calc_bias(M, Omega_m, omegab_h2, sigma_8, h):
    
    Omega_b = 0.02205/h**2.

    var_S = calc_S(M, Omega_m, Omega_b, sigma_8, h)
    #print 'variance S:', var_S
    
    bias = Bias_Tinker10(var_S)
    #print 'Bias:', bias
    
    return bias
    

def main():

    M = [0, 1.e12, 1.e13, 1.e14]
    z = [0.1, 0.2, 0.3, 0.4]

    Om = 0.315
    Ol = 0.685
    h = 1.0

    nRbins = 10
    Rmin = 20.
    Rmax = 2000.
    Rstep = (np.log10(Rmax)-np.log10(Rmin))/(nRbins)
    Rbins = 10.**np.arange(np.log10(Rmin), np.log10(Rmax), Rstep)
    Rbins = np.append(Rbins,Rmax)
    R = np.array([(Rbins[r]+Rbins[r+1])/2 for r in xrange(nRbins)])/1.e3

    sigma_8 = 0.829
    omegab_h2 = 0.02205
    n = 0.9603
    
    for i in xrange(len(M)):
        
        bias = calc_bias(M[i], Om, omegab_h2, sigma_8, h)
        print 'Bias:', bias, 'at M =', M[i]
        
        dsigma = twohalo_mm.dsigma_mm(sigma_8, h, omegab_h2, Om, Ol, n, z[i], R)[0]
        print 'dSigma:', dsigma, 'at z =', z[i]
        
        esd_2halo = bias * dsigma

        plt.plot(R, esd_2halo)
        print 'ESD-2halo:', esd_2halo[i]
        print
        
    plt.xscale('log')
    plt.yscale('log')
    
    plt.show()
    
    return
    
main()


