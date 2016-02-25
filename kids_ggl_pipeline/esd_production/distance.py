#!/usr/bin/python

"""
This utility function calculates the radial diameter distance 'Da' from the
redshift 'z' for a given matter and lambda density 'Om' and 'Ol'). Multiplying
Da by an opening angle 'theta' on the sky will give its actual size 'x'.

"""

import pyfits
from scipy import integrate
import numpy as np
import sys
import glob
from astropy import constants as const, units as u


def angular(z,Om,Ol):

    # Important constants
    c = const.c.value # Speed of light (in m/s)
    Hdim = h*1e-1 # dimension of H (H = h*100 km/s/Mpc = h*100*1e3m/s/1e6pc = h*0.1 m/s/pc)

    n = 1000.

    integral = 0
    for zp in arange(0.,z,z/n):
        insint = 1/(Om*(1+zp)**3 + Ol)**0.5
        integral = integral + insint
    integral = integral*(z/n)

    Da = (c/Hdim)*1/(1+z)*integral

    return Da

def comoving(z,Om,Ol,h):

    # Important constants
    c = const.c.value # Speed of light (in m/s)
    Hdim = h*1e-1 # dimension of H (H = h*100 km/s/Mpc = h*100*1e3m/s/1e6pc = h*0.1 m/s/pc)

    f = lambda zp: 1/(Om*(1+zp)**3 + Ol)**0.5
    integral = integrate.quad(f,0.,z)#, epsrel=1./n, epsabs=0)

    Dc = (c/Hdim)*integral[0]

    return Dc

def Da_Edo(z_low,z_high, POmega_L=0.73, POmega_M=0.27, POmega_R=0., \
           POmega_K=0., Ph=0.7, Pw=-1., Niter=10000):

    Pc = 299792.458

    r2 = 1. / (1. + z_low)      #   ! Lower limit of integral
    r1 = 1. / (1. + z_high)       # ! Higher limit of integral
    dr = (r2 - r1) / Niter
    r  = r1

    summ = 0.

# Iterate integral
    for i in xrange(Niter):
        r = r + dr

        t1 = POmega_L * r**(1. - (3.*Pw))
        t2 = POmega_M * r
        t3 = POmega_R
        t4 = POmega_K * r**2.

        v1 = 1. / (t1 + t2 + t3 + t4)**0.5

        t1 = POmega_L * (r + dr)**(1. - (3.*Pw))
        t2 = POmega_M * (r + dr)
        t3 = POmega_R
        t4 = POmega_K * (r + dr)**2.

        v2 = 1. / (t1 + t2 + t3 + t4)**0.5

        summ = summ + ( (v1 + v2) / 2. ) * dr #! trapezium rule

    Pt_h=1./100000.
    distance = Pc * summ * ( (1. / (1. + z_high) ) * Pt_h ) / Ph

    return distance * 1e6
