"""
Auxiliary functions. This needs to be refactioned to use astropy.cosmology.
This should also take input from the config file.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from astropy.cosmology import FlatLambdaCDM
try:
    from itertools import izip
except ImportError:
    izip = zip
from numpy import array, exp, inf, loadtxt, median, pi
from scipy.stats import rv_discrete


def cM_duffy08(M, z, h=1):
    # critical density
    #return 5.71 / (M/(2e12/h))**0.084 / (1+z)**0.47
    # mean density
    return 10.14 / (M/(2e12/h))**0.089 / (1+z)**1.01

def cM_dutton14(M, z, h=1):
    b = -0.101 + 0.026 * z
    a = 0.520 + (0.905 - 0.520) * exp(-0.617 * z**1.21)
    return 10**a * (M/(1e12/h))**b

def cM_maccio08(M, z, h=1):
    # WMAP5
    return 10**0.830 / (M/(1e12/h))**0.098

def cM_maccio08_WMAP1(M, z, h=1):
    return 10**0.917 / (M/(1e12/h))**0.104

def cM_maccio08_WMAP3(M, z, h=1):
    return 10**0.769 / (M/(1e12/h))**0.083

def delta(c):
    from numpy import log
    return 200./3. * c**3 / (log(1+c) - c/(1.+c))

def density_critical(z, h=1, Om=0.3, Ol=0.7):
    # 3 * H(z)**2 / (8*pi*G), in Msun / Mpc^3
    # H(z) = 100 * h * E(z)
    return 2.7746e11 * h**2 * (Om * (1+z)**3 + Ol)

def density_average(z, h=1, Om=0.3, Ol=0.7):
    # 3 * H_0**2 * Omega_M * (1+z)**3 / (8*pi*G), in Msun/Mpc^3
    return 2.7746e11 * h**2 * Om * (1+z)**3

def r200m(M, z, h=1, Om=0.3, Ol=0.7):
    return (3*M / (800*pi*density_average(z, h, Om, Ol))) ** (1./3)

def sigma_crit(zl, zs):
    b = cosmology.dA(zs) / (cosmology.dA(zl) * cosmology.dA(zl, zs))
    #s = constants.c**2 / (4*scipy.pi*constants.G) * (b/u.Mpc)
    #return s.to(u.Msun/u.pc**2)
    return 1.662454e6 * b

def r3d_from_2d(Rsat, n_Rsat, z, cgroup, Mgroup, n=300000,
                h=1, Om=0.3, Ol=0.7):
    """
    R is an array of projected distances in Mpc
    """
    rho_c = density_average(z, h, Om, Ol)
    aux = 200*rho_c * 4*3.14159265/3
    rsample = 1e3 * numpy.array([rsat_range[rsat_range > Ri] for Ri in Rsat])
    nfw_dist = nfw_profile(rsample, cgroup, Mgroup, aux)
    nfw_dist /= nfw_dist.sum()
    # draw random samples from a given distribution
    dist = rsample[numpy.digitize(numpy.random(n), numpy.cumsum(nfw_dist))]
    #print(dist.shape)
    rsat = dist[numpy.digitize(numpy.random(n), numpy.cumsum(n_Rsat))] / 1000.
    #print(rsat.shape)
    return rsat

def chi2(model, esd, esd_err):
    return (((esd - model) / esd_err) ** 2).sum()

def nfw_profile(r, c, M, aux):
    """
    This internal function is to draw 3D radii from a projected radius.
    aux is the same aux as in, e.g., joint_rtchandra()

    """
    #r200 = (M / aux) ** (1./3.)
    #rs = r200 / c
    #x = r / rs
    x = c * r / (M / aux) ** (1./3.)
    #return dc * rho_c / (x*(1+x)**2)
    return 1 / (x * (1+x)**2)

def gauleg(a, b, n):
    from numpy import arange, cos
    x = arange(n+1) # x[0] unused
    w = arange(n+1) # w[0] unused
    eps = 3.0E-14
    m = (n + 1) / 2
    xm = 0.5 * (b + a)
    xl = 0.5 * (b - a)
    for i in xrange(1, m+1):
        z = cos(3.141592654 * (i-0.25) / (n+0.5))
        while 1:
            p1 = 1.0
            p2 = 0.0
            for j in xrange(1, n+1):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            pp = n * (z * p1 - p2) / (z**2 - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= eps:
                break
        x[i] = xm - xl * z
        x[n+1-i] = xm + xl * z
        w[i] = 2 * xl / ((1 - z**2) * pp**2)
        w[n+1-i] = w[i]
    return x[1:], w1[1:]

def read_covariance(filename):
    #import readfile
    from collections import OrderedDict
    from numpy import array
    # returns unique values
    obsbins = [array(list(OrderedDict.fromkeys(d))) for d in data[:2]]
    rbins = [array(list(OrderedDict.fromkeys(ri))) for ri in data[2:4]]
    Nobsbins = len(obsbins[0])
    Nrbins = len(rbins[0])
    cov = data[4].reshape((Nobsbins,Nobsbins,Nrbins,Nrbins))
    return cov

def read_header(hdr):
    from numpy import array

    paramtypes = ('read', 'uniform',
                  'normal', 'lognormal', 'fixed', 'derived')
    params = []
    prior_types = []
    val1 = []
    val2 = []
    val3 = []
    val4 = []
    exclude_bine = None
    file = open(hdr)
    for line in file:
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'acceptance_fraction':
            print('acceptance_fraction =', array(line[1].split(',')))
        elif line[0] == 'acor':
            print('acor =', line[1])
        elif line[0] == 'nwalkers':
            nwalkers = int(line[1])
        elif line[0] == 'nsteps':
            nsteps = int(line[1])
        elif line[0] == 'nburn':
            nburn = int(line[1])
        elif line[0] == 'datafile':
            datafile = line[1].split(',')
            if len(datafile) == 1:
                datafile = datafile[0]
        elif line[0] == 'cols':
            cols = [int(i) for i in line[1].split(',')]
        elif line[0] == 'covfile':
            covfile = line[1]
        elif line[0] in ('covcol', 'covcols'):
            covcols = [int(i) for i in line[1].split(',')]
        elif line[0] == 'exclude_bins':
            exclude_bins = [int(i) for i in line[1].split(',')]
        elif line[0] == 'model':
            model = line[1]
        elif line[0] == 'metadata':
            meta_names.append(line[1].split(','))
            fits_format.append(line[2].split(','))
        if line[1] in paramtypes:
            params.append(line[0])
            prior_types.append(line[1])
            if line[1] in ('read', 'fixed'):
                v1 = line[2].split(',')
                if len(v1) == 1:
                    val1.append(float(v1[0]))
                else:
                    val1.append(array(v1, dtype=float))
                val2.append(-1)
                val3.append(-1)
                val4.append(-1)
            else:
                val1.append(float(line[2]))
                val2.append(float(line[3]))
                val3.append(float(line[4]))
                val4.append(float(line[5]))
    file.close()
    #out = (array(params), prior_types, sat_profile, group_profile,
    out = (array(params), prior_types,
           array(val1), array(val2), array(val3), array(val4),
           datafile, cols, covfile, covcols, exclude_bins,
           model, nwalkers, nsteps, nburn)
    return out
