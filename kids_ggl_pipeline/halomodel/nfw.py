"""
Excess surface densities of NFW profiles with and without truncation (see
Wright & Brainerd, 2000; Baltz et al., 2009; Pastor Mira et al., 2011).

WARNING: Masses are not yet implemented

Functions
---------
    Call the following functions to get the different ESDs
        esd(x, sigma_s) -- Regular NFW
        esd_offset(x, xoff, sigma_s)
        esd_trunc5(x, tau, sigma_s) -- NFW drop3.14159265ng as 1/r^5 outside r_t
        esd_trunc7(x, tau, sigma_s) -- NFW drop3.14159265ng as 1/r^7 outside r_t
        esd_sharp(x, tau, sigma_s) -- NFW sharply cut at r_t

    There are also functions to get the mass as a function of radius
        mass(x, sigma_s)
        mass_trunc5(x, tau, sigma_s)
        mass_trunc7(x, tau, sigma_s)
        mass_sharp(x, tau, sigma_s)

    And the enclosed masses are returned by
        mass_enclosed(x, sigma_s)
        mass_enclosed_trunc5(x, tau, sigma_s)
        mass_enclosed_trunc7(x, tau, sigma_s)
        mass_enclosed_sharp(x, tau, sigma_s)

    The surface densities can be obtained by calling
        sigma(x, sigma_s)
        sigma_trunc5(x, tau, sigma_s)
        sigma_trunc7(x, tau, sigma_s)
        sigma_sharp(x, tau, sigma_s)

    The enclosed surface densities can be obtained by calling
        barsigma(x, sigma_s)
        barsigma_trunc5(x, tau, sigma_s)
        barsigma_trunc7(x, tau, sigma_s)
        barsigma_sharp(x, tau, sigma_s)

Parameters
----------
    Functions take the following parameters:
        x       : numpy array of floats
                  Distances in units of the scale radius
        rt      : float
                  Truncation radius, in units of the scale radius (only for
                  functions nfw_trunc5, nfw_trunc7, nfw_sharp)
        sigma_s : float
                  Characteristic surface mass density, r_s * delta_c * rho_c,
                  in Msun/pc^2

Notes
-----
    -Remember that the "sharp" profile is only defined up to R=r_t!

"""
#import numpy
from itertools import izip
from numpy import arctan, arctanh, arccos, array, cos, \
                  hypot, log, log10, logspace, \
                  meshgrid, ones, sum as nsum, transpose, zeros
from scipy.integrate import cumtrapz, trapz
from scipy.interpolate import interp1d

#-----------------------------------#
#--    Excess surface densities
#-----------------------------------#

def esd(x, sigma_s):
    """
    Regular NFW.
    Eqs. 1, 2, 11, 13 of Wright & Brainerd (2000).

    See module help for parameters

    """
    return barsigma(x, sigma_s) - sigma(x, sigma_s)

def esd_trunc5(x, tau, sigma_s):
    """
    NFW that drops as 1/r^5 at large radius.
    Eqs. A3-A11 of Baltz et al. (2009), or
    Eqs. 18-21 of Pastor Mira et al. (2011)

    See module help for parameters

    """
    return barsigma_trunc5(x, tau, sigma_s) - sigma_trunc5(x, tau, sigma_s)

def esd_trunc7(x, tau, sigma_s):
    """
    NFW that drops as 1/r^7 at large radius.
    Eqs. A26-A29 of Baltz et al. (2009)

    See module help for parameters

    """
    return barsigma_trunc7(x, tau, sigma_s) - sigma_trunc7(x, tau, sigma_s)

def esd_sharp(x, tau, sigma_s):
    """
    NFW sharply truncated at the truncation radius.
    Eqs. A34-A37 of Baltz et al. (2009)

    See module help for parameters

    """
    return barsigma_sharp(x, tau, sigma_s) - sigma_sharp(x, tau, sigma_s)

#-----------------------------------#
#--    Offset ESDs
#-----------------------------------#

"""
Note that this implementation requires that the group distribution is given.
Will later try to

"""

def esd_offset(x, xoff, n, sigma_s, x_range, angles, interp_kind='slinear'):
    """
    Remember that the first value of x *must* be zero for cumtrapz to
    give the right result; this value is discarded in the output. This is
    because we never expect to have a data point at x=0 in real observations,
    so this datum is added automatically in my emcee wrapper

    x_range are the samples for trapezoidal integration.

    """
    s = sigma_distribution(xoff, n, sigma_s, x_range, angles, interp_kind)
    s_at_x = s(x[1:]) / (2.0*3.14159265)
    s_within_x = cumtrapz(sigma_within(s, x), x, initial=0)
    return 2.0 * array(s_within_x)[1:] / x[1:]**2.0 - s_at_x

def sigma_azimuthal(angles, x_range, xoff, sigma_s):
    xo = (xoff**2.0 + x_range**2.0 + 2.0*x_range*xoff*cos(angles)) ** 0.5
    return sigma(xo, sigma_s)

def sigma_distribution(xoff, n, sigma_s, x_range, angles,
                       interp_kind='slinear'):
    s = sigma_integrate(x_range, xoff, sigma_s, angles)
    # weight the contribution of each radial bin
    sarray = [ni * si(x_range) for ni, si in izip(n, s)]
    s = nsum(sarray, axis=0) / n.sum()
    # go back to the required format
    s = interp1d(x_range, s, kind=interp_kind)
    return s

def sigma_integrate(x_range, xoff, sigma_s, angles):
    x_range2d, angles2d = meshgrid(x_range, angles)
    y = [sigma_azimuthal(angles2d, x_range2d, x, sigma_s) for x in xoff]
    y = transpose(y, axes=(0,2,1))
    itable = trapz(y, angles, axis=2)
    s = [interp1d(x_range, i, kind='slinear') for i in itable]
    return s

def sigma_offset(x, xoff, n, sigma_s, x_range, angles, interp_kind='slinear'):
    s = sigma_distribution(xoff, n, sigma_s, x_range, angles, interp_kind)
    return s(x[1:]) / (2.0*3.14159265)

def sigma_within(s, x):
    return s(x) * x / (2.0*3.14159265)

#-----------------------------------#
#--    Surface densities
#-----------------------------------#

def sigma(x, sigma_s):
    s = ones(x.shape)
    s[x == 0] = 0
    s[x == 1] = 1 / 3.
    j = (x > 0) & (x < 1)
    s[j] = arctanh(((1 - x[j]) / (1 + x[j]))**0.5)
    s[j] = (1 - 2.*s[j] / (1.-x[j]**2)**0.5) / (x[j]**2 - 1)
    j = x > 1
    s[j] = arctan(((x[j] - 1) / (1 + x[j]))**0.5)
    s[j] = (1 - 2.*s[j] / (x[j]**2 - 1)**0.5) / (x[j]**2 - 1.)
    return 2 * sigma_s * s / 1e12

def sigma_trunc5(x, tau, sigma_s):
    s = (tau**2 + 1) / (x**2 - 1) * (1 - F(x))
    s += 2*F(x) - 3.14159265 / (tau**2 + x**2)**0.5
    s += (tau**2 - 1) / (tau * (tau**2 + x**2)**0.5) * L(x, tau)
    return 2 * sigma_s * tau**2 / (tau**2 + 1)**2 * s / 1e12

def sigma_trunc7(x, tau, sigma_s):
    xtau2 = tau**2 + x**2
    s = 2 * (tau**2 + 1) / (x**2 - 1) * (1 - F(x)) + 8*F(x)
    s += (tau**4 - 1) / (tau**2 * xtau2)
    s -= 3.14159265 * (4 * xtau2 + tau**2 + 1) / xtau2**1.5
    s += (tau**2 * (tau**4 - 1) + xtau2 * (3*tau**4 - 6*tau**2 - 1)) * \
         L(x, tau) / (tau**3 * xtau2**1.5)
    return tau**4 * sigma_s / (tau**2 + 1)**3 * s / 1e12

def sigma_sharp(x, tau, sigma_s):
    mask = x < tau
    xmask = x[mask]
    s = zeros(x.shape)
    s[mask] = (tau**2-xmask**2)**0.5 / (1.+tau) - T(xmask, tau)
    return 2 * sigma_s / (x**2 - 1) * s.real / 1e12

#-----------------------------------#
#--    Enclosed surface densities
#-----------------------------------#

def barsigma(x, sigma_s):
    s = ones(x.shape)
    s[x == 0] = 0
    s[x == 1] = 1 + log(0.5)
    j = (x > 0) & (x < 1)
    s[j] = arctanh(((1 - x[j])/(1 + x[j]))**0.5)
    s[j] = 2 * s[j] / (1 - x[j]**2)**0.5
    s[j] = (s[j] + log(0.5*x[j])) / x[j]**2
    j = x > 1
    s[j] = arctan(((x[j] - 1)/(1 + x[j]))**0.5)
    s[j] = 2 * s[j] / (x[j]**2 - 1)**0.5
    s[j] = (s[j] + log(0.5*x[j])) / x[j]**2
    return 4 * sigma_s * s / 1e12

def barsigma_trunc5(x, tau, sigma_s):
    s = hypot(tau, x) * (-3.14159265 + (tau**2 - 1) / tau * L(x, tau))
    s += tau * 3.14159265 + (tau**2 - 1) * log(tau)
    s += (tau**2 + 1 + 2 * (x**2 - 1)) * F(x)
    s *= (4 * sigma_s / x**2) * (tau**2 / (tau**2 + 1)**2)
    return s / 1e12

def barsigma_trunc7(x, tau, sigma_s):
    xtau2 = tau**2 + x**2
    s1 = 2 * (tau**2 + 1 + 4*(x**2-1)) * F(x)
    s1 += (3.14159265 * (3*tau**2-1) + 2 * tau * (tau**2-3) * \
           log(tau)) / tau
    s2 = - tau**3 * 3.14159265 * (4 * xtau2 - tau**2 - 1)
    s2  -= (tau**2 * (tau**4-1) - xtau2 * (3*tau**4 - 6*tau**2 - 1)) * \
           L(x, tau)
    s2 /= tau**3 * hypot(x, tau)
    s = 2 * sigma_s * tau**4 / (x**2 * (tau**2 + 1)**3) * (s1 + s2)
    return s / 1e12

def barsigma_sharp(x, tau, sigma_s):
    x = x + 0j
    s = log(1.+tau) - (tau - (tau**2-x**2)**0.5) / (1.+tau)
    s -= arctanh((tau**2 - x**2)**0.5 / tau) - T(x, tau)
    s *= 4 * sigma_s / x**2
    return s.real / 1e12

#-----------------------------------#
#--    Enclosed projected masses
#-----------------------------------#

def mproj_enclosed(x, rs, sigma_s):
    """
    Regular NFW.
    Eqs. 1, 2, 11, 13 of Wright & Brainerd (2000).

    See module help for parameters

    """
    return 2*3.14159265 * (rs*x)**2 * (1e12 * barsigma(x, sigma_s))

def mproj_enclosed_trunc5(x, rs, tau, sigma_s):
    """
    NFW that drops as 1/r^5 at large radius.
    Eqs. A3-A11 of Baltz et al. (2009), or
    Eqs. 18-21 of Pastor Mira et al. (2011)

    See module help for parameters

    """
    return 2*3.14159265 * (rs*x)**2 * (1e12 * barsigma_trunc5(x, tau, sigma_s))

def mproj_enclosed_trunc7(x, rs, tau, sigma_s):
    """
    NFW that drops as 1/r^7 at large radius.
    Eqs. A26-A29 of Baltz et al. (2009)

    See module help for parameters

    """
    return 2*3.14159265 * (rs*x)**2 * (1e12 * barsigma_trunc7(x, tau, sigma_s))

def mproj_enclosed_sharp(x, rs, tau, sigma_s):
    """
    NFW sharply truncated at the truncation radius.
    Eqs. A34-A37 of Baltz et al. (2009)

    See module help for parameters

    """
    return 2*3.14159265 * (rs*x)**2 * (1e12 * barsigma_sharp(x, tau, sigma_s))

#-----------------------------------#
#--    Enclosed 3D masses
#-----------------------------------#

def mass_enclosed(x, rs, sigma_s):
    """
    Regular NFW.
    Eqs. 1, 2, 11, 13 of Wright & Brainerd (2000).

    See module help for parameters

    """
    return 4 * 3.14159265 * rs**2 * sigma_s * (log(1.+x) - x / (1.+x))

def mass_enclosed_trunc5(x, rs, tau, sigma_s):
    """
    NFW that drops as 1/r^5 at large radius.
    Eqs. A3-A11 of Baltz et al. (2009), or
    Eqs. 18-21 of Pastor Mira et al. (2011)

    See module help for parameters

    """
    f = lambda X: ((2 * (1+tau**2)) / (1.+X) + 4 * tau * arctan(X/tau) + \
                   2 * (-1+tau**2) * log(1.+X) - \
                   (-1 + tau**2) * log(tau**2+X**2)) / (2 * (1.+tau**2)**2)
    return 4 * 3.14159265 * rs**2 * sigma_s * tau**2 * (f(x) - f(0))

def mass_enclosed_trunc7(x, rs, tau, sigma_s):
    """
    NFW that drops as 1/r^7 at large radius.
    Eqs. A26-A29 of Baltz et al. (2009)

    See module help for parameters

    """
    f = lambda X: ((tau**2 + 1) * (tau**2 + 2*X - 1) / (tau**2 + X**2) - \
                   (tau**2 - 3) * log(tau**2 + X**2) + \
                   2 * (tau**2 + 1) / (X + 1) + \
                   2 * (tau**2 - 3) * log(1. + X) + \
                   2 * (3*tau**2 - 1) * arctan(X / tau) / tau) / \
                  (2 * (tau**2 + 1)**3)
    return 4 * 3.14159265 * rs**2 * sigma_s * tau**4 * (f(x) - f(0))

def mass_enclosed_sharp(x, rs, tau, sigma_s):
    """
    NFW sharply truncated at the truncation radius.
    Eqs. A34-A37 of Baltz et al. (2009)

    See module help for parameters

    """
    return 4 * 3.14159265 * rs**2 * sigma_s * (log(1.+x) - x / (1.+x))

#-----------------------------------#
#--    Total  masses
#-----------------------------------#

def mass_total(rs, c, sigma_s):
    return 4 * 3.14159265 * rs**2 * sigma_s * (log(1.+c) - c / (1.+c))

def mass_total_trunc5(rs, tau, sigma_s):
    """
    NFW that drops as 1/r^5 at large radius.
    Eqs. A3-A11 of Baltz et al. (2009), or
    Eqs. 18-21 of Pastor Mira et al. (2011)

    See module help for parameters

    """
    m = 4 * 3.14159265 * rs**2 * sigma_s * tau**2 / (tau**2 + 1)**2
    return m * ((tau**2 - 1) * log(tau) + tau*3.14159265 - (tau**2 + 1))

def mass_total_trunc7(rs, tau, sigma_s):
    """
    NFW that drops as 1/r^7 at large radius.
    Eqs. A26-A29 of Baltz et al. (2009)

    See module help for parameters

    """
    m = 4 * 3.14159265358 * rs**2 * sigma_s * tau**2 / (2 * (tau**2 + 1)**3)
    return m * (2 * tau**2 * (tau**2 - 3) * log(tau) - \
                (3*tau**2 - 1) * (tau**2 + 1 - tau*3.14159265))

def mass_total_sharp(rs, tau, sigma_s):
    """
    NFW sharply truncated at the truncation radius.
    Eqs. A34-A37 of Baltz et al. (2009)

    See module help for parameters

    """
    return 4 * 3.14159265 * rs**2 * sigma_s * (log(1.+tau) - tau / (1.+tau))

#-----------------------------------#
#--    3-dimensional density profiles
#-----------------------------------#

def rho(x, delta_c, rho_bg):
    return delta_c * rho_bg / (x*(1+x)**2)

def rho_trunc5(x, tau, delta_c, rho_bg):
    return delta_c * rho_bg / (x*(1+x)**2) * (tau**2 / (tau**2+x**2))

def rho_trunc7(x, tau, delta_c, rho_bg):
    return delta_c * rho_bg / (x*(1+x)**2) * (tau**2 / (tau**2+x**2))**2

def rho_sharp(x, tau, delta_c, rho_bg):
    y = zeros(x.size)
    y[x < tau] = delta_c * rho_bg / (x[x < tau]*(1+x[x < tau])**2)
    return y

#-----------------------------------#
#--    Auxiliary functions
#-----------------------------------#

def F(x):
    x = array(x)
    f = ones(x.shape)
    f[x < 1] = log(1./x[x < 1] + (1/x[x < 1]**2 - 1)**0.5)
    f[x < 1] /= (1-x[x < 1]**2)**0.5
    f[x > 1] = arccos(1./x[x > 1]) / (x[x > 1]**2 - 1)**0.5
    return f

def L(x, tau):
    return log(x / ((tau**2+x**2)**0.5 + tau))

def T(x, tau):
    x = x + 0j
    tx2diff = (tau**2 - x**2)**0.5
    x2diff = (x**2 - 1)**0.5
    t = arctan(tx2diff / x2diff) - arctan(tx2diff / (tau*x2diff))
    t /= x2diff
    t[x == 1] = ((tau-1) / (tau+1))**0.5
    # the imaginary parts of the two arctans cancel out
    return t

def Sigma_s(rs, delta_c, rho_c):
    return rs * delta_c * rho_c

def delta(c):
    return 200. * c**3 / (3 * (log(1.+c) - c / (1.+c)))
