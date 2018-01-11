import numpy as np
from numpy import array, exp, log, log10, pi
from scipy.integrate import simps, trapz
import scipy.special as sp

from tools import Integrate

"""
# Population functions - average number of galaxies 
# (central or satelites or all) in a halo of mass M - from HOD or CLF!
"""


"""
# Zheng 2005
"""

#~ def ncm(mass_func, m, M):
    #~
    #~ M_min = 11.6222
    #~ M_1 = 12.851
    #~ alpha = 1.049
    #~ M_0 = 11.5047
    #~ sig_logm = 0.26
    #~
    #~ nc = 0.5 * (1 + sp.erf((np.log10(M) - M_min) / sig_logm))
    #~
    #~ return nc
    #~
    #~
#~ def nsm(mass_func, m, M):
    #~
    #~ M_min = 11.6222
    #~ M_1 = 12.851
    #~ alpha = 1.049
    #~ M_0 = 11.5047
    #~ sig_logm = 0.26
    #~
    #~ ns = np.zeros_like(M)
    #~ ns[M > 10 ** M_0] = ((M[M > 10 ** M_0] - 10 ** M_0) / 10 ** M_1) ** alpha
    #~
    #~ return ns
    #~
    #~
#~ def ngm(mass_func, m, M):
    #~
    #~ ng = ncm(mass_func, m, M) + nsm(mass_func, m, M)
    #~
    #~ return ng


"""
# Conditional mass function derived
"""

def phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2):
    # Conditional stellar mass function - centrals
    # m - stellar mass, M - halo mass
    # FROM OWLS HOD FIT: sigma = 4.192393813649759049e-01
    #sigma = 0.125
    if np.iterable(M):
        phi = np.zeros((M.size,m.size))
    else:
        M = np.array([M])
        phi = np.zeros((1,m.size))
    Mo = m_0(M, A, M_1, gamma_1, gamma_2)
    for i in xrange(M.size):
        phi[i] = np.exp(-((np.log10(m)-np.log10(Mo[i]))**2.0) / (2.0*(sigma**2.0))) / ((2.0*pi)**0.5 * sigma * m * np.log(10.0))
    return phi


def phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, Ac2s):
    # Conditional stellar mass function - satellites
    # m - stellar mass, M - halo mass
    #alpha = -2.060096789583814925e+00
    if np.iterable(M):
        phi = np.zeros((M.size,m.size))
    else:
        M = np.array([M])
        phi = np.zeros((1,m.size))
    Mo = Ac2s * m_0(M, A, M_1, gamma_1, gamma_2)
    for i in xrange(M.size):
        phi[i] = phi_0(M[i], b_0, b_1, b_2) * ((m/Mo[i])**(alpha + 1.0)) * exp(-(m/Mo[i])**2.0) / m
    return phi


def phi_t(m, M, sigma, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, Ac2s):
    # Sum of phi_c and phi_s
    # m - stellar mass, M - halo mass
    phi = phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2) + \
          phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, Ac2s)
    return phi


def phi_i(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
          b_0, b_1, b_2):
    # Integrated phi_t!
    # m - stellar mass, M - halo mass

    phi = np.ones(m.size)
    phi_int = phi_t(m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
                    b_0, b_1, b_2).T
    for i in xrange(m.size):
        integ = phi_int[i] * mass_func.dndm
        phi[i] = Integrate(integ, M)
    return phi.T


def av_cen(m, M, sigma, A, M_1, gamma_1, gamma_2):

    if not np.iterable(M):
        M = np.array([M])

    phi = np.ones(M.size)

    phi_int = phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2) #Centrals!
    for i in xrange(M.size):
        #integ = phi_int[i,:]*m
        #phi[i] = Integrate(integ, m)
        phi[i] = trapz(phi_int[i] * m, m)

    return phi


def av_sat(m, M, alpha, A, M_1, gamma_1, gamma_2, b_0, b_1, b_2, Ac2s):
    #if not np.iterable(M):
        #M = np.array([M])
    phi = np.ones(M.size)
    phi_int = phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2,
                    b_0, b_1, b_2, Ac2s)
    for i in xrange(M.size):
        #integ = phi_int[i,:]*m
        #phi[i] = Integrate(integ, m)
        phi[i] = trapz(phi_int[i] * m, m)
    return phi


def m_0(M, A, M_1, gamma_1, gamma_2):
    """
    Stellar mass as a function of halo mass

    """
    return 10.0**A * ((M/10.0**M_1)**(gamma_1)) / \
          ((1.0 + (M/10.0**M_1))**(gamma_1 - gamma_2))


def phi_0(M, b_0, b_1, b_2):
    # Functional form for phi_0 - taken directly from Cacciato 2009
    # Fit as a result in my thesis!

    #b_0 = -5.137787703823422092e-01
    #b_1 = 7.478552629547742525e-02
    #b_2 = -7.938925982752477462e-02
    M12 = M/(10.0**12.0)

    log_phi = b_0 + b_1*np.log10(M12) + b_2*(np.log10(M12))**2.0

    phi = 10.0**log_phi

    return phi


def ncm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
        b_0, b_1, b_2):
    nc = np.ones(M.size)
    phi_int = phi_c(m, M, sigma, A, M_1, gamma_1, gamma_2)
    for i in xrange(M.size):
        nc[i] = Integrate(phi_int[i], m)
    # This works, but above more general, for different definition of
    # CLF/CMF.
    #func = lambda x: 0.5 * (sp.erf((np.log10(x/m_0(M, A, M_1, gamma_1, gamma_2))) / (sigma * (2.0**0.5))))
    #nc = (func(m[-1]) - func(m[0]))
    return nc


def nsm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
        b_0, b_1, b_2, Ac2s):
    ns = np.ones(M.size)
    phi_int = phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2,
                    b_0, b_1, b_2, Ac2s)
    for i in xrange(M.size):
        ns[i] = Integrate(phi_int[i], m)
    #p = phi_0(M, b_0, b_1, b_2)
    #y = 0.562 * m_0(M, A, M_1, gamma_1, gamma_2)

    # This does not work yet, as gamma function is not propery
    # defined in scipy. It needs some additional lines...
    #func = lambda x: (-0.5) * p * ((x/y)**(alpha-1.0)) * \
                     #((x/y)**(1.0-alpha)) * \
                     #(sp.gammaincc(np.abs((alpha+1.0)/2.0),((x/y)**2.0))) * \
                     #sp.gamma(np.abs((alpha+1.0)/2.0))
    #ns = func(m[-1]) - func(m[0])
    return ns


def ncm_simple(mass_func, M, M_1, sigma):
    
    M_1 = 10.0**M_1
    #"""
    nc = exp(-(log10(M)-log10(M_1))**2.0 / (2.0*(sigma**2.0))) / \
        ((2.0*pi)**0.5 * sigma * log(10))
    #"""
    #nc = 0.5 * (1.0+sp.erf((log10(M)-log10(M_1))/sigma))
    
    return nc


def nsm_simple(mass_func, M, M_1, sigma, alpha):
    M_sat = 0.5*M_1
    ns = np.where(M >= M_1, ((M - M_1)/M_sat)**alpha, 0.0)
    
    return ns



def ngm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
        b_0, b_1, b_2):
    ng = ncm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
             b_0, b_1, b_2) + \
         nsm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
             b_0, b_1, b_2)
    return ng

if __name__ == '__main__':
    main()
