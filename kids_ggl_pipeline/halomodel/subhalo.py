from numpy import arange, array, exp, pi, logspace
from scipy.integrate import trapz

from halo import *
#from nfw import *


def model(theta, R, h=0.7, Om=0.315, Ol=0.685,
          expansion=100, expansion_stars=160, n_bins=10000,
          lnk_min=-13., lnk_max=17.):
    #seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')

    # note that we give log-masses to halo.model()
    z, f, sigma_c, A, M_1, gamma_1, gamma_2, \
        fc_nsat, alpha_s, b_0, b_1, b_2, Ac2s, \
        alpha_star, beta_gas, r_t0, r_c0, \
        Mh_min, Mh_max, Mh_step, Mstar_bin_min, Mstar_bin_max, \
        centrals, satellites, taylor_procedure, include_baryons, \
        smth1, smth2 = theta
    # free parameters (these need to be in theta):
    #     stellar-to-total mass relation: logMo, logM1=10.3, beta, sigma
    #     subhalo mass function: alpha=0.9, beta=0.13 (vdBosch+05)

    # HMF set up parameters
    k_step = (lnk_max-lnk_min) / n_bins
    k_range = arange(lnk_min, lnk_max, k_step)
    k_range_lin = exp(k_range)
    # Halo mass and concentration
    #mass_range = _logspace(M_min, M_max, int((M_max-M_min)/M_step))
    Mh_range = 10**arange(Mh_min, Mh_max, Mh_step)
    ch = Con(z, Mh_range, fh)
    # subhalo masses and concentrations
    Msub_min = 5
    Msub_max = Mh_max
    Msub_step = 0.01
    Msub_range = 10**arange(Msub_min, Mh_max, Msub_step)
    # for now taking the Duffy+ relation as well.
    csub = Con(z, Msub_range, fsub)
    # stellar masses - I don't think I need an HOD for satellite-only
    # (or central-only) samples, do I? I'm just taking the mid point in the
    # bin here. Note that if this really is the case then it's better to
    # provide the adopted value per bin in the config file (e.g., the
    # median logMstar in each bin), but let's keep it like this for now.
    logMstar = (Mstar_bin_min + Mstar_bin_max) / 2
    n_bins_obs = Mstar_bin_min.size
    print Msub_range.shape, Mh_range.shape, logMstar.shape

    # average Msub given Mh
    Msub_Mh = trapz([[Msub_range * nsub(Msub_range, Mh, a, b) * \
                      p_nsub(Msub_range, logMstar_bin,
                             logMo, logM1, beta, sigma)
                      for Mh in Mh_range] for logMstar_bin in logMstar],
                    axis=2)

    hmfparams = {'sigma_8': sigma8, 'H0': 100*h,'omegab_h2': Obh2,
                 'omegam': Om, 'omegav': Ov, 'n': ns,
                 'lnk_min': -12 ,'lnk_max': 12,
                 'dlnk': 0.01, 'transfer_fit': 'BBKS', 'z': z,
                 'force_flat': True}
    hmf = Mass_Function(Mh_min, Mh_max, Mh_step, 'Tinker10', **hmfparams)
    nh = hmf.dndm

    # average Msub accounting for the HMF
    print Msub_Mh.shape, nh.shape
    Msub_avg = trapz(Msub_Mh * nh, Mh_range, axis=1)
    print 'Msub =', Msub, Msub_avg.shape

    # average ESD given Mh
    # How do I include n(Rsat) here?

    return


def nsub(Msub, Mh, a=0.9, b=0.13):
    """
    subhalo mass function of vdBosch+05. Missing the normalization.

    """
    return (b / (Msub/Mh))**a * exp(-(Msub/Mh) / b) / (Msub/Mh)

def p_nsub(Msub, logMstar, logMo, logM1, beta, sigma):
    """
    stellar-to-subhalo mass relation; mu and sigma are the mean and scatter
    We assume a constant scatter for now, but can easily be generalized to
    depend on mass

    """
    mu = logMo + beta * (logMstar-logM1)
    return exp((log10(Msub) - mu)**2 / (2*sigma**2)) / ((2*pi)**0.5*sigma)

