import numpy
import pylab
from numpy import arange, array, exp, iterable, log, log10, logspace, \
                  newaxis, ones, pi, squeeze, transpose
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.special import gammainc

from dark_matter import *
from halo import *


def model(theta, R, h=0.7, Om=0.315, Ol=0.685,
          expansion=100, expansion_stars=160, n_bins=10000,
          lnk_min=-13., lnk_max=17.):
    """
    parameters in theta are:
        sigma8, h, Om, Obh2, ns : cosmological parameters
        z  : redshift
        fh  : normalization of the c(M) relation for the host halos
        logMo  : normalization of the SSHMR
        logM1  : pivot log-mass of the SSHMR
        beta_sshmr  : slope of the SSHMR
        sigma_sshmr  : scatter in the SSHMR
        fs  : subhalo mass fraction
        alpha_shmf  : low-mass-end slope of the SHMF
        beta_shmf  : normalization of the exponential cut-off of the SHMF
        omega_shmf  : high-mass exponential slope of the SHMF
        a_cMsub  : normalization of the subhalo c(M) relation
        b_cMsub  : slope of the subhalo c(M)
        g_cMsub  : slope of the redshift dependence of the subhalo c(M)
        Mo_cMsub  : pivot mass of the subhalo c(M)
        logMh_min  : minimum host log-mass (to integrate the HMF)
        logMh_max  : maximum host log-mass (to integrate the HMF)
        logMh_step  : resolution of the logMh range
        psi_min  : minimum Msub/Mh to integrate the SHMF
        psi_max  : maximum Msub/Mh to integrate the SHMF
        psi_step  : resolution of the Msub/Mh range
        Mstar  : mean stellar masses (or luminosities) of each bin
        lnk_min  : minimum lnk for the power spectrum
        lnk_max  : maximum lnk for the power spectrum
        lnk_step  : lnk step for the power spectrum sampling

    Missing ingredients:
        -Offset central contribution
        -Observed distribution of Rsat (now using an NFW). The problem with
         this is that the observed distribution will not have an analytical
         FT. Could think of fitting an NFW to the distributions in
         lensing_signal.py, from which I can then read the parameters and
         pass here.
        -Calculate the subhalo mass within the radius containing the
         average background density

    NOTE
        -Need to account for the different limiting Msub for each Mh
         (namely, Mh itself!)

    """
    #seterr(divide='ignore', over='ignore', under='ignore', invalid='ignore')

    sigma8, h, Om, Obh2, ns, \
        z, fh, fc_nsat, logMo, logM1, beta_sshmr, sigma_sshmr, \
        fs, alpha_shmf, beta_shmf, omega_shmf, \
        a_cMsub, b_cMsub, g_cMsub, Mo_cMsub, \
        logMh_min, logMh_max, logMh_step, \
        logpsi_min, logpsi_max, logpsi_step, logMstar_min, logMstar_max, \
        lnk_min, lnk_max, lnk_step, \
        smth1, smth2 = theta
    # for now. Remember that I need to read logmstar, which is currently not
    # stored by lensing_signal.py
    logMstar_min = log10(logMstar_min)
    logMstar_max = log10(logMstar_max)

    # if we want to fit for a relation instead of individual masses
    # then logMo should be a scalar.
    if not iterable(logMo):
        logMo = logMo * ones(logMstar_min.size)
    print 'logMo =', logMo

    # HMF set up parameters
    #k_step = (lnk_max-lnk_min) / n_bins
    k_range = arange(lnk_min, lnk_max, lnk_step)
    k_range_lin = exp(k_range)
    # Halo mass and concentration
    Mh_range = 10**arange(logMh_min, logMh_max, logMh_step)

    # subhalo masses and concentrations
    # here I need to change the Msub range depending on Mh
    psi_range = 10**arange(logpsi_min, logpsi_max, logpsi_step)
    Msub_range = array([psi * Mh_range for psi in psi_range])
    print 'psi =', psi_range.shape
    print 'Msub_range =', Msub_range.shape
    print 'Mh_range =', Mh_range.shape
    print 'logMstar =', (logMstar_min+logMstar_max)/2, logMstar_min.shape

    mstar_hod = array([logspace(Mlo, Mhi, 100, endpoint=False,
                                dtype=numpy.longdouble)
                       for Mlo, Mhi in izip(logMstar_min, logMstar_max)])

    hmfparams = {'sigma_8': sigma8, 'H0': 100.*h,'omegab_h2': Obh2,
                 'omegam': Om, 'omegav': 1-Om, 'n': ns,
                 'lnk_min': lnk_min,'lnk_max': lnk_max,
                 'dlnk': lnk_step, 'transfer_fit': 'BBKS', 'z': z,
                 'force_flat': True}
    hmf = Mass_Function(logMh_min, logMh_max, logMh_step,
                        'Tinker10', **hmfparams)
    #omegab = hmf.omegab
    #omegac = hmf.omegac
    nh = hmf.dndm
    #mass_func = hmf.dndlnm
    rho_mean = hmf.mean_dens_z
    rho_crit = rho_mean / (hmf.omegac+hmf.omegab)
    rho_dm = rho_mean * baryons.f_dm(hmf.omegab, hmf.omegac)
    # subhalo mass function
    # I could also set a very low psi_res (default 1e-4) and multiply this
    # by an error function to account for incompleteness
    # need the newaxis every time I call it because I always use it
    # per Mh bin and then integrate over Mh
    #shmf = nsub_vdB05(Msub_range, Mh, a_shmf, b_shmf)[:,newaxis]
    #shmf = nsub(psi_range, alpha_shmf, beta_shmf,
                #omega_shmf, fs)[:,newaxis]
    shmf = nsub(Msub_range/Mh_range, alpha_shmf, beta_shmf,
                omega_shmf, fs)
    print 'shmf =', shmf.shape

    # remove all hard-coded numbers here
    rvirh_range_lin = virial_radius(Mh_range, rho_mean, 200.0)
    rvirh_range = log10(rvirh_range_lin)
    rvirh_range_3d = logspace(-3.2, 4, 200, endpoint=True)
    rvirh_range_3d_i = logspace(-2.5, 1.2, 25, endpoint=True)
    rvirh_range_2d_i = R[0][1:]
    rvirs_range_lin = virial_radius(Msub_range, rho_mean, 200.0)
    rvirs_range = np.log10(rvirs_range_lin)
    rvirs_range_3d = logspace(-3.2, 4, 200, endpoint=True)
    rvirs_range_3d_i = logspace(-2.5, 1.2, 25, endpoint=True)
    rvirs_range_2d_i = R[0][1:]

    # HOD
    print 'mstar_hod =', mstar_hod.shape
    phi = array([phi_sub(log10(Msub_range), logmstar, logMi, logM1,
                         beta_sshmr, sigma_sshmr)
                 for logmstar, logMi in izip(log10(mstar_hod), logMo)])
    print 'phi =', phi.shape
    #"""
    # number of subhalos in each observable bin
    Nsub_obs_Mh = array([trapz(phi_i, mstar, axis=0)
                         for phi_i, mstar in izip(phi, mstar_hod)])
    #print 'Nsub_obs_Mh_Msub =', Nsub_obs_Mh_Msub.shape
    #Nsub_obs_Mh = array([trapz(Nsub_obs_Mh_Msub_i * shmf, axis=0)
                         #for Nsub_obs_Mh_Msub_i in Nsub_obs_Mh_Msub])
    print 'Nsub_obs_Mh =', Nsub_obs_Mh.shape
    nsub_bar_Mh = array([trapz(shmf * Nsub_obs_Mh_i, Msub_range, axis=0)
                         for Nsub_obs_Mh_i in Nsub_obs_Mh])
    print 'nsub_bar_Mh =', nsub_bar_Mh.shape
    Msub_Mh = trapz(Msub_range * shmf * Nsub_obs_Mh,
                    Msub_range[newaxis], axis=1) / nsub_bar_Mh
    # average Msub accounting for the HMF.
    print 'Msub_Mh =', Msub_Mh.shape
    print 'nh =', nh.shape
    Msub_avg = trapz(Msub_Mh * nh, Mh_range, axis=1) / trapz(nh, Mh_range)
    print 'Msub_avg =', Msub_avg, Msub_avg.shape
    Nsub_obs = trapz(Nsub_obs_Mh * nh, Mh_range, axis=2)
    print 'Nsub_obs =', Nsub_obs[:,0], Nsub_obs[:,-1], Nsub_obs.shape
    nsub_bar = trapz(nsub_bar_Mh * nh, Mh_range, axis=1)
    print 'nsub_bar =', nsub_bar, nsub_bar.shape
    #"""

    # 2nd try
    """
    Nsub_Mh_Msub = array([trapz(phi_i, mstar, axis=0)
                          for phi_i, mstar in izip(phi, mstar_hod)])
    print 'Nsub_Mh_Msub =', Nsub_Mh_Msub.shape
    nsub_Mh = array([trapz(Nsub_Mh_Msub_i * shmf, Msub_range, axis=0)
                     for Nsub_Mh_Msub_i in Nsub_Mh_Msub])
    print 'nsub_Mh =', nsub_Mh.shape
    nsub_bar = array([trapz(nsub_Mh_i * nh, Mh_range)
                      for nsub_Mh_i in nsub_Mh])
    print 'nsub_bar =', nsub_bar.shape, nsub_bar
    # this one isn't normalized by the average number of subhalos
    Msub_Mh = array([trapz(Msub_range * shmf * Nsub_Mh_Msub_i, axis=0)
                     for Nsub_Mh_Msub_i in Nsub_Mh_Msub]) / nsub_Mh
    print 'Msub_Mh =', Msub_Mh.shape
    Msub = array([trapz(Msub_Mh_i * nh, Mh_range)
                  for Msub_Mh_i in Msub_Mh]) / trapz(nh, Mh_range)
    print 'Msub =', Msub.shape, Msub
    exit()
    """


    # damping of the 1h power spectra at small k
    F_k1 = f_k(k_range_lin)
    # Fourier Transform of the host NFW profile
    ch = Con(z, Mh_range, fh)
    uk = NFW_f(z, rho_dm, fh, Mh_range, rvirh_range_lin, k_range_lin, ch)
    uk = uk/uk[0]
    print 'uk =', uk.shape
    # FT of the subhalo NFW profile
    csub = cM(z, Msub_range, a_cMsub, b_cMsub, g_cMsub)
    axes = (1,0,2)
    uk_s = transpose([NFW_f(z, rho_dm, 0, Msub_i, rvirs, k_range_lin, c)
                      for Msub_i, rvirs, c
                      in izip(Msub_range, rvirs_range_lin, csub)],
                     axes=axes)
    uk_s = uk_s / uk_s[0]
    print 'uk_s =', uk_s.shape, axes
    # and of the NFW profile of the satellites. Ideally this would be
    # the actual Rsat distribution measured from the data, but this doesn't
    # have an analytical FT. Therefore maybe fitting an NFW to the observed
    # distribution (in lensing_signal.py) would work, but only when I don't
    # select on Rsat.
    uk_Rsat = NFW_f(z, rho_dm, fc_nsat, Mh_range, rvirh_range_lin,
                    k_range_lin)
    uk_Rsat = uk_Rsat / uk_Rsat[0]
    print 'uk_Rsat =', uk_Rsat.shape

    print 'hmf =', hmf.dndlnm.shape
    Pg_s = F_k1 * array([GM_sub_analy(hmf, uk, uk_s, rho_dm,
                                      Nsub_i, ngal_i, Mh_range,
                                      Msub_range, shmf[:,0])
                         for Nsub_i, ngal_i
                         in izip(Nsub_obs, nsub_bar)])

    # I am missing the offset central contribution
    Pg_k = Pg_s
    print 'Pg_k =', Pg_k.shape, Pg_k.max()
    #import pylab
    #for i in xrange(0, len(Pg_k), 4):
        #pylab.loglog(k_range_lin, Pg_k[i])
    #pylab.show()

    # apparently these two are the same
    #P_inter = [UnivariateSpline(k_range_lin, logPg_k, s=0, ext=0)
    P_inter = [InterpolatedUnivariateSpline(k_range_lin, logPg_k, ext=0)
               for logPg_k in izip(log(Pg_k))]

    #print 'rvirs_range_3d =', rvirs_range_3d
    # correlation functions
    xi2 = np.zeros((logMstar_min.size,rvirs_range_3d.size))
    for i in xrange(logMstar_min.size):
        xi2[i] = power_to_corr_ogata(P_inter[i], rvirs_range_3d)
    print 'xi2 =', xi2.shape

    # surface density
    sur_den2 = array([sigma(xi2_i, rho_mean, rvirs_range_3d, rvirs_range_3d_i)
                      for xi2_i in xi2])
    for i in xrange(logMstar_min.size):
        sur_den2[i][(sur_den2[i] <= 0.0) | (sur_den2[i] >= 1e20)] = np.nan
        sur_den2[i] = fill_nan(sur_den2[i])
    print 'sur_den2 =', sur_den2.shape

    # excess surface density
    d_sur_den2 = array([np.nan_to_num(d_sigma(sur_den2_i,
                                              rvirs_range_3d_i,
                                              rvirs_range_2d_i))
                        for sur_den2_i in izip(sur_den2)]) / 1e12
    print 'd_sur_den2 =', d_sur_den2[0], d_sur_den2.shape

    #out_esd_tot = array([UnivariateSpline(rvirs_range_2d_i,
                                          #np.nan_to_num(d_sur_den2_i), s=0)
                         #for d_sur_den2_i in izip(d_sur_den2)])
    #out_esd_tot_inter = np.zeros((logMstar_min.size, rvirs_range_2d_i.size))
    #for i in xrange(logMstar_min.size):
        #out_esd_tot_inter[i] = out_esd_tot[i](rvirs_range_2d_i)
    #print 'out_esd_tot_inter =', out_esd_tot_inter, out_esd_tot_inter.shape

    #return [out_esd_tot_inter, Msub_avg, 0]
    return [d_sur_den2, Msub_avg, 0]


def Nsub_avg(population, hmf, shmf, Mh_range, Msub_range):
    """
    Number of subhalos given the HOD and (S)HMF. Not to be
    confused with nsub(), which is the SHMF itself.

    """
    n_gal_Mh = trapz(shmf * population, Msub_range, axis=0)
    #return trapz(hmf.dndm * n_gal_Mh, Mh_range)


#def nsub_bar(shmf, Nsub, Msub_range):
    #return trapz(nsub * shmf, Msub_range)


def nsub_vdB05(psi, a=0.9, b=0.13):
    """
    subhalo mass function of vdBosch+05. Missing the normalization.

    """
    return (b / psi)**a * exp(-psi / b) / psi


def nsub(psi, alpha, beta, omega, fs, psi_min=1e-4):
    """
    Subhalo mass function of Jiang & van den Bosch (2016), Eqs. 13 and 16.

    I put a minus sign in gamma because something is wrong and it comes
    out negative. However with that minus the shape looks right so until
    the code itself is working I might just leave it there.

    """
    s = (1+alpha) / omega
    gamma = -fs * omega * beta**s / \
            (gammainc(s, beta*psi_min**omega) - gammainc(s, beta))
    # should I use dndpsi or dndlnpsi? Using dndlnpsi for now
    #return gamma * psi**alpha * exp(-beta*psi**omega)# / log(10)
    return gamma * psi**(alpha) * exp(-beta*psi**omega)


def nsub_Msub(phi, Mstar):
    """
    Since this takes the HOD directly it is more flexible - just
    write a new phi_sub itself to change the HOD

    """
    #print phi.shape, Mstar.shape
    #n = ones(phi.shape[0])
    #for i in xrange(n.size):
        #n[i] = trapz(phi[i], Mstar)
    n = trapz(phi, Mstar, axis=0)
    return n


def phi_sub(logMsub, logMstar, logMo, logM1, beta, sigma):
    """
    stellar-to-subhalo mass relation through a conditional mass function

    mu and sigma are the mean and scatter in the SSHMR.
    We assume a constant scatter for now, but can easily be generalized to
    depend on mass.

    To fit for individual masses fix beta to zero.

    """
    if not iterable(logMo):
        logMo = array([logMo])
    # mean subhalo mass given a stellar mass
    #mu = logMo + beta * (logMstar-logM1)
    # should I divide by Msub or Mstar?
    #prob = array([exp(-(logMsub - mu_i)**2 / (2*sigma**2)) / \
                     #((2*pi)**0.5*sigma*Mstar)
                  #for mu_i, Mstar in izip(mu, 10**logMstar)]) / log(10)
    #return prob
    mu = logMo + beta * (logMsub-logM1)
    prob = array([exp(-(logm - mu)**2 / (2*sigma**2)) / 10**logm
                  for logm in logMstar])
    return prob / ((2*pi)**0.5 * sigma * log(10))


