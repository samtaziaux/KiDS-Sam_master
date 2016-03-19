import numpy
from numpy import arange, array, exp, iterable, log, log10, logspace, \
                  newaxis, ones, pi, squeeze
from scipy.integrate import trapz
from scipy.special import gammainc

from dark_matter import *
from halo import *
#from nfw import *


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

    # note that we give log-masses to halo.model()
    #z, f, sigma_c, A, M_1, gamma_1, gamma_2, \
        #fc_nsat, alpha_s, b_0, b_1, b_2, Ac2s, \
        #alpha_star, beta_gas, r_t0, r_c0, \
        #Mh_min, Mh_max, Mh_step, Mstar_bin_min, Mstar_bin_max, \
        #centrals, satellites, taylor_procedure, include_baryons, \
        #smth1, smth2 = theta
    # free parameters (these need to be in theta):
    #     stellar-to-subhalo mass relation (SSHMR):
    #         logMo, logM1=10.3, beta, sigma
    #     subhalo mass function: alpha=0.9, beta=0.13 (vdBosch+05)
    #     subhalo mass-concentration relation: acsub, bcsub, gcsub=0
    # halo.model takes the Mstar bins, not means, to input in the HOD.
    # However since here we have only one population this isn't very
    # relevant, so let's just take the mean here
    sigma8, h, Om, Obh2, ns, \
        z, fh, fc_nsat, logMo, logM1, beta_sshmr, sigma_sshmr, \
        fs, alpha_shmf, beta_shmf, omega_shmf, \
        a_cMsub, b_cMsub, g_cMsub, Mo_cMsub, \
        logMh_min, logMh_max, logMh_step, \
        logpsi_min, logpsi_max, logpsi_step, logMstar, \
        lnk_min, lnk_max, lnk_step, \
        smth1, smth2 = theta
    print 'logMo =', logMo

    # HMF set up parameters
    #k_step = (lnk_max-lnk_min) / n_bins
    k_range = arange(lnk_min, lnk_max, lnk_step)
    k_range_lin = exp(k_range)
    # Halo mass and concentration
    #Mh_range = logspace(M_min, M_max, int((M_max-M_min)/M_step))
    Mh_range = 10**arange(logMh_min, logMh_max, logMh_step)
    ch = Con(z, Mh_range, fh)

    # subhalo masses and concentrations
    #Msub_min = 5
    #Msub_max = Mh_max
    #Msub_step = 0.01
    #Msub_range = 10**arange(logMsub_min, Mh_max, Msub_step)
    # here I need to change the Msub range depending on Mh
    psi_range = 10**arange(logpsi_min, logpsi_max, logpsi_step)
    Msub_range = array([psi * Mh_range for psi in psi_range])

    # for now taking the Duffy+ relation as well.
    #csub = Con(z, Msub_range, fsub)
    csub = cM(z, Msub_range, a_cMsub, b_cMsub, g_cMsub, Mo_cMsub)

    # stellar masses - I don't think I need an HOD for satellite-only
    # (or central-only) samples, do I? I'm just taking the mid point in the
    # bin here. Note that if this really is the case then it's better to
    # provide the adopted value per bin in the config file (e.g., the
    # median logMstar in each bin), but let's keep it like this for now.
    #logMstar = (Mstar_bin_min + Mstar_bin_max) / 2
    #n_bins_obs = Mstar_bin_min.size
    print 'psi =', psi_range.shape
    print 'Msub_range =', Msub_range[0][:10], Msub_range.shape
    print 'Mh_range =', Mh_range.shape
    print 'logMstar =', logMstar, logMstar.shape

    hmfparams = {'sigma_8': sigma8, 'H0': 100.*h,'omegab_h2': Obh2,
                 'omegam': Om, 'omegav': 1-Om, 'n': ns,
                 'lnk_min': lnk_min,'lnk_max': lnk_max,
                 'dlnk': lnk_step, 'transfer_fit': 'BBKS', 'z': z,
                 'force_flat': True}
    hmf = Mass_Function(logMh_min, logMh_max, logMh_step,
                        'Tinker10', **hmfparams)
    omegab = hmf.omegab
    omegac = hmf.omegac
    nh = hmf.dndm
    mass_func = hmf.dndlnm
    rho_mean = hmf.mean_dens_z
    rho_crit = rho_mean / (omegac+omegab)
    rho_dm = rho_mean * baryons.f_dm(omegab, omegac)
    # subhalo mass function
    # I could also set a very low psi_res (default 1e-4) and multiply this
    # by an error function to account for incompleteness
    #shmf = nsub_vdB05(Msub_range, Mh, a_shmf, b_shmf)
    shmf = nsub(psi_range, alpha_shmf, beta_shmf, omega_shmf, fs)
    print 'shmf =', shmf.shape

    # what do I need all of these for?
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

    # here I can use the stellar-to-halo mass relation typically used
    # for centrals, to get the stellar-to-subhalo mass relation of satellites
    # I believe I don't need this because I assume all galaxies are
    # satellites and so the number of them doesn't matter
    #pop_s = array([ncm(hmf, i[0], Msub_range, sigma_c, alpha_s, A, M_1,
                       #gamma_1, gamma_2)
                   #for i in izip(hod_mass)])
    # for now, just anything with the right shape
    pop_s = 1
    ngal = 1

    # average Msub given Mh
    phi = phi_sub(log10(Msub_range), logMstar, logMo, logM1, beta_sshmr,
                  sigma_sshmr)
    print 'phi =', phi.shape
    Msub_Mh = trapz(Msub_range * shmf[:,newaxis] * phi, axis=1)
    # average Msub accounting for the HMF. This is a probability.
    print 'Msub_Mh =', Msub_Mh.shape
    print 'nh =', nh.shape
    Msub_avg = trapz(Msub_Mh * nh, Mh_range, axis=1)
    print 'Msub =', Msub_avg, Msub_avg.shape

    # damping of the 1h power spectra at small k
    F_k1 = f_k(k_range_lin)
    # Fourier Transform of the host NFW profile
    concentration = Con(z, Mh_range, fh)
    uk = NFW_f(z, rho_dm, fh, Mh_range, rvirh_range_lin, k_range_lin,
               c=concentration)
    uk = uk/uk[0]
    # FT of the subhalo NFW profile
    csub = cM(z, Msub_range, a_cMsub, b_cMsub, g_cMsub)
    uk_s = array([NFW_f(z, rho_dm, 0, Msub_i, rvirs, k_range_lin, c)
                  for Msub_i, rvirs, c
                  in izip(Msub_range, rvirs_range_lin, csub)]).transpose(0,2,1)
    # and of the NFW profile of the satellites. This I would like to be
    # the actual Rsat distribution measured from the data (right?)
    #print fc_nsat
    uk_Rsat = NFW_f(z, rho_dm, fc_nsat, Mh_range, rvirh_range_lin, k_range_lin)
    uk_Rsat = uk_Rsat/uk_Rsat[0]

    print 'hmf =', hmf.dndlnm.shape
    print 'uk =', uk.shape
    print 'uk_s =', uk_s.shape
    #print 'pop_s =', pop_s.shape
    #print 'ngal =', ngal.shape
    print 'Mh_range =', Mh_range.shape
    print 'Msub_range =', Msub_range.shape
    print 'shmf =', shmf.shape
    #Pg_s = F_k1 * array([GM_sub_analy(hmf, uk, uk_s, rho_dm,
                                      #pop_s_i, ngal_i, Mh_range,
                                      #Msub_range, shmf)
                         #for pop_s_i, ngal_i in izip(pop_s, ngal)])
    Pg_s = F_k1 * array([GM_sub_analy(hmf, uk, uk_s_i, rho_dm,
                                      pop_s, ngal, Mh_range,
                                      Msub_i, shmf_i)
                         for uk_s_i, Msub_i, shmf_i
                         in izip(uk_s, Msub_range, shmf)])

    # I am missing the offset central contribution
    # the normalization is just to have numbers make more sense.
    # the proper normalization is encoded in ngal and pop_s, which
    # I haven't implemented yet.
    Pg_k = Pg_s / Pg_s.max() / 100
    print 'Pg_k =', Pg_k.shape, Pg_k.max()
    #import pylab
    #for i in xrange(0, len(Pg_k), 4):
        #pylab.loglog(k_range_lin, Pg_k[i])
    #pylab.show() 

    P_inter = [UnivariateSpline(k_range_lin, np.log(Pg_k_i), s=0, ext=0)
               for Pg_k_i in izip(Pg_k)]

    #print 'rvirs_range_3d =', rvirs_range_3d
    # correlation functions
    xi2 = np.zeros((logMstar.size,rvirs_range_3d.size))
    for i in xrange(logMstar.size):
        xi2[i] = power_to_corr_ogata(P_inter[i], rvirs_range_3d)
    print 'xi2 =', xi2, xi2.shape

    # surface density
    sur_den2 = array([sigma(xi2_i, rho_mean, rvirs_range_3d, rvirs_range_3d_i)
                      for xi2_i in xi2])
    for i in xrange(logMstar.size):
        sur_den2[i][(sur_den2[i] <= 0.0) | (sur_den2[i] >= 1e20)] = np.nan
        sur_den2[i] = fill_nan(sur_den2[i])
    print 'sur_den2 =', sur_den2, sur_den2.shape

    # excess surface density
    d_sur_den2 = array([np.nan_to_num(d_sigma(sur_den2_i,
                                              rvirs_range_3d_i,
                                              rvirs_range_2d_i))
                        for sur_den2_i in izip(sur_den2)]) / 1e12
    print 'd_sur_den2 =', d_sur_den2, d_sur_den2.shape

    out_esd_tot = array([UnivariateSpline(rvirs_range_2d_i,
                                          np.nan_to_num(d_sur_den2_i), s=0)
                         for d_sur_den2_i in izip(d_sur_den2)])
    print 'out_esd_tot =', out_esd_tot, out_esd_tot.shape
    out_esd_tot_inter = np.zeros((logMstar.size, rvirs_range_2d_i.size))
    for i in xrange(logMstar.size):
        out_esd_tot_inter[i] = out_esd_tot[i](rvirs_range_2d_i)

    return [out_esd_tot_inter, Msub_avg, csub, 0]


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
    # should I use dndpsi or dndlnpsi? Using dndpsi for now
    return gamma * psi**alpha * exp(-beta*psi**omega) / log(10)


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
    mu = logMo + beta * (logMstar-logM1)
    #return squeeze([exp((logMsub - mu_i)**2 / (2*sigma_i**2)) / \
    prob = array([exp(-(logMsub - mu_i)**2 / (2*sigma**2)) / \
                      ((2*pi)**0.5*sigma)
                  for mu_i in mu])
    return prob


