from itertools import count, izip
from numpy import all as npall, array, average, cumsum, digitize
from numpy import linspace, log, log10, max as npmax, median, newaxis
from numpy import ones, outer, sum as npsum, transpose, zeros
from numpy.random import random, rayleigh
from scipy.stats import rv_discrete
from scipy.integrate import romberg
from time import time
import numpy as np

# local
from nfw import esd, esd_offset, esd_sharp, esd_trunc5, esd_trunc7
from nfw import mass_enclosed, mass_total_sharp
from utils import cM_duffy08, delta, density_average

from twohalo_mm import dsigma_mm
import calc_bias


def fiducial4(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    
    sat_profile, host_profile, central_profile, \
        Rsat, n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4, \
        fsat, fc_sat, logMsat1, logMsat2, logMsat3, logMsat4, \
        fc_central1, fc_central2, fc_central3, fc_central4, \
        logMcentral1, logMcentral2, logMcentral3, logMcentral4, \
        z, Mstar, Rrange, angles = theta
        
    n_Rsat = (n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4)
    j = [(ni > 0) for ni in n_Rsat]
    Rsat = [Rsat[i] for i in j]
    n_Rsat = [n[i] for i, n in _izip(j, n_Rsat)]
    Msat = _array([logMsat1, logMsat2, logMsat3, logMsat4])
    Mcentral = _array([logMcentral1, logMcentral2,
                           logMcentral3, logMcentral4])
    fc_central = _array([fc_central1, fc_central2,
                           fc_central3, fc_central4])
    csat = fc_sat * _cM_duffy08(Msat, z, h)
    ccentral = fc_central * _cM_duffy08(Mcentral, z, h)
    
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200*rho_m * 4*3.14159265/3
    
    # more parameters
    r200_sat = (Msat / aux) ** (1./3)
    rs_sat = r200_sat / csat
    sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mcentral / aux) ** (1./3)
    rs_cent = (Mcentral / aux) ** (1./3) / ccentral
    sigma_central = rs_cent * _delta(ccentral) * rho_m
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2)
                 for Mi, Ri in izip(Mstar, R)]
                 
    # satellite signal
    esd_sat = [pm + sat_profile(Ri[1:]/rs, sigma)
               for pm, Ri, rs, sigma in izip(pointmass, R, rs_sat, sigma_sat)]
    esd_sat = _array(esd_sat)
    
    # satellite host halo signal
    esd_host = [host_profile(x[0]/x[1], x[2]/x[1], x[3], x[4],
                             x[5]/x[1], angles)
                for x in izip(R, rs_cent, Rsat, n_Rsat,
                              sigma_central, Rrange)]
    esd_host = _array(esd_host)
    
    # central signal
    esd_central = [pm + central_profile(Ri[1:]/rs, sigma)
                   for pm, Ri, rs, sigma in izip(pointmass, R,
                                                 rs_cent, sigma_central)]
    esd_central = _array(esd_central)
    #lnPderived = _log(nfw_profile(rsat, rs_cent, sigma_group)).sum()
    #lnPderived -= _log(_array([romberg(nfw_profile, x[0].min(), rmax,
                                       #args=(x[1],x[2]))
                               #for x in _izip(Rsat, rs_cent,
                                              #sigma_group)])).sum()
                                              
    esd_total = _array([f * (esat + ehost) + (1-f) * ecentral
                        for f, esat, ehost, ecentral in izip(fsat, esd_sat,
                                                             esd_host,
                                                             esd_central)])
    Mavg = ( fsat * Msat**(2./3.) + (1-fsat) * Mcentral**(2./3.) )**(3./2.)
    out = [esd_total, esd_sat, esd_host, esd_central, Mavg, 0]
    return out



def fiducial4_2halo(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    
    sat_profile, host_profile, central_profile, \
        Rsat, n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4, \
        fsat, fc_sat, logMsat1, logMsat2, logMsat3, logMsat4, \
        fc_central1, fc_central2, fc_central3, fc_central4, \
        logMcentral1, logMcentral2, logMcentral3, logMcentral4, \
        z, Mstar, Rrange, angles = theta
        
    n_Rsat = (n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4)
    j = [(ni > 0) for ni in n_Rsat]
    Rsat = [Rsat[i] for i in j]
    n_Rsat = [n[i] for i, n in _izip(j, n_Rsat)]
    Msat = _array([logMsat1, logMsat2, logMsat3, logMsat4])
    Mcentral = _array([logMcentral1, logMcentral2,
                           logMcentral3, logMcentral4])
    fc_central = _array([fc_central1, fc_central2,
                           fc_central3, fc_central4])
    csat = fc_sat * _cM_duffy08(Msat, z, h)
    ccentral = fc_central * _cM_duffy08(Mcentral, z, h)
    
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200*rho_m * 4*3.14159265/3
    
    # more parameters
    r200_sat = (Msat / aux) ** (1./3)
    rs_sat = r200_sat / csat
    sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mcentral / aux) ** (1./3)
    rs_cent = (Mcentral / aux) ** (1./3) / ccentral
    sigma_central = rs_cent * _delta(ccentral) * rho_m
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2)
                 for Mi, Ri in izip(Mstar, R)]
                 
    # satellite signal
    esd_sat = [pm + sat_profile(Ri[1:]/rs, sigma)
               for pm, Ri, rs, sigma in izip(pointmass, R, rs_sat, sigma_sat)]
    esd_sat = _array(esd_sat)
    
    # satellite host halo signal
    esd_host = [host_profile(x[0]/x[1], x[2]/x[1], x[3], x[4],
                             x[5]/x[1], angles)
                for x in izip(R, rs_cent, Rsat, n_Rsat,
                              sigma_central, Rrange)]
    esd_host = _array(esd_host)
    
    # central signal
    esd_central = [pm + central_profile(Ri[1:]/rs, sigma)
                   for pm, Ri, rs, sigma
                   in izip(pointmass, R, rs_cent, sigma_central)]
    esd_central = _array(esd_central)
    #lnPderived = _log(nfw_profile(rsat, rs_cent, sigma_group)).sum()
    #lnPderived -= _log(_array([romberg(nfw_profile, x[0].min(), rmax,
                                       #args=(x[1],x[2]))
                               #for x in _izip(Rsat, rs_cent,
                                              #sigma_group)])).sum()
    
    # 2-halo term signal
    sigma_8 = 0.829
    omegab_h2 = 0.02205
    n = 0.9603
    bias = calc_bias.calc_bias(Mcentral, Om, omegab_h2, sigma_8, h)
    esd_2halo = [biasi * dsigma_mm(sigma_8, h, omegab_h2, Om, Ol, n, zi, Ri[1:])[0]
                                        for biasi, zi, Ri
                                        in izip(bias, z, R)]
    esd_2halo = _array(esd_2halo)

    # Total signal
    esd_total = _array([f * (esat + ehost) + (1-f) * ecentral + e2halo
                        for f, esat, ehost, ecentral, e2halo in izip(fsat, esd_sat,
                                                             esd_host,
                                                             esd_central,
                                                             esd_2halo)])
    Mavg = ( fsat * Msat**(2./3.) + (1-fsat) * Mcentral**(2./3.) )**(3./2.)
    out = [esd_total, esd_sat, esd_host, esd_central, esd_2halo, Mavg, 0]
    return out
    
    
def fiducial_auto(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    
    sat_profile, host_profile, central_profile, \
        Rsat, n_Rsat, fsat, fc_sat, Msat, \
        fc_central, A_2halo, Mcentral, \
        z, Mstar, twohalo, Rrange, \
        angles = theta
        
#    n_Rsat = (n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4)
    j = [(ni > 0) for ni in n_Rsat]
    Rsat = [Rsat[i] for i in j]
    n_Rsat = [n[i] for i, n in _izip(j, n_Rsat)]

#    Msat = _array([logMsat1, logMsat2, logMsat3, logMsat4])
#    Mcentral = _array([logMcentral1, logMcentral2, logMcentral3, logMcentral4])
#    fc_central = _array([fc_central1, fc_central2, fc_central3, fc_central4])

#    A_2halo = _array([A_2halo1, A_2halo2, A_2halo3, A_2halo4])
    csat = fc_sat * _cM_duffy08(Msat, z, h)
    ccentral = fc_central * _cM_duffy08(Mcentral, z, h)
    
#    dsigma_mm = (twohalo1, twohalo2, twohalo3, twohalo4)
    j = [(ni > 0) for ni in twohalo]
    dsigma_mm = [n[i] for i, n in _izip(j, twohalo)]
    
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200*rho_m * 4*3.14159265/3
    
    # more parameters
    r200_sat = (Msat / aux) ** (1./3)
    rs_sat = r200_sat / csat
    sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mcentral / aux) ** (1./3)
    rs_cent = (Mcentral / aux) ** (1./3) / ccentral
    sigma_central = rs_cent * _delta(ccentral) * rho_m
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2)
                 for Mi, Ri in izip(Mstar, R)]
    
                 
    # satellite signal
    esd_sat = [pm + sat_profile(Ri[1:]/rs, sigma)
               for pm, Ri, rs, sigma in izip(pointmass, R, rs_sat, sigma_sat)]
    esd_sat = _array(esd_sat)
    
    # satellite host halo signal
    esd_host = [host_profile(x[0]/x[1], x[2]/x[1], x[3], x[4],
                             x[5]/x[1], angles)
                for x in izip(R, rs_cent, Rsat, n_Rsat,
                              sigma_central, Rrange)]
    esd_host = _array(esd_host)

    
    # central signal
    esd_central = [pm + central_profile(Ri[1:]/rs, sigma)
                   for pm, Ri, rs, sigma
                   in izip(pointmass, R, rs_cent, sigma_central)]
    esd_central = _array(esd_central)

    
    # 2-halo term signal
    sigma_8 = 0.829
    omegab_h2 = 0.02205
    bias = calc_bias.calc_bias(Mcentral, Om, omegab_h2, sigma_8, h)
    esd_2halo = [A_2haloi * biasi * dsigma_mmi \
                                    for A_2haloi, biasi, dsigma_mmi \
                                    in izip(A_2halo, bias, dsigma_mm)]
    esd_2halo = _array(esd_2halo)

    # Total signal
    esd_total = _array([f * (esat + ehost) + (1-f) * ecentral + e2halo
                        for f, esat, ehost, ecentral, e2halo in izip(fsat, esd_sat,
                                                             esd_host,
                                                             esd_central,
                                                             esd_2halo)])
    Mavg = ( fsat * Msat**(2./3.) + (1-fsat) * Mcentral**(2./3.) )**(3./2.)
    out = [esd_total, esd_sat, esd_host, esd_central, esd_2halo, Mavg, 0]
    return out
    
    
def fiducial4_auto(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    
    sat_profile, host_profile, central_profile, \
        Rsat, n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4, \
        fsat, fc_sat, logMsat1, logMsat2, logMsat3, logMsat4, \
        fc_central1, fc_central2, fc_central3, fc_central4, \
        A_2halo1, A_2halo2, A_2halo3, A_2halo4, \
        logMcentral1, logMcentral2, logMcentral3, logMcentral4, \
        z, Mstar, twohalo1, twohalo2, twohalo3, twohalo4, Rrange, \
        angles = theta
        
    n_Rsat = (n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4)
    j = [(ni > 0) for ni in n_Rsat]
    Rsat = [Rsat[i] for i in j]
    n_Rsat = [n[i] for i, n in _izip(j, n_Rsat)]
    Msat = _array([logMsat1, logMsat2, logMsat3, logMsat4])
    Mcentral = _array([logMcentral1, logMcentral2,
                           logMcentral3, logMcentral4])
    fc_central = _array([fc_central1, fc_central2,
                           fc_central3, fc_central4])
    A_2halo = _array([A_2halo1, A_2halo2, A_2halo3, A_2halo4])
    csat = fc_sat * _cM_duffy08(Msat, z, h)
    ccentral = fc_central * _cM_duffy08(Mcentral, z, h)
    
    dsigma_mm = (twohalo1, twohalo2, twohalo3, twohalo4)
    j = [(ni > 0) for ni in dsigma_mm]
    dsigma_mm = [n[i] for i, n in _izip(j, dsigma_mm)]
    
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200*rho_m * 4*3.14159265/3
    
    # more parameters
    r200_sat = (Msat / aux) ** (1./3)
    rs_sat = r200_sat / csat
    sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mcentral / aux) ** (1./3)
    rs_cent = (Mcentral / aux) ** (1./3) / ccentral
    sigma_central = rs_cent * _delta(ccentral) * rho_m
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2)
                 for Mi, Ri in izip(Mstar, R)]
                 
    # satellite signal
    esd_sat = [pm + sat_profile(Ri[1:]/rs, sigma)
               for pm, Ri, rs, sigma in izip(pointmass, R, rs_sat, sigma_sat)]
    esd_sat = _array(esd_sat)
    
    # satellite host halo signal
    esd_host = [host_profile(x[0]/x[1], x[2]/x[1], x[3], x[4],
                             x[5]/x[1], angles)
                for x in izip(R, rs_cent, Rsat, n_Rsat,
                              sigma_central, Rrange)]
    esd_host = _array(esd_host)
    
    # central signal
    esd_central = [pm + central_profile(Ri[1:]/rs, sigma)
                   for pm, Ri, rs, sigma
                   in izip(pointmass, R, rs_cent, sigma_central)]
    esd_central = _array(esd_central)
    #lnPderived = _log(nfw_profile(rsat, rs_cent, sigma_group)).sum()
    #lnPderived -= _log(_array([romberg(nfw_profile, x[0].min(), rmax,
                                       #args=(x[1],x[2]))
                               #for x in _izip(Rsat, rs_cent,
                                              #sigma_group)])).sum()
    
    # 2-halo term signal
    sigma_8 = 0.829
    omegab_h2 = 0.02205
    bias = calc_bias.calc_bias(Mcentral, Om, omegab_h2, sigma_8, h)
    esd_2halo = [A_2haloi * biasi * dsigma_mmi \
                                    for A_2haloi, biasi, dsigma_mmi \
                                    in izip(A_2halo, bias, dsigma_mm)]
    esd_2halo = _array(esd_2halo)

    # Total signal
    esd_total = _array([f * (esat + ehost) + (1-f) * ecentral + e2halo
                        for f, esat, ehost, ecentral, e2halo in izip(fsat, esd_sat,
                                                             esd_host,
                                                             esd_central,
                                                             esd_2halo)])
    Mavg = ( fsat * Msat**(2./3.) + (1-fsat) * Mcentral**(2./3.) )**(3./2.)
    out = [esd_total, esd_sat, esd_host, esd_central, esd_2halo, Mavg, 0]
    return out

def fiducial_bias(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    
    #sat_profile, central_profile, fsat, fc_sat, logMsat1, logMsat2, fc_central1, fc_central2, logMcentral1, logMcentral2, z, Mstar, Rrange, angles = theta
    central_profile, fc_central, logMcentral, b_in, z, Mstar, Rrange, angles = theta
    
    
    #Msat = 10.0**_array([logMsat1, logMsat2])
    Mcentral = 10.0**logMcentral #_array([logMcentral1, logMcentral2])
    b = b_in #_array([b_in1, b_in2])
    
    Mstar = 10.0**Mstar
    
    #fc_central = _array([fc_central1, fc_central2])
    
    #csat = fc_sat * _cM_duffy08(Msat, z, h)
    ccentral = fc_central * _cM_duffy08(Mcentral, z, h)
    
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200.0 * rho_m * 4.0*3.14159265/3
    
    # more parameters
    #r200_sat = (Msat / aux) ** (1./3.)
    #rs_sat = r200_sat / csat
    #sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mcentral / aux) ** (1./3)
    rs_cent = (Mcentral / aux) ** (1./3.) / ccentral
    sigma_central = rs_cent * _delta(ccentral) * rho_m
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2.) for Mi, Ri in izip(Mstar, R)]
    
    # satellite signal
    #esd_sat = [sat_profile(Ri[1:]/rs, sigma) for Ri, rs, sigma in izip(R, rs_sat, sigma_sat)]
    #esd_sat = _array(esd_sat)
    
    
    # central signal
    esd_central = [pm + central_profile(Ri[1:]/rs, sigma) for pm, Ri, rs, sigma in izip(pointmass, R, rs_cent, sigma_central)]
    esd_central = _array(esd_central)
    
    
    # 2-halo term signal
    sigma_8 = 0.829
    omegab_h2 = 0.02205
    n = 0.9603
    bias = calc_bias.calc_bias(Mcentral, Om, omegab_h2, sigma_8, h)
    esd_2halo = [b_i * biasi * dsigma_mm(sigma_8, h, omegab_h2, Om, Ol, n, zi, Ri[1:])[0] for b_i, biasi, zi, Ri in izip(b, bias, z, R)]
    esd_2halo = _array(esd_2halo)
    
    # Total signal
    #esd_total = _array([f * (esat) + (1.-f) * ecentral + e2halo for f, esat, ecentral, e2halo in izip(fsat, esd_sat, esd_central, esd_2halo)])
    esd_total = _array([ecentral + e2halo for ecentral, e2halo in izip(esd_central, esd_2halo)])
    
    Mavg = Mcentral#( fsat * Msat**(2./3.) + (1.-fsat) * Mcentral**(2./3.) )**(3./2.)
    
    #out = [esd_total, esd_central, esd_sat, esd_2halo, Mavg, 0]
    out = [esd_total, esd_central, esd_2halo, Mavg, 0]
    #print logMsat1, logMsat2, fc_central1, fc_central2, logMcentral1, logMcentral2
    print fc_central, logMcentral, b_in
    return out


def fiducial_bias_off(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    _linspace = linspace
    
    #sat_profile, central_profile, fsat, fc_sat, logMsat1, logMsat2, fc_central1, fc_central2, logMcentral1, logMcentral2, z, Mstar, Rrange, angles = theta
    central_profile, host_profile, fc_central, logMcentral, alpha_in, f_off_in, b_in, z, Mstar, Rrange, angles = theta
    
    
    #Msat = 10.0**_array([logMsat1, logMsat2])
    Mcentral = 10.0**logMcentral #_array([logMcentral1, logMcentral2])
    b = b_in #_array([b_in1, b_in2])
    alpha = alpha_in #_array([alpha_in1, alpha_in2])
    f_off = f_off_in #_array([f_off_in1, f_off_in2])
    
    Mstar = 10.0**Mstar
    
    #fc_central = _array([fc_central1, fc_central2])
    fc_central = fc_central
    
    #csat = fc_sat * _cM_duffy08(Msat, z, h)
    ccentral = fc_central * _cM_duffy08(Mcentral, z, h)
    
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200.0 * rho_m * 4*3.14159265/3
    
    # more parameters
    #r200_sat = (Msat / aux) ** (1./3.)
    #rs_sat = r200_sat / csat
    #sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mcentral / aux) ** (1./3)
    r_vir = (Mcentral / aux) ** (1./3.)
    rs_cent = (Mcentral / aux) ** (1./3.) / ccentral
    sigma_central = rs_cent * _delta(ccentral) * rho_m
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2.) for Mi, Ri in izip(Mstar, R)]
    
    # satellite signal
    #esd_sat = [sat_profile(Ri[1:]/rs, sigma) for Ri, rs, sigma in izip(R, rs_sat, sigma_sat)]
    #esd_sat = _array(esd_sat)
    
    
    # central signal
    esd_central = [pm + central_profile(Ri[1:]/rs, sigma) for pm, Ri, rs, sigma in izip(pointmass, R, rs_cent, sigma_central)]
    esd_central = _array(esd_central)
    
    # mis-centered central signal
    
    Roff = _linspace(Rrange[0][1], Rrange[0][-1], 100)
    n_Roff = lambda x: (Roff/(x)**2.0) * np.exp(-0.5 * (Roff/x)**2.0)
    #n_Roff = lambda x: np.exp(-0.5 * (Roff/x)**2.0)
    
    #n_Roff = (n_Roff(alpha[0]*r_vir[0]), n_Roff(alpha[1]*r_vir[1]))
    n_Roff = [n_Roff(alpha_i * r_vi) for alpha_i, r_vi in izip(alpha, r_vir)]
    
    j = [(ni > 0) for ni in n_Roff]
    Roff = [Roff[i] for i in j]
    n_Roff = [n[i] for i, n in _izip(j, n_Roff)]
    
    esd_host = [host_profile(x[0]/x[1], x[2]/x[1], x[3], x[4], x[5]/x[1], angles) for x in izip(R, rs_cent, Roff, n_Roff, sigma_central, Rrange)]
    esd_host = _array(esd_host)
    
    # 2-halo term signal
    sigma_8 = 0.829
    omegab_h2 = 0.02205
    n = 0.9603
    bias = calc_bias.calc_bias(Mcentral, Om, omegab_h2, sigma_8, h)
    esd_2halo = [b_i * biasi * dsigma_mm(sigma_8, h, omegab_h2, Om, Ol, n, zi, Ri[1:])[0] for b_i, biasi, zi, Ri in izip(b, bias, z, R)]
    esd_2halo = _array(esd_2halo)
    
    # Total signal
    #esd_total = _array([f * (esat) + (1.-f) * ecentral + e2halo for f, esat, ecentral, e2halo in izip(fsat, esd_sat, esd_central, esd_2halo)])
    esd_total = _array([fo * ecentral + (1.0 - fo)*eoff + e2halo for ecentral, eoff, e2halo, fo in izip(esd_central, esd_host, esd_2halo, f_off)])
    
    Mavg = Mcentral #( fsat * Msat**(2./3.) + (1.-fsat) * Mcentral**(2./3.) )**(3./2.)
    
    #out = [esd_total, esd_central, esd_sat, esd_2halo, Mavg, 0]
    out = [esd_total, esd_central, esd_host, esd_2halo, pointmass, Mavg, 0]
    #print logMsat1, logMsat2, fc_central1, fc_central2, logMcentral1, logMcentral2
    print fc_central, logMcentral, alpha_in, f_off_in, b_in
    return out


def fiducial_bias_cm(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    
    #sat_profile, central_profile, fsat, fc_sat, logMsat1, logMsat2, fc_central1, fc_central2, logMcentral1, logMcentral2, z, Mstar, Rrange, angles = theta
    central_profile, fc_central, logMcentral, b_in, z, Mstar, Rrange, angles = theta
    
    
    #Msat = 10.0**_array([logMsat1, logMsat2])
    Mcentral = 10.0**logMcentral #_array([logMcentral1, logMcentral2, logMcentral3, logMcentral4])
    b = b_in #_array([b_in1, b_in2, b_in3, b_in4])
    
    Mstar = 10.0**Mstar
    
    #fc_central = _array([fc_central1, fc_central2, fc_central3, fc_central4])
    
    #csat = fc_sat * _cM_duffy08(Msat, z, h)
    ccentral = fc_central * _cM_duffy08(Mcentral, z, h)
    
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200.0 * rho_m * 4*3.14159265/3
    
    # more parameters
    #r200_sat = (Msat / aux) ** (1./3.)
    #rs_sat = r200_sat / csat
    #sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mcentral / aux) ** (1./3)
    rs_cent = (Mcentral / aux) ** (1./3.) / ccentral
    sigma_central = rs_cent * _delta(ccentral) * rho_m
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2.) for Mi, Ri in izip(Mstar, R)]
    
    # satellite signal
    #esd_sat = [sat_profile(Ri[1:]/rs, sigma) for Ri, rs, sigma in izip(R, rs_sat, sigma_sat)]
    #esd_sat = _array(esd_sat)
    
    
    # central signal
    esd_central = [pm + central_profile(Ri[1:]/rs, sigma) for pm, Ri, rs, sigma in izip(pointmass, R, rs_cent, sigma_central)]
    esd_central = _array(esd_central)
    
    
    
    # 2-halo term signal
    sigma_8 = 0.829
    omegab_h2 = 0.02205
    n = 0.9603
    bias = calc_bias.calc_bias(Mcentral, Om, omegab_h2, sigma_8, h)
    esd_2halo = [b_i * biasi * dsigma_mm(sigma_8, h, omegab_h2, Om, Ol, n, zi, Ri[1:])[0] for b_i, biasi, zi, Ri in izip(b, bias, z, R)]
    esd_2halo = _array(esd_2halo)
    
    # Total signal
    #esd_total = _array([f * (esat) + (1.-f) * ecentral + e2halo for f, esat, ecentral, e2halo in izip(fsat, esd_sat, esd_central, esd_2halo)])
    esd_total = _array([ecentral + e2halo for ecentral, e2halo in izip(esd_central, esd_2halo)])
    
    Mavg = Mcentral#( fsat * Msat**(2./3.) + (1.-fsat) * Mcentral**(2./3.) )**(3./2.)
    
    #out = [esd_total, esd_central, esd_sat, esd_2halo, Mavg, 0]
    out = [esd_total, esd_central, esd_2halo, Mavg, 0]
    #print logMsat1, logMsat2, fc_central1, fc_central2, logMcentral1, logMcentral2
    print fc_central, logMcentral, b_in
    return out
