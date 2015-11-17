from itertools import count, izip
from numpy import all as npall, array, average, cumsum, digitize
from numpy import linspace, log, log10, max as npmax, median, newaxis
from numpy import ones, outer, sum as npsum, transpose, zeros
from numpy.random import random, rayleigh
from scipy.stats import rv_discrete
from scipy.integrate import romberg
from time import time

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
                   for pm, Ri, rs, sigma in izip(pointmass, R,
                                                 rs_cent, sigma_central)]
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
    bias = calc_bias.bias(M, Om, omegab_h2, sigma_8, h)
    esd_2halo = [bias * dsigma_mm(sigma_8, h, omegab_h2, Om, Ol, n, zi, Ri)
                   for Ri in izip(R)]
    esd_2halo = _array(esd_2halo)
    
    esd_total = _array([f * (esat + ehost) + (1-f) * ecentral
                        for f, esat, ehost, ecentral in izip(fsat, esd_sat,
                                                             esd_host,
                                                             esd_central)])
                                                             
    Mavg = ( fsat * Msat**(2./3.) + (1-fsat) * Mcentral**(2./3.) )**(3./2.)
    out = [esd_total, esd_sat, esd_host, esd_central, Mavg, 0]
    return out
