from itertools import count, izip
from numpy import all as npall, array, average, cumsum, digitize
from numpy import linspace, log, log10, max as npmax, median
from numpy import ones, outer, sum as npsum, transpose, zeros
from numpy.random import random, rayleigh
from scipy.stats import rv_discrete
from scipy.integrate import romberg
from time import time

# local
from nfw import esd, esd_offset, esd_sharp, esd_trunc5, esd_trunc7
from nfw import mass_enclosed, mass_total_sharp
from utils import cM_duffy08, delta, density_average

def fiducial(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _izip = izip
    #_nfw_profile = nfw_profile
    sat_profile, host_profile, central_profile, \
        Rsat, n_Rsat1, n_Rsat2, n_Rsat3, n_Rsat4, \
        fsat, fc_sat, logMsat1, logMsat2, logMsat3, logMsat4, \
        fc_central, logMcentral1, logMcentral2, logMcentral3, logMcentral4, \
        z, Mstar, Rrange, angles = theta
    n_Rsat = (n_Rsat1, n_Rsat2, n_Rsat3)
    j = [(ni > 0) for ni in n_Rsat]
    Rsat = [Rsat[i] for i in j]
    n_Rsat = [n[i] for i, n in _izip(j, n_Rsat)]
    Msat = 10**_array([logMsat1, logMsat2, logMsat3, logMsat4])
    Mcentral = 10**_array([logMcentral1, logMcentral2,
                           logMcentral3, logMcentral4])
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
    pointmass = Mstar / (3.14159265*(1e6*R[1:])**2)
    # satellite signal
    esd_sat = pointmass + sat_profile(R[1:]/rs_sat, sigma_sat)
    # satellite host halo signal
    esd_host = host_profile(R/rs_cent, Rsat/rs_cent, n_Rsat,
                            sigma_central, Rrange/rs_cent, angles)
    # central signal
    esd_central = pointmass + central_profile(R[1:]/rs_cent, sigma_central)
    #lnPderived = _log(nfw_profile(rsat, rs_cent, sigma_group)).sum()
    #lnPderived -= _log(_array([romberg(nfw_profile, x[0].min(), rmax,
                                       #args=(x[1],x[2]))
                               #for x in _izip(Rsat, rs_cent,
                                              #sigma_group)])).sum()
    esd_total = fsat * (esd_sat + esd_host) + (1-fsat) * esd_central
    Mavg = fsat * Msat + (1-fsat) * Mcentral
    #out = [esd_total, esd_sat, esd_group,
           #rsat, rt, Msat_rt+Mstar, lnPderived]
    out = [esd_total, esd_sat, esd_host, esd_central,
           log10(Mavg), logMsat, logMcentral, 0]
    return out
