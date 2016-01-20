
from itertools import izip
from numpy import array
# local
from utils import cM_duffy08, delta, density_average

def fiducial(theta, R):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _delta = delta
    _izip = izip

    sat_profile, host_profile, central_profile, \
        Rsat, n_Rsat, fsat, fc_sat, Msat, fc_central, Mcentral, \
        z, Mstar, Rrange, angles = theta

    j = [(ni > 0) for ni in n_Rsat]
    Rsat = [Rsat[i] for i in j]
    n_Rsat = [n[i] for i, n in _izip(j, n_Rsat)]
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
                                              
    esd_total = _array([f * (esat + ehost) + (1-f) * ecentral
                        for f, esat, ehost, ecentral in izip(fsat, esd_sat,
                                                             esd_host,
                                                             esd_central)])
    Mavg = (fsat * Msat**(2./3.) + (1-fsat) * Mcentral**(2./3.))**(3./2.)
    out = [esd_total, esd_sat, esd_host, esd_central, Mavg, 0]
    return out

