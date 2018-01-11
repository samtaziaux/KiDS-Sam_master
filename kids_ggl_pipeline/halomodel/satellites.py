from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
from itertools import count
if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
from numpy import all as npall, array, average, cumsum, digitize
from numpy import linspace, log, log10, max as npmax, median
from numpy import ones, outer, sum as npsum, transpose, zeros
from numpy.random import random, rayleigh
from scipy.stats import rv_discrete
from scipy.integrate import romberg

# local
from .nfw import (esd, esd_offset, esd_sharp, esd_trunc5, esd_trunc7
                  mass_enclosed, mass_total_sharp)
from ,utils import cM_duffy08, delta, density_average, nfw_profile


def fiducial(theta, R, h=1, Om=0.315, Ol=0.685, rmax=2):
    # local variables are accessed much faster than global ones
    _abs = abs
    _array = array
    _cM_duffy08 = cM_duffy08
    _cumsum = cumsum
    _delta = delta
    _digitize = digitize
    _izip = izip
    _log = log
    _mass_total_sharp = mass_total_sharp
    _min = min
    _nfw_profile = nfw_profile
    _npall = npall
    _random = random
    _transpose = transpose
    _zeros = zeros
    sat_profile, group_profile, \
        Rsat, n_Rsat1, n_Rsat2, n_Rsat3, rsat_range, \
        fc_sat, logMsat1, logMsat2, logMsat3, \
        fc_group, logMgroup1, logMgroup2, logMgroup3, \
        z, Mstar, Rranges, angles = theta
    n_Rsat = (n_Rsat1, n_Rsat2, n_Rsat3)
    j = [(ni > 0) for ni in n_Rsat]
    Rsat = [Rsat[i] for i in j]
    n_Rsat = [n[i] for i, n in _izip(j, n_Rsat)]
    Msat = 10**_array([logMsat1, logMsat2, logMsat3])
    Mgroup = 10**_array([logMgroup1, logMgroup2, logMgroup3])
    csat = fc_sat * _cM_duffy08(Msat, z, h)
    cgroup = fc_group * _cM_duffy08(Mgroup, z, h)
    # some auxiliaries
    rho_m = density_average(z, h, Om, Ol)
    aux = 200*rho_m * 4*3.14159265/3
    # more parameters
    r200_sat = (Msat / aux) ** (1./3)
    rs_sat = r200_sat / csat
    sigma_sat = rs_sat * _delta(csat) * rho_m
    #r200_gr = (Mgroup / aux) ** (1./3)
    rs_gr = (Mgroup / aux) ** (1./3) / cgroup
    sigma_group = rs_gr * _delta(cgroup) * rho_m
    rsat = zeros(3)
    rsat_prior = zeros(3)
    for i, Rs, ni, rs, sigma, auxi in _izip(count(), Rsat, n_Rsat,
                                            rs_gr, sigma_group, aux):
        j = _array([(rsat_range >= Ri) & (rsat_range < rmax)
                    for Ri in Rs], dtype=float)
        nweighted = _transpose([g*n*_nfw_profile(rsat_range, rs, sigma, auxi)
                               for g, n in _izip(j, ni)])
        rdistrib = _array([n.sum() / jj.sum() if jj.sum() else 0
                           for n, jj in _izip(nweighted, j.T)])
        # draw random samples from a given normalized distribution
        rsat[i] = rsat_range[_digitize(_random(1),
                                       _cumsum(rdistrib/rdistrib.sum()))]
    # the Roche/Jacobi/Hill radius of satellites
    Mgroup_rsat = mass_enclosed(rsat/rs_gr, rs_gr, sigma_group)
    dlnM_dlnr = rsat**2 / (rs_gr+rsat)**2 / \
                (_log((rs_gr+rsat)/rs_gr) - rsat/(rs_gr+rsat))
    factor = (3 - dlnM_dlnr) * Mgroup_rsat
    # just starters for the loop
    rt = _array([0.5, 0.5, 0.5])
    rto = _array([0.1, 0.1, 0.1])
    while npall(_abs(rto-rt)/rt > 1e-2):
        rto = rt
        Msat_rt = _mass_total_sharp(rs_sat, rt/rs_sat, sigma_sat)
        rt = (Msat_rt / factor) ** (1./3.) * rsat
    rt = [_min(rti, r200) for rti, r200 in _izip(rt, r200_sat)]
    pointmass = [Mi / (3.14159265*(1e6*Ri[1:])**2)
                 for Mi, Ri in _izip(Mstar, R)]
    # satellite signals
    esd_sat = _array([pm + sat_profile(Ri[1:]/rsi, rti/rsi, si)
                      for pm, Ri, rsi, rti, si in _izip(pointmass, R, rs_sat,
                                                        rt, sigma_sat)])
    # group signals
    esd_group = _array([group_profile(x[0]/x[1], x[2]/x[1], x[3], x[4],
                                      x[5]/x[1], angles)
                        for x in _izip(R, rs_gr, Rsat, n_Rsat,
                                       sigma_group, Rranges)])
    lnPderived = _log(nfw_profile(rsat, rs_gr, sigma_group, aux)).sum()
    lnPderived -= _log(_array([romberg(nfw_profile, x[0].min(), rmax,
                                       args=(x[1],x[2], x[3]))
                               for x in _izip(Rsat, rs_gr,
                                              sigma_group, aux)])).sum()
    #out = [esd_sat + esd_group, esd_sat, esd_group,
           #rsat, rt, log10(Msat_rt+Mstar), lnPderived]
    out = [esd_sat + esd_group, esd_sat, esd_group,
           rsat, rt, Msat_rt+Mstar, lnPderived]
    return out
