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

def mass_nfw(theta, R, h=1, Om=0.315, Ol=0.685):
    # local variables are accessed much faster than global ones
    _array = array
    _cM_duffy08 = cM_duffy08
    _delta = delta
    _izip = izip
    profile, mass, zgal = theta
    