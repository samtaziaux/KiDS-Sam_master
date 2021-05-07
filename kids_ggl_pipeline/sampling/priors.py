from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import append, array, inf, isfinite, log, log10, pi, where
from scipy.stats import expon, t as tstudent


fixed_priors = ['array', 'fixed', 'function', 'read']

# note that all prior functions must receive 3 arguments even if not
# all are used!
lnprior_functions = {
    'exp': lambda v, v1, v2: log(expon.pdf(v)),
    'jeffreys': lambda v, v1, v2: -log(v),
    'lognormal': lambda v, v1, v2: \
        -v - (log(v)-v1)**2 / (2*v2**2) - log(2*pi*v2**2),
    'normal': lambda v, v1, v2: \
        -(v-v1)**2 / (2*v2**2) - log(2*pi*v2**2),
    'student': lambda v, v1, v2: log(tstudent.pdf(v, v1)),
    'uniform': lambda v, v1, v2: -log(v2-v1)}
# true number of arguments defined here separately
nargs = {'exp': 0, 'jeffreys': 0, 'lognormal': 2, 'normal': 2,
         'student': 1, 'uniform': 2}
for p in fixed_priors:
    nargs[p] = 0

free_priors = list(lnprior_functions.keys())
valid_priors = append(fixed_priors, free_priors)
valid_priors = append(valid_priors, 'repeat')


def calculate_lnprior(lnprior, theta, prior_types, parameters, jfree):
    val1, val2, val3, val4 = parameters[1][parameters[0].index('parameters')]
    v1free = val1[where(jfree)].flatten()
    v2free = val2[where(jfree)].flatten()
    v3free = val3[where(jfree)].flatten()
    v4free = val4[where(jfree)].flatten()
    if not isfinite(v1free.sum()):
        return -inf
    for pt, fp in lnprior_functions.items():
        j = (prior_types == pt)
        lnprior[j] = array(
            [fp(v, v1, v2) if v3 <= v <= v4 else -inf
             for v, v1, v2, v3, v4
             in zip(theta[j], v1free[j], v2free[j], v3free[j], v4free[j])])
        #print(theta[j], v1free[j], v2free[j], v3free[j], v4free[j], lnprior[j])
    return lnprior.sum()


def define_limits(prior, args):
    # just in case
    args = [float(a) for a in args]
    if prior == 'uniform':
        return args
    # 10-sigma
    if prior == 'normal':
        return [args[0]-10*args[1], args[0]+10*args[1]]
    if prior == 'lognormal':
        return [log(args[0])-10*log(args[1]), log(args[0])+10*log(args[1])]
    if prior == 'exp':
        # cumulative probability ~ 2e-9
        return [-10, 10]
    if prior == 'student':
        # cumulative probability ~ 3e-7
        return [-100000, 1000000]
    if prior == 'jeffreys':
        # no choice but to assume this number is of order 0-1
        # (but cannot be exactly zero)
        return [1e-10, 100]


def draw(prior, args, bounds=None, size=None):
    """Draw random numbers given a prior function and limiting values"""
    # large enough to beat poisson noise (though if the range is too
    # large there might still be trouble)
    bds = define_limits(prior, args)
    if bounds is None:
        bounds = bds
    if not np.isfinite(bounds[0]):
        bounds = [bds[0], bounds[1]]
    if not np.isfinite(bounds[1]):
        bounds = [bounds[0], bds[1]]
    rng = np.linspace(bounds[0], bounds[1], 10000)
    weights = np.exp(lnprior_functions[prior](rng, *args))
    if prior == 'uniform':
        weights = np.ones(rng.size) * weights
    else:
        # make sure it's both positive and normalized (but if we apply
        # this to the uniform pdf we're left with only zeros)
        weights -= weights.min()
    weights /= weights.sum()
    if size is None:
        return rng[np.digitize(np.random.random(1), np.cumsum(weights))][0]
    return rng[np.digitize(np.random.random(size), np.cumsum(weights))]


