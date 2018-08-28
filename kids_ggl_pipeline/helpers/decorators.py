"""KiDS-GGL Decorators

This module contains custom decorators used in KiDS-GGL for convenience

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


def logdist(func):
    """Account for whether observable and halo mass are in
    linear or logarithmic space

    Examples
    --------
    >>> @logdist
    >>> def lognormal(obs, M, sigma, obs_is_log=False):
    >>>     return np.exp(-((np.log10(obs/M)**2) / (2*sigma**2))) \
    >>>         / ((2*np.pi)**0.5 * sigma * obs * np.log(10))
    # if obs and M are in log space:
    >>> lognormal(9, 12, 0.3, obs_is_log=True)
    1.11390724e-31
    >>> # now use the linear-space quantities
    >>> lognormal(1e9, 1e12, 0.3, obs_is_log=False)
    1.11390724e-31
    """
    def decorated(*args, **kwargs):
        if 'obs_is_log' not in kwargs:
            kwargs['obs_is_log'] = False
        if kwargs['obs_is_log']:
            args = list(args)
            args[0] = 10**args[0]
            args[1] = 10**args[1]
        return func(*args, **kwargs)
    return decorated


def logfunc(func):
    """Make the output of a function linear or logarithmic as
    appropriate

    Decorator to return a quantity in linear or log space. In order to
    control this, the decorated function must have a kwarg
    ``return_log``. If said kwarg is not present, the function is
    returned in log space.

    Note that this assumes that the function is defined in log-space.
    For instance to use this decorator on a power law, define it as
    a straight line:

        > def powerlaw(x, a, b, return_log=True): return a + x*b

    not to

        > def powerlaw(x, a, b, return_log=True): return 10**a * x**b

    Examples
    --------
    >>> @logfunc
    >>> def f(x, a, b, return_log=True): return a + b*x
    # return a+b*x
    >>> f(2, 0, 1, return_log=True)
    2
    >>> # now return 10 to the power of that:
    >>> f(2, 0, 1, return_log=False)
    100
    """
    def decorated(*args, **kwargs):
        if 'return_log' not in kwargs:
            kwargs['return_log'] = False
        if kwargs['return_log']:
            return func(*args, **kwargs)
        return 10**func(*args, **kwargs)
    return decorated