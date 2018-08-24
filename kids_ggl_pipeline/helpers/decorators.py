"""KiDS-GGL Decorators

This module contains custom decorators used in KiDS-GGL for convenience

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


def logify_dist(func):
    """Account for whether observable and halo mass are in
    linear or logarithmic space

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


def logify_function(func):
    """Make the output of a function linear or logarithmic as
    appropriate

    Decorator to return a quantity in linear or log space. In order to
    control this, the decorated function must have a kwarg
    ``return_log``. If said kwarg is not present, the function is
    returned in log space.

    Examples
    --------
    >>> @logify
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