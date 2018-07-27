from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
from numpy import array, exp, log, log10, newaxis, pi

if sys.version_info[0] == 2:
    from itertools import izip as zip


"""
Intrinsic scatter functions commonly used in the halo occupation distribution
"""

# don't forget to update this!
__all__ = ('lognormal', 'schechter_mod')

def lognormal(m, M, sigma):
    """Log-normal scatter

    Parameters
    ----------
    M : output of `occupation.Phi`

    """
    return array([exp(-((log10(m/Mi)**2) / (2*sigma**2))) \
                     / ((2*pi)**0.5 * sigma * m * log(10))
                 for Mi in M])


def schechter_mod(m, M, M12, alpha, phi_s, b):
    """Modified Schechter scatter (eq. 17 in van Uitert et al. 2016)

    This is generally used for satellite galaxies, and the mor
    would be `double_powerlaw_scaled`

    Parameters
    ----------
    M : output of `occupation.Phi`
    """
    phi_s = b[:,newaxis] * log10(M12)**arange(b.size)[:,newaxis]
    return array([phi_s_i/m * (m/Mi)**alpha * exp(-(m/Mi)**2)
                  for phi_s_i, Mi in zip(phi_s, M)])

