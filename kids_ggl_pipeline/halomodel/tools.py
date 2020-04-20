#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  tools.py
#
#  Copyright 2014 Andrej Dvornik <dvornik@dommel.strw.leidenuniv.nl>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
from hmf import MassFunction, fitting_functions as ff
import numpy as np
import matplotlib.pyplot as pl
import scipy
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d
import scipy.special as sp



"""
# Mathematical tools.
"""

def Integrate(func_in, x_array, **kwargs):
    """Simpson integration on fixed spaced data"""
    func_in = np.nan_to_num(func_in)
    result = trapz(func_in, x_array, **kwargs)
    return result


def Integrate1(func_in, x_array): # Gauss - Legendre quadrature!

    import scipy.integrate as intg

    #func_in = np.nan_to_num(func_in)

    c = interp1d(np.log(x_array), np.log(func_in), kind='slinear', \
                 bounds_error=False, fill_value=0.0)

    integ = lambda x: np.exp(c(np.log(x)))
    result = intg.fixed_quad(integ, x_array[0], x_array[-1])

    #print ("Integrating over given function.")
    return result[0]


def extrap1d(x, y, step_size, method):

    y_out = np.ones(len(x))


    # Step for stars at n = 76 = 0.006, for DM 0.003 and for gas same as for stars
    #~ step = len(x)*0.005#len(x)*0.0035#0.005!
    step = len(x)*step_size

    xi = np.log10(x)
    yi = np.log10(y)

    xs = xi
    ys = yi

    minarg = np.argmin(np.nan_to_num(np.gradient(yi)))
    grad = np.min(np.nan_to_num(np.gradient(yi)))

    if step < 3:
        step = 3

    #~ print np.exp(xi[minarg])
    #~ print np.exp(xi[minarg] - 0.04)

    if method == 1:
        yslice = ys[(minarg-step):(minarg):1]
        xslice = np.array([xs[(minarg-step):(minarg):1], np.ones(step)])

        w = np.linalg.lstsq(xslice.T, yslice)[0]

    elif method == 2:
        yslice = ys[(minarg-step):(minarg):1]
        xslice = xs[(minarg-step):(minarg):1]

        w = np.polyfit(xslice, yslice, 2)

    elif method == 3:
        #~ yslice = y[(minarg-20):(minarg):1] #60
        #~ xslice = x[(minarg-20):(minarg):1]
        yslice = y[(minarg-20):(minarg):1] #60
        xslice = x[(minarg-20):(minarg):1]

        from scipy.optimize import curve_fit
        def func(x, a, b, c):
            return a * (1.0+(x/b))**(-2.0)

        popt, pcov = curve_fit(func, xslice, yslice, p0=(y[0], x[minarg], 0.0))
        #print popt

    for i in range(len(x)):

        if i > minarg: #100! #65, 125
        #~ if i+start > minarg: #100! #65, 125
        #if i+50 > (np.where(grad < grad_val)[0][0]):

            """
            # Both procedures work via same reasoning, second one actually
            # fits the slope from all the points, so more precise!
            """

            if method == 1:
                y_out[i] = 10.0**(ys[minarg] + (xi[i] - xi[minarg])*(w[0]))

            elif method == 2:
                y_out[i] = 10.0**(w[2] + (xi[i])*(w[1]) + ((xi[i])**2.0)*(w[0]))

            elif method == 3:
                y_out[i] = (func(x[i], popt[0], popt[1], popt[2]))

        else:
            y_out[i] = 10.0**(yi[i])

    return y_out


def extrap2d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def fill_nan(a):
    not_nan = np.logical_not(np.isnan(a))
    indices = np.arange(len(a))
    if not_nan.sum() == 0:
        return a
    else:
        return np.interp(indices, indices[not_nan], a[not_nan])


def load_hmf(z, setup, cosmo_model, transfer_params):
    hmf = []
    rho_mean = np.zeros(z.shape[0])
    rho_mean_z = np.zeros(z.shape[0])
    for i, zi in enumerate(z):
        hmf.append(MassFunction(
            Mmin=setup['logM_min'], Mmax=setup['logM_max'],
            dlog10m=setup['mstep'],
            hmf_model=ff.Tinker10, delta_h=setup['delta'],
            delta_wrt=setup['delta_ref'], delta_c=1.686,
            cosmo_model=cosmo_model, z=zi,
            transfer_model=setup['transfer'], **transfer_params)
            )
        # shouldn't this be the mean density at the lens redshift?
        rho_mean[i] = hmf[i].mean_density0
        rho_mean_z[i] = hmf[i].mean_density # Add to return
    return hmf, rho_mean


def virial_mass(r, rho_mean, delta_halo):
    """
    Returns the virial mass of a given halo radius
    """

    return 4.0 * np.pi * r ** 3.0 * rho_mean * delta_halo / 3.0

def virial_radius(m, rho_mean, delta_halo):
    """
    Returns the virial radius of a given halo mass
    """

    return ((3.0 * m) / (4.0 * np.pi * rho_mean * delta_halo)) ** (1.0 / 3.0)


if __name__ == '__main__':
    main()




