#!/usr/bin/env python
"""
Galaxy-galaxy lensing EMCEE wrapper

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import emcee
import numpy as np
import os
import sys
from astropy.io.fits import BinTableHDU, Column, Header, PrimaryHDU
from itertools import count
from matplotlib import cm, pylab
from numpy import all as npall, array, append, concatenate, dot, inf, isnan, \
                  isfinite, log, log10, outer, pi, sqrt, squeeze, transpose, \
                  where, zeros
from os import remove
from os.path import isfile
from scipy import stats
from six import string_types
from time import ctime, time

# Python 2/3 compatibility
if sys.version_info[0] == 2:
    from itertools import izip as zip
    range = xrange
    input = raw_input

# local
from . import priors, sampling_utils
from ..helpers import io, plotting


def run(hm_options, options, args):

    function, parameters, names, prior_types, \
        nparams, repeat, join, starting, output = \
            hm_options

    val1, val2, val3, val4 = parameters[1][parameters[0].index('parameters')]
    setup = parameters[1][parameters[0].index('setup')]

    print_opening_msg(args, options)

    # identify fixed and free parameters
    jfree = np.array([(p in priors.free_priors) for p in prior_types])
    ndim = len(val1[where(jfree)])
    assert len(starting) == ndim, \
        'ERROR: Not all starting points defined for free parameters.'

    if args.cov or args.mock:
        #mock_options = parameters[1][parameters[0].index('mock')]
        mock(
            args, function, options, setup,# mock_options,
            parameters, join, starting, jfree, repeat, nparams)
        return

    print('Starting values =', starting)

    metadata, meta_names, fits_format = \
        sampling_utils.initialize_metadata(options, output)

    # some additional requirements of lnprob
    fail_value = sampling_utils.initialize_fail_value(metadata)
    # initialize
    lnprior = np.zeros(ndim)

    # load data files
    Ndatafiles = len(options['data'][0])
    assert Ndatafiles > 0, 'No data files found'
    # Rrange, angles are used in nfw_stack only
    R, esd, cov, Rrange, angles = io.load_data(options)
    #val1 = np.append(val1, [Rrange, angles])
    cov, icov, likenorm, esd_err, cov2d = cov
    # utility variables
    Nobsbins, Nrbins = esd.shape
    rng_obsbins = range(Nobsbins)
    rng_rbins = range(Nrbins)

    # are we just running a demo?
    if args.demo:
        demo(args, function, R, esd, esd_err, cov, icov, options, setup,
             prior_types, parameters, join, starting, jfree, repeat, nparams,
             names, lnprior, rng_obsbins, fail_value, Ndatafiles, array, dot,
             inf, outer, pi, zip)
        return

    # write header file
    hdrfile = io.write_hdr(options, function, parameters, names, prior_types)

    # initialize sampler
    sampler = emcee.EnsembleSampler(
        options['nwalkers'], ndim, lnprob, threads=options['threads'],
        args=(R,esd,icov,function,names,prior_types[jfree],
              parameters,nparams,join,jfree,repeat,lnprior,likenorm,
              rng_obsbins,fail_value,array,dot,inf,zip,outer,pi))

    # set up starting point for all walkers
    po = sample_ball(
        names, prior_types, jfree, starting, parameters,
        options['nwalkers'], ndim)

    # burn-in
    if options['nburn'] > 0:
        pos, prob, state, blobs = sampler.run_mcmc(po, options['nburn'])
        sampler.reset()
        print('{1}: {0} Burn-in steps finished'.format(
            options['nburn'], ctime()))
    else:
        pos = po

    # incrementally save output
    # this array contains lnprior, chi2, lnlike
    chi2 = [zeros(options['nwalkers']*options['nsteps']//options['thin'])
            for i in range(3)]
    nwritten = 0
    for i, result in enumerate(
            sampler.sample(pos, iterations=options['nsteps'],
                           thin=options['thin'])):
        if i > 0 and ((i+1)*options['nwalkers'] \
                      > options['nwalkers']*nwritten + options['update']):
            out = io.write_chain(
                sampler, options, chi2, names, jfree, output, metadata, i,
                nwritten, Nobsbins, fail_value)
            metadata, nwritten = out

    io.finalize_hdr(sampler, hdrfile)

    tmp = options['output'].replace('.fits', '.temp.fits')
    if os.path.isfile(options['output']):
        cmd = 'mv {0} {1}'.format(options['output'], tmp)
        print(cmd)
        os.system(cmd)
    print('Saving everything to {0}...'.format(options['output']))
    #write_to_fits(
    io.write_chain(
        sampler, options, chi2, names, jfree, output, metadata, i+1,
        nwritten, Nobsbins, fail_value)
    if os.path.isfile(tmp):
        os.remove(tmp)
    print('Everything saved to {0}!'.format(options['output']))

    return


def demo(args, function, R, esd, esd_err, cov, icov, options, setup,
         prior_types, parameters, join, starting, jfree, repeat, nparams,
         names, lnprior, rng_obsbins, fail_value, Ndatafiles,
         array, dot, inf, outer, pi, zip):
    to = time()
    lnlike, model = lnprob(
        starting, R, esd, icov, function, names, prior_types[jfree],
        parameters, nparams, join, jfree, repeat, lnprior, 0,
        rng_obsbins, fail_value, array, dot, inf, zip, outer, pi)
    print('\nDemo run took {0:.2e} seconds'.format(time()-to))
    chi2 = model[-2]
    if chi2 == fail_value[-2]:
        msg = 'Could not calculate model prediction. It is likely that one' \
              ' of the parameters is outside its allowed prior range.'
        raise ValueError(msg)
    dof = esd.size - starting.size - 1
    print()
    print(' ** chi2 = {0:.2f}/{1:d} **'.format(chi2, dof))
    print()
    # make plots
    output = '{0}_demo_{1}.pdf'.format(
        '.'.join(options['output'].split('.')[:-1]), setup['return'])
    plotting.signal(
        R, esd, esd_err, model=model[0], observable=setup['return'],
        output=output)
    if not args.no_demo_cov:
        output = output.replace('.pdf', '_cov.pdf')
        plotting.covariance(R, cov, output)

    return


def mock(args, function, options, setup, parameters, join, starting, jfree,
         repeat, nparams):
    """Generate mock observations given a set of cosmological and
    astrophysical parameters, as well as the observational setup

    Perhaps this function can be used to generate the covariance
    including astrophysical terms through the --cov cmd line option

    steps:
    * call halo.model just like in lnprob
    * call covariance module, and pass mock_options to it (it
      should be possible to disable this though)
    """
    R = np.array(
        [np.logspace(setup['logR_min'], setup['logR_max'],
                     setup['logR_bins'])])
    R, _ = sampling_utils.setup_integrand(R, options['precision'])
    p = update_parameters(starting, parameters, nparams, join, jfree, repeat)
    # run model!
    model = function(p, R, calculate_covariance=args.cov)
    # this is vestigial from when I integrated the offset
    # centrals for the satellite lensing measurements
    R = R[0,1:]
    # save
    if args.cov:
        pass
    else:
        outputs = io.write_signal(R, model[0], options, setup)
        print('Saved mock signal to {0}'.format(outputs))
    return


def lnprob(theta, R, esd, icov, function, names, prior_types,
           parameters, nparams, join, jfree, repeat, lnprior, likenorm,
           rng_obsbins, fail_value, array, dot, inf, zip, outer, pi):
    """
    Probability of a model given the data, i.e., log-likelihood of the data
    given a model, times the prior on the model parameters.

    Parameters
    ----------
        theta
            whatever *free* parameters are received by the model selected.
        R
            lens-source separations
        esd
            excess surface density at distances R
        icov
            inverse covariance
        function
            the model used to calculate the likelihood
        prior_types
            one value per parameter in *theta*, {'normal', 'uniform', 'fixed'}
        parameters
            list of length 4, containing all parameters of the models. In
            terms of the configuration file, these are [ingredients,
            observable, parameters, setup]; where `parameters` is another
            length-4 list
        jfree
            indices of the free values
        lnprior
            just a placeholder, should be an array with a length
            equal to theta, so that it doesn't have to be defined every
            time

    """
    lnprior_total = priors.calculate_lnprior(
        lnprior, theta, prior_types, parameters, jfree)
    if not isfinite(lnprior_total):
        return -inf, fail_value

    p = update_parameters(theta, parameters, nparams, join, jfree, repeat)
    # run model!
    model = function(p, R)
    # no covariance
    #chi2 = (((esd-model[0]) / esd_err) ** 2).sum()
    # full covariance included
    residuals = esd - model[0]
    chi2 = array([dot(residuals[m], dot(icov[m][n], residuals[n]))
                  for m in rng_obsbins for n in rng_obsbins]).sum()
    if not isfinite(chi2):
        return -inf, fail_value
    lnlike = -chi2/2. + likenorm
    model.append(lnprior_total)
    model.append(chi2)
    model.append(lnlike)
    return lnlike + lnprior_total, model



def print_opening_msg(args, options):
    print('\nRunning KiDS-GGL pipeline - sampler\n')
    if args.cov:
        print(' ** Calculating covariance matrix only **')
    elif args.demo:
        print(' ** Running demo only **')
    elif args.mock:
        print(' ** Generating mock observations **')
    elif isfile(options['output']) and not args.force_overwrite:
        msg = 'Warning: output file {0} exists. Overwrite? [y/N] '.format(
            options['output'])
        answer = input(msg)
        if len(answer) == 0:
            sys.exit()
        if answer.lower() not in ('y', 'yes'):
            sys.exit()
    if not args.demo:
        print('\n{0}: Started\n'.format(ctime()))
    return


def sample_ball(names, prior_types, jfree, starting, parameters, nw, ndim):
    """Create a ball around the starting values by drawing random
    samples from the prior
    """
    val1, val2, val3, val4 = parameters[1][parameters[0].index('parameters')]
    v1free = val1[where(jfree)]
    v2free = val2[where(jfree)]
    v3free = val3[where(jfree)]
    v4free = val4[where(jfree)]
    names_free = names[where(jfree)]
    prior_free = prior_types[where(jfree)]

    ball = np.zeros((nw, ndim))
    for n, p in enumerate(prior_free):
        ball[:,n] = priors.draw(
            p, (v1free[n],v2free[n]), (v3free[n],v4free[n]), size=nw)
    return ball


def update_parameters(theta, parameters, nparams, join, jfree, repeat):
    # update parameters
    v1 = parameters[1][parameters[0].index('parameters')][0].copy()
    v1[where(jfree)] = theta
    # joined parameters
    v1_list = list(v1)
    for j in join[::-1]:
        try:
            v1_list[j[0]] = np.array(v1[j], dtype=float)
        except ValueError:
            v1_list[j[0]] = v1[j]
    # repeat parameters
    for i, j in enumerate(repeat):
        if j != -1:
            v1_list[i] = v1_list[j]
    # do this all the way at the end to preserve indices
    for j in join:
        # remove parameters that have just been joined
        for i in j[1:][::-1]:
            v1_list.pop(i)
    # convert histograms (especially joined histograms) into np arrays
    for i, v in enumerate(v1_list):
        if not isinstance(v, string_types) and hasattr(v, '__iter__'):
            if hasattr(v[0], '__iter__'):
                v1_list[i] = [np.array(vi) for vi in v]
            v1_list[i] = np.array(v1_list[i], dtype=float)
    # join into sections and subsections as per the config file
    # by doing this I'm losing Rrange and angles, but I suppose
    # those should be defined in the config file as well.
    v1 = np.array([v1_list[sum(nparams[:i]):sum(nparams[:i+1])]
                   for i in range(len(nparams))])
    # note that by now we discard the other v's!
    # we don't want to overwrite the old list now that we've
    # changed one of its components
    p = [parameters[0], [p for p in parameters[1]]]
    p[1][p[0].index('parameters')] = v1
    return p



